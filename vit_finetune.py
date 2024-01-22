import os
from tqdm.autonotebook import tqdm, trange

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import math

from torchsummary import summary

from PIL import Image
from transformers import ViTForImageClassification
from transformers import ViTImageProcessor, AutoImageProcessor, ViTMAEForPreTraining
import requests
import timm.optim.optim_factory as optim_factory
from timm.data.mixup import Mixup
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import cv2
import wandb

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import models_mae 
import models_vit
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import utils.misc as misc
import utils.lr_sched as lr_sched
import utils.pos_embed as pos_embed
import utils.lr_decay as lrd

import argparse
import PIL

#use fp16
mixed_precision=True

#for time correction
gmt_dst = 2

BATCH_SIZE = 64 if mixed_precision else 16
ACCUM_ITER = int(1024/BATCH_SIZE)
workers = 8
EPOCHS = 100
WARMUP_EPOCHS = 5
BEST_ACC = 0.0
MIXUP_DEFAULT = 0
CUTMIX_DEFAULT = 0
input_size = 224
init_lr = 5e-4
init_weight_decay = 0.05

model_type = 'base'

PATH_TO_SAVE = './runs'

wandb_log_interval = 5 if mixed_precision else 10

wandb_config = {"batch_size": BATCH_SIZE,
                "num_workers": workers,
                "input size": input_size,
                "epochs": EPOCHS,
                "precision": 16 if mixed_precision else 32,
                "optimizer": "AdamW",
                "blr": init_lr,
                "weight_decay": init_weight_decay,
                "aug": "no",
                "warmup_epochs": WARMUP_EPOCHS,
                "mixup": MIXUP_DEFAULT,
                "cutmix": CUTMIX_DEFAULT,
                }

def train(args, model, dataloaders, optimizer, criterion, loss_scaler, num_epochs=10, mixup_fn=None):

    if device == torch.device("cuda"):
        model = torch.compile(model)

    running_loss = 0.0
    processed_data = 0
    best_top1 = 0.0
    best_top5 = 0.0
    saved_epochs = 0

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if args.resume:
        saved_epochs = args.start_epoch
        print("wandb id:", args.wandb_id)
        train_path = os.path.join(PATH_TO_SAVE, args.resume)
        wandb.init(id=args.wandb_id, resume=True, project="ViT finetune", name = args.resume, config=wandb_config)
        print("Resume training:", args.resume)
    else:
        time_struct = time.gmtime()
        time_now = str(time_struct.tm_year)+'-'+str(time_struct.tm_mon)+'-'+str(time_struct.tm_mday)+'_'+str(time_struct.tm_hour+gmt_dst)+'-'+str(time_struct.tm_min)+'-'+str(time_struct.tm_sec)
        print("New train:",  model_type+'_ft_'+time_now)
        wandb_saved_id = wandb.util.generate_id()
        args.wandb_id = wandb_saved_id
        wandb.init(id=wandb_saved_id, resume=True, project="ViT finetune", name = model_type+'_ft_'+time_now, config=wandb_config)
        train_path = os.path.join(PATH_TO_SAVE, model_type+'_ft_'+time_now)
        if not os.path.exists(train_path):
            os.makedirs(train_path)

    time_beginning = time.time()
    for epoch in range(saved_epochs, num_epochs):
        epoch_losses_train = 0
        epoch_losses_val = 0
        epoch_accuracy_train = 0
        epoch_accuracy_val = 0

        for phase in ["train", "val"]:
            if phase == 'train':
                model.train(True)
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            processed_data = 0
            iter_top1 = 0
            iter_top5 = 0
            correct_top5 = 0
            correct_top1 = 0
            #for inputs, labels in dataloader:
            with tqdm(dataloaders[phase], leave=False, desc=f"{phase} iter") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}, {phase} iter")
                for data_iter_step, data in enumerate(tepoch):
                    if data_iter_step % accum_iter == 0:
                        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(dataloaders) + epoch, args)
                    
                    inputs, labels = data
                    #print(inputs.size(), labels.size())

                    if train_on_gpu:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                    if mixup_fn is not None:
                        inputs, labels_long = mixup_fn(inputs, labels)

                    if mixed_precision:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            outputs = model(inputs) #logits, mask, ids_restore, hidden_states=None, attentions=None
                            if mixup_fn is not None:
                                loss = criterion(outputs, labels_long)
                            else:
                                loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        if mixup_fn is not None:
                            loss = criterion(outputs, labels_long)
                        else:
                            loss = criterion(outputs, labels)
                    
                    loss_value = loss.item()
                    preds = torch.argmax(outputs, 1)
                    
                    if not math.isfinite(loss_value):
                        print("Loss is {}, stopping training".format(loss_value))
                        quit()

                    loss /= accum_iter
                    #lr = optimizer.param_groups[0]["lr"]
                    min_lr = 10.
                    max_lr = 0.
                    for group in optimizer.param_groups:
                        min_lr = min(min_lr, group["lr"])
                        max_lr = max(max_lr, group["lr"])

                    processed_data += inputs.size(0)
                    running_loss += loss_value * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_accuracy=(running_corrects/processed_data)

                    if phase == "train":
                        if mixed_precision:
                            loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                parameters=model.parameters(), create_graph=False,
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
                        else:
                            loss.backward()
                            optimizer.step()
                        
                        if (data_iter_step + 1) % accum_iter == 0:
                            optimizer.zero_grad()
                            #print("zero grad")
                                                    
                        tepoch.set_postfix(loss=loss_value, accuracy=running_accuracy.item())

                    if phase == "val":
                        iter_top1 = 0
                        iter_top5 = 0

                        iter_top1 += torch.sum(preds == labels.data)
                        iter_top5 += iter_top1

                        for _ in range(4):
                            for i in range(inputs.size(0)):
                                outputs[i,preds[i]] = -1000
                            preds = torch.argmax(outputs, 1)
                            iter_top5 += torch.sum(preds == labels.data)
                        
                        correct_top1 += iter_top1
                        correct_top5 += iter_top5
                        tepoch.set_postfix(loss=loss_value, top1=(correct_top1 / processed_data).item(), top5=(correct_top5 / processed_data).item())
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'train':
                epoch_losses_train = epoch_loss
                epoch_accuracy_train = epoch_acc
                wandb.log({"train": {"loss": epoch_loss, "accuracy": epoch_acc, "lr": max_lr}})
                
            if phase == 'val':
                epoch_losses_val = epoch_loss
                correct_top1 = correct_top1.item() / processed_data
                correct_top5 = correct_top5.item() / processed_data

                if correct_top1 > args.best_par:
                    save_best = True
                    args.best_par = correct_top1
                else:
                    save_best = False

                misc.save_model(args, epoch, model, optimizer, loss_scaler, train_path, save_best)
                wandb.log({"val": {"loss": epoch_losses_val, "top1 accuracy": correct_top1, "top5 accuracy": correct_top5}})

        processed_data = 0
        print("\nEpoch %s/%s" % (epoch+1, num_epochs), "train acc", "{:.4f}".format(epoch_accuracy_train), "train loss", "{:.4f}".format(epoch_losses_train), 
                                                       "val loss", "{:.4f}".format(epoch_losses_val), "val top1:", correct_top1, "val top5:", correct_top5)
        
    time_elapsed = time.time() - time_beginning
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed // 60) - (time_elapsed // 3600)*60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(args.best_par))
    print("Final top1:", correct_top1, "top5:", correct_top5)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ViT mae pretraining")
    parser.add_argument("--resume", required=False, help="run id to resume")
    parser.add_argument("--finetune", required=False, help="run id to resume")
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    parser.add_argument("--start_epoch", default=0, type=int, help="resumed epoch")
    parser.add_argument("--wandb_id", default=1, type=int, help="wandb id to resume")
    parser.add_argument("--best_par", default=BEST_ACC, type=float, help="previous best loss")
    parser.add_argument("--epochs", default=EPOCHS, type=int, help="number of warmup_epochs")
    parser.add_argument("--warmup_epochs", default=WARMUP_EPOCHS, type=int, help="number of warmup_epochs")

    #LR
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=init_lr, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    
    #Batch size
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=ACCUM_ITER, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=init_weight_decay,
                        help='weight decay (default: 0.05)')

    #Mixup params
    parser.add_argument('--mixup', type=float, default=MIXUP_DEFAULT,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=CUTMIX_DEFAULT,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    
    args, unknown = parser.parse_known_args()

    #should be 4096
    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
        #args.lr = args.blr * 1024 / 256

    wandb_config["lr"] = args.lr

    #print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("GPU Batch size", args.batch_size)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    #Check if GPU is enable
    # datapath file example - datasets/imagenet/ILSVRC/Data/CLS-LOC
    if torch.cuda.is_available():
        print('CUDA is available! Training on GPU ...')
        train_on_gpu = True
        mixed_precision=True
        #cudnn.benchmark = True # only if input size is not changing
        device = torch.device("cuda")
        with open("cuda_datapath_imagenet.txt", 'r') as f:
            imagenet_dir = f.read()
            print("cuda imagenet path:", imagenet_dir)
    else:
        if torch.backends.mps.is_available():
            print('GPU on M1 MAC is available! Training on GPU ...')
            train_on_gpu = True
            device = torch.device("mps")
            with open("mps_datapath_imagenet.txt", 'r') as f:
                imagenet_dir = f.read()
                print("mps imagenet path:", imagenet_dir)
        else:
            print('CUDA and MPS are not available. Training on CPU ...')
            train_on_gpu = False
            with open("cuda_datapath_imagenet.txt", 'r') as f:
                imagenet_dir = f.read()
                print("cuda imagenet path:", imagenet_dir)

    #processor = AutoImageProcessor.from_pretrained('./vit-mae-base')
    #image_mean, image_std = processor.image_mean, processor.image_std
    #size = processor.size["height"]
    #print("Image size:", size)
    # 224/16 = 14
    # 14*14 = 196 - number of patches
    
    #Loading dataset - like Imagenet, where 1k folders with each classes
    data_transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    data_transforms_val = transforms.Compose([
        transforms.Resize(int(input_size*256/224)), #, interpolation=PIL.Image.BICUBIC
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    imagenet_data = {x: torchvision.datasets.ImageFolder(os.path.join(imagenet_dir, x), transform=data_transforms_train if x=='train' else data_transforms_val)
                    for x in ['train', 'val']}
    data_loaders = {x: torch.utils.data.DataLoader(imagenet_data[x], batch_size=args.batch_size, shuffle=True, num_workers=workers, drop_last=True if x=='train' else False)
                    for x in ['train', 'val']}
    dataset_sizes = {x: len(imagenet_data[x]) for x in ['train', 'val']}
    class_names = imagenet_data['train'].classes
    iter_per_epoch = int(len(imagenet_data['train'])/args.batch_size)
    print(dataset_sizes)
    #pin_memory
    #train_features, train_labels = next(iter(data_loaders['val']))

    #print("Model:", MODEL_PATH)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    #model = ViTMAEForPreTraining.from_pretrained('./vit-mae-base')
    model = models_vit.__dict__['vit_base_patch16'](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed.interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device) 
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    #print("Model = %s" % str(model))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    #vit-base-patch16-224 - Final top5: 0.959 top1: 0.813
    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)

    train(args, model, data_loaders, optimizer, criterion, loss_scaler, EPOCHS, mixup_fn=mixup_fn)