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
import cv2
import wandb

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import models_mae 
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
import utils.misc as misc
import utils.lr_sched as lr_sched

import argparse

#use fp16
mixed_precision=True

#for time correction
gmt_dst = 2

BATCH_SIZE = 128 if mixed_precision else 5
ACCUM_ITER = int(4096/BATCH_SIZE)
workers = 8
EPOCHS = 100
WARMUP_EPOCHS = 1
BEST_LOSS = 100000.0
input_size = 224
init_lr = 0.05
init_weight_decay = 0.05

load_from_file = 0
saved_model = ''
wandb_saved_id = 0
norm_pixel_loss=True
mask_ratio=0.75

model_type = 'base'

PATH_TO_SAVE = './runs'

wandb_log_interval = 5 if mixed_precision else 10

wandb_config = {"batch_size": BATCH_SIZE,
                "num_workers": workers,
                "input size": input_size,
                "epochs": EPOCHS,
                "pin_memory": False,  
                "precision": 16 if mixed_precision else 32,
                "optimizer": "Adam",
                "lr": init_lr,
                "weight_decay": init_weight_decay,
                "aug": "no",
                "warmup_epochs": WARMUP_EPOCHS
                }

def train(args, model, dataloaders, optimizer, loss_scaler, num_epochs=10):

    if device == torch.device("cuda"):
        model = torch.compile(model)

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()

    time_beginning = time.time()

    model.train(True)

    running_loss = 0.0
    processed_data = 0
    saved_epochs = 0

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if args.resume:
        saved_epochs = args.start_epoch
        print("wandb id:", args.wandb_id)
        train_path = os.path.join(PATH_TO_SAVE, saved_model)
        wandb.init(id=args.wandb_id, resume=True, project="ViT MAE Pretrain", name = args.resume, config=wandb_config)
        print("Resume training:", saved_model)
    else:
        time_struct = time.gmtime()
        time_now = str(time_struct.tm_year)+'-'+str(time_struct.tm_mon)+'-'+str(time_struct.tm_mday)+'_'+str(time_struct.tm_hour+gmt_dst)+'-'+str(time_struct.tm_min)+'-'+str(time_struct.tm_sec)
        print("New train:",  model_type+'_pt_'+time_now)
        wandb_saved_id = wandb.util.generate_id()
        args.wandb_id = wandb_saved_id
        wandb.init(id=wandb_saved_id, resume=True, project="ViT MAE Pretrain", name = model_type+'_pt_'+time_now, config=wandb_config)
        train_path = os.path.join(PATH_TO_SAVE, model_type+'_pt_'+time_now)
        if not os.path.exists(train_path):
            os.makedirs(train_path)

    for epoch in range(saved_epochs, num_epochs):
        with tqdm(dataloaders, leave=False, desc=f"train iter") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
            for data_iter_step, data in enumerate(tepoch):
                if data_iter_step % accum_iter == 0:
                    lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(dataloaders) + epoch, args)
                
                inputs, labels = data

                if train_on_gpu:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                if mixed_precision:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        loss, pred, mask = model(inputs, mask_ratio=mask_ratio) #logits, mask, ids_restore, hidden_states=None, attentions=None
                else:
                    loss, pred, mask = model(inputs, mask_ratio=mask_ratio)

                loss_value = loss.item()
                
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    quit()

                #save_inference_images(inputs, outputs)
                loss /= accum_iter
                if mixed_precision:
                    loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
                else:
                    loss.backward()
                    optimizer.step()
                
                if (data_iter_step + 1) % accum_iter == 0:
                    optimizer.zero_grad()
                    #print("zero grad")

                lr = optimizer.param_groups[0]["lr"]
                
                processed_data += inputs.size(0)
                running_loss += loss_value * inputs.size(0) #needed???
                #print("loss:", loss_value, "processed_data:", processed_data, "running_loss:", running_loss)

                tepoch.set_postfix(loss=loss_value)

        #calculate avaerage loss for epoch
        running_loss /= processed_data
        if running_loss < args.best_par:
            save_best = True
            args.best_par = running_loss
        else:
            save_best = False
        misc.save_model(args, epoch, model, optimizer, loss_scaler, train_path, save_best)
        print("Epoch:", epoch+1, "train loss:", running_loss, "lr:", lr)
        wandb.log({"train": {"loss": running_loss, "lr": lr}})
        processed_data = 0
        running_loss = 0
        
    print("Total train time:", time.time() - time_beginning)
    print("Total loss", running_loss)
    #Print log each epoch
    #print("\nEpoch %s/%s" % (epoch+1, num_epochs), "train acc", "{:.4f}".format(accuracy['train'][epoch]), "train loss", "{:.4f}".format(losses['train'][epoch]), 
    #                                                "val acc", "{:.4f}".format(accuracy['val'][epoch]), "val loss", "{:.4f}".format((losses['val'][epoch])))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ViT mae pretraining")
    parser.add_argument("--resume", required=False, help="run id to resume")
    parser.add_argument("--start_epoch", default=0, type=int, help="resumed epoch")
    parser.add_argument("--wandb_id", default=1, type=int, help="wandb id to resume")
    parser.add_argument("--best_par", default=BEST_LOSS, type=float, help="previous best loss")
    parser.add_argument("--epochs", default=EPOCHS, type=int, help="number of warmup_epochs")
    parser.add_argument("--warmup_epochs", default=WARMUP_EPOCHS, type=int, help="number of warmup_epochs")
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=ACCUM_ITER, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    args, unknown = parser.parse_known_args()

    #should be 4096
    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
        #args.lr = args.blr * 4096 / 256

    saved_model = args.resume
    if saved_model is not None:
        load_from_file = True

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
        cudnn.benchmark = True # only if input size is not changing
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
    data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    imagenet_train_data = torchvision.datasets.ImageFolder(os.path.join(imagenet_dir, 'train'), transform=data_transforms)
    data_loaders = torch.utils.data.DataLoader(imagenet_train_data, batch_size=args.batch_size, shuffle=True, num_workers=workers)
    dataset_sizes = len(imagenet_train_data)
    class_names = imagenet_train_data.classes
    print("train", dataset_sizes)
    #train_features, train_labels = next(iter(data_loaders['val']))

    #print("Model:", MODEL_PATH)

    #model = ViTMAEForPreTraining.from_pretrained('./vit-mae-base')
    if model_type == 'base':
        model = models_mae.__dict__['mae_vit_base_patch16_dec512d8b'](norm_pix_loss=norm_pixel_loss)
    
    model.to(device) 

    #vit-base-patch16-224 - Final top5: 0.959 top1: 0.813
    param_groups = optim_factory.add_weight_decay(model, init_weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model=model, optimizer=optimizer, loss_scaler=loss_scaler)

    train(args, model, data_loaders, optimizer, loss_scaler, EPOCHS)