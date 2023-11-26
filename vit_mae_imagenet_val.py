import os
from tqdm.autonotebook import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time

from torchsummary import summary

from PIL import Image
from transformers import ViTForImageClassification
from transformers import ViTImageProcessor, AutoImageProcessor, ViTMAEForPreTraining
import requests

import cv2

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

batch_size = 128
workers = 4
EPOCHS = 1
n_classes = 1000
#MODEL_PATH = './mae_finetuned_vit_base.pth'

def unpatchify(x, patch_size=16):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = patch_size # self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

def save_inference_images(inputs, outputs):
        loss = outputs.loss
        mask = outputs.mask # [bs, 196]
        ids_restore = outputs.ids_restore # [bs, 196]
        logits = outputs.logits # [bs, 196, 768] 768 - 16*16*3
        #print("ids_restore", ids_restore.shape, torch.min(ids_restore), torch.max(ids_restore))
        #print("unpatchify", unpatchify(logits).shape)

        #Image.fromarray((output_pic * 255).astype(np.uint8))
        #output to opencv

        input_pic = torch.einsum('nchw->nhwc', inputs)[0].cpu().numpy()

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, 16**2 *3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach()[0].cpu().numpy()
        mask_pic = cv2.cvtColor((input_pic*(1-mask) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        input_pic = cv2.cvtColor((input_pic * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        output_pic = torch.einsum('nchw->nhwc', unpatchify(logits))[0].cpu().numpy()
        output_pic = cv2.cvtColor(((output_pic*mask) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        output_pic = output_pic + mask_pic

        #print("mask", mask.shape)
        
        #combine pics
        both_pics = np.concatenate((input_pic, mask_pic, output_pic), axis=1)
        cv2.imwrite("both_pics.jpg", both_pics)

def val(model, dataloaders):
    model.to(device) 

    time_beginning = time.time()

    phase = "val"

    model.eval()

    running_loss = 0.0
    processed_data = 0

    #for inputs, labels in dataloader:
    with tqdm(dataloaders[phase], leave=False, desc=f"{phase} iter") as tepoch:
        #tepoch.set_description(f"{phase} iter")
        for data in tepoch:
            inputs, labels = data

            if train_on_gpu:
                inputs = inputs.to(device)
                labels = labels.to(device)
            #print(inputs[0])
            with torch.no_grad():
                    outputs = model(inputs) #logits, mask, ids_restore, hidden_states=None, attentions=None

            loss = outputs.loss
            mask = outputs.mask # [bs, 196]
            ids_restore = outputs.ids_restore # [bs, 196]
            logits = outputs.logits # [bs, 196, 768] 768 - 16*16*3
            #print("ids_restore", ids_restore.shape, torch.min(ids_restore), torch.max(ids_restore))
            #print("unpatchify", unpatchify(logits).shape)

            save_inference_images(inputs, outputs)
            
            processed_data += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            #correct_top1 += iter_top1
            #correct_top5 += iter_top5

            tepoch.set_postfix(loss=running_loss / processed_data)
    
    #correct_top1 = correct_top1.item() / processed_data
    #correct_top5 = correct_top5.item() / processed_data
    print("Total loss", running_loss / processed_data)
    #Print log each epoch
    #print("\nEpoch %s/%s" % (epoch+1, num_epochs), "train acc", "{:.4f}".format(accuracy['train'][epoch]), "train loss", "{:.4f}".format(losses['train'][epoch]), 
    #                                                "val acc", "{:.4f}".format(accuracy['val'][epoch]), "val loss", "{:.4f}".format((losses['val'][epoch])))

if __name__ == '__main__':
    #Check if GPU is enable
    # datapath file example - datasets/imagenet/ILSVRC/Data/CLS-LOC
    if torch.cuda.is_available():
        print('CUDA is available!  Validating on GPU ...')
        train_on_gpu = True
        device = torch.device("cuda")
        with open("cuda_datapath_imagenet.txt", 'r') as f:
            imagenet_dir = f.read()
            print("cuda imagenet path:", imagenet_dir)
    else:
        if torch.backends.mps.is_available():
            print('GPU on M1 MAC is available!  Validating on GPU ...')
            train_on_gpu = True
            device = torch.device("mps")
            with open("mps_datapath_imagenet.txt", 'r') as f:
                imagenet_dir = f.read()
                print("mps imagenet path:", imagenet_dir)
        else:
            print('CUDA and MPS are not available.  Validating on CPU ...')
            with open("cuda_datapath_imagenet.txt", 'r') as f:
                imagenet_dir = f.read()
                print("cuda imagenet path:", imagenet_dir)

    processor = AutoImageProcessor.from_pretrained('./vit-mae-base')

    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    print("Image size:", size)
    # 224/16 = 14
    # 14*14 = 196 - number of patches

    #Loading dataset - like Imagenet, where 1k folders with each classes
    data_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        #transforms.Normalize(mean=image_mean, std=image_std)
        ])
    imagenet_data = {x: torchvision.datasets.ImageFolder(os.path.join(imagenet_dir, x), transform=data_transforms)
                    for x in ['train', 'val']}
    


    data_loaders = {x: torch.utils.data.DataLoader(imagenet_data[x], batch_size=batch_size, shuffle=True, num_workers=workers)
                    for x in ['train', 'val']}
    dataset_sizes = {x: len(imagenet_data[x]) for x in ['train', 'val']}
    class_names = imagenet_data['train'].classes
    print(dataset_sizes['val'])
    #train_features, train_labels = next(iter(data_loaders['val']))

    #print("Model:", MODEL_PATH)

    model = ViTMAEForPreTraining.from_pretrained('./vit-mae-base')
    #print(model)
    #quit()
    #vit-base-patch16-224 - Final top5: 0.959 top1: 0.813

    val(model, data_loaders)