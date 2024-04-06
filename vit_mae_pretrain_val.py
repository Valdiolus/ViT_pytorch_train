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
mixed_precision=False

#for time correction
gmt_dst = 2

BATCH_SIZE = 128 if mixed_precision else 5
workers = 8
input_size = 224


def train(model, dataloaders):

    if device == torch.device("cuda"):
        model = torch.compile(model)

    time_beginning = time.time()

    model.eval()

    running_loss = 0.0
    processed_data = 0
    saved_epochs = 0



    with tqdm(dataloaders, leave=False, desc=f"val iter") as tepoch:
        #tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
        for data_iter_step, data in enumerate(tepoch):
            
            inputs, labels = data

            if train_on_gpu:
                inputs = inputs.to(device)
                labels = labels.to(device)

            if mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss, pred, mask = model(inputs, mask_ratio=0.75) #logits, mask, ids_restore, hidden_states=None, attentions=None
            else:
                loss, pred, mask = model(inputs, mask_ratio=0.75)

            loss_value = loss.item()
            
            processed_data += inputs.size(0)
            running_loss += loss_value * inputs.size(0) #needed???
            #print("loss:", loss_value, "processed_data:", processed_data, "running_loss:", running_loss)

            tepoch.set_postfix(loss=running_loss/processed_data)

    #calculate avaerage loss for epoch
    running_loss /= processed_data
        
    print("Total train time:", time.time() - time_beginning)
    print("Total loss", running_loss)
    #Print log each epoch
    #print("\nEpoch %s/%s" % (epoch+1, num_epochs), "train acc", "{:.4f}".format(accuracy['train'][epoch]), "train loss", "{:.4f}".format(losses['train'][epoch]), 
    #                                                "val acc", "{:.4f}".format(accuracy['val'][epoch]), "val loss", "{:.4f}".format((losses['val'][epoch])))

if __name__ == '__main__':

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
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    imagenet_train_data = torchvision.datasets.ImageFolder(os.path.join(imagenet_dir, 'val'), transform=data_transforms)
    data_loaders = torch.utils.data.DataLoader(imagenet_train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
    dataset_sizes = len(imagenet_train_data)
    class_names = imagenet_train_data.classes
    print("val", dataset_sizes)
    #train_features, train_labels = next(iter(data_loaders['val']))

    #print("Model:", MODEL_PATH)

    #model = ViTMAEForPreTraining.from_pretrained('./vit-mae-base')
    model = models_mae.__dict__['mae_vit_tiny_patch16_dec96d1b'](norm_pix_loss=True)
    model.to(device) 

    # base_test/checkpoint-799 loss=0.405
    # tiny_test/checkpoint-0 loss=0.8
    checkpoint = torch.load('./tiny_test/checkpoint-0.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    train(model, data_loaders)