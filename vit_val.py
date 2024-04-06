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
from transformers import ViTImageProcessor
import requests

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

batch_size = 128
workers = 4
EPOCHS = 1
n_classes = 1000
#MODEL_PATH = './mae_finetuned_vit_base.pth'

def val(model, dataloaders):
    model.to(device) 

    time_beginning = time.time()

    phase = "val"

    model.eval()

    iter_top1 = 0
    iter_top5 = 0
    correct_top5 = 0
    correct_top1 = 0
    processed_data = 0

    #for inputs, labels in dataloader:
    with tqdm(dataloaders[phase], leave=False, desc=f"{phase} iter") as tepoch:
        #tepoch.set_description(f"{phase} iter")
        for data in tepoch:
            inputs, labels = data

            if train_on_gpu:
                inputs = inputs.to(device)
                labels = labels.to(device)

            with torch.no_grad():
                    outputs = model(inputs).logits

            preds = torch.argmax(outputs, 1)

            iter_top1 = 0
            iter_top5 = 0

            iter_top1 += torch.sum(preds == labels.data)
            iter_top5 += iter_top1

            for _ in range(4):
                for i in range(inputs.size(0)):
                    outputs[i,preds[i]] = -1000
                preds = torch.argmax(outputs, 1)
                iter_top5 += torch.sum(preds == labels.data)
            
            processed_data += inputs.size(0)
            correct_top1 += iter_top1
            correct_top5 += iter_top5

            tepoch.set_postfix(top1=(correct_top1 / processed_data).item(), top5=(correct_top5 / processed_data).item())
    
    correct_top1 = correct_top1.item() / processed_data
    correct_top5 = correct_top5.item() / processed_data
    print("Final top5:", correct_top5, "top1:", correct_top1)
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

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    #Loading dataset - like Imagenet, where 1k folders with each classes
    data_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
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
    #id2label=id2label,label2id=label2id - specify the numbers of output neurons - for training
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    #print(model)
    #quit()

    #vit-base-patch16-224 - Final top5: 0.959 top1: 0.813

    val(model, data_loaders)