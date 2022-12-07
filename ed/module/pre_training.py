import sys
import os
import matplotlib.pyplot as plt
# import PIL
# import cv2
# import time
# import skimage
import numpy as np
# from sklearn.model_selection import train_test_split

from torchsummary import summary
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as vision_models

# pip install fastai==2.4
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18, resnet101
from fastai.vision.models.unet import DynamicUnet

from tqdm import tqdm

# module import 
from utils import *
from dataset import *
from transforms import *
from model_Unet import *

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_colab = None

print(f"device: {device}")

home = "./input"
os.listdir("./input/ab/ab")


def build_fastai_model(in_channels=1, out_channels=2, image_shape=(224, 224)):
    model_body = create_body(resnet18(), n_in=in_channels, cut=-2)
    model = DynamicUnet(encoder=model_body, n_out=out_channels, img_size=image_shape)
    return model.to(device)

def build_fastai_model_50(in_channels=1, out_channels=2, image_shape=(224, 224)):
    model_body = create_body(resnet101(), n_in=in_channels, cut=-2)
    model = DynamicUnet(encoder=model_body, n_out=out_channels, img_size=image_shape)
    return model.to(device)    


def pretrain_generator(net_G, train_dl, opt, criterion=nn.L1Loss(), epochs=20):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for x, y in tqdm(train_dl):
            L, ab = x.to(device), y.to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), L.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")

# print data
# ab_train.shape, l_train.shape, ab_test.shape, l_test.shape

input_shape = [224, 224]
batch_size = 32
num_examples = -1
device=  "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
epochs = 100
plot_freq = 5


ab_train, ab_test, l_train, l_test = load_data("input", channels_first=True)
    
input_transforms = [transform_expand_dim(axis=2),
                    to_channel_first,
                   transform_divide(255.0)
                   ]
output_transforms = [
                    to_channel_first,
                   transform_divide(255.0)
                   ]

ds_train = DatasetImg(l_train[:num_examples], ab_train[:num_examples], input_transforms=input_transforms, output_transforms=output_transforms)
train_dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

ds_test = DatasetImg(l_test[:num_examples], ab_test[:num_examples], input_transforms=input_transforms, output_transforms=output_transforms)
test_dl = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

        
net_G = build_fastai_model(in_channels=1, out_channels=2, image_shape=(224,224))
opt = torch.optim.Adam(net_G.parameters(), lr=1e-4)
criterion = nn.L1Loss()        
pretrain_generator(net_G, train_dl, opt, criterion, 20)
torch.save(net_G.state_dict(), "res18-unet2.pt")



