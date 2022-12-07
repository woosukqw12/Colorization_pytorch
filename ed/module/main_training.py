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
    # DynamicUnet (encoder, n_out, img_size, blur=False, blur_final=True,
    #           self_attention=False, y_range=None, last_cross=True,
    #           bottle=False, act_cls=<class
    #           'torch.nn.modules.activation.ReLU'>, init=<function
    #           kaiming_normal_>, norm_type=None, **kwargs)
    return model.to(device)

def build_fastai_model_50(in_channels=1, out_channels=2, image_shape=(224, 224)):
    model_body = create_body(resnet101(), n_in=in_channels, cut=-2)
    model = DynamicUnet(encoder=model_body, n_out=out_channels, img_size=image_shape)
    return model.to(device)    


input_shape = (224, 224)
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

        
                
# net_G = build_fastai_model(in_channels=1, out_channels=2, image_shape=(224,224))
# opt = torch.optim.Adam(net_G.parameters(), lr=1e-4)
# criterion = nn.L1Loss()        
# pretrain_generator(net_G, train_dl, opt, criterion, 20)
# torch.save(net_G.state_dict(), "res18-unet.pt")

model = build_fastai_model(in_channels=1, out_channels=2, image_shape=(224,224))
model.load_state_dict(torch.load("res18-unet.pt", map_location=device))
# train_model(model, train_dl, 20)


def train(model, loss_fn, optimizer, dataloader_train, device):
    # model = model.to(device)
    N = len(dataloader_train.dataset)
    n_batch = int(N / dataloader_train.batch_size)
    
    model.train()
    losses = []
    for batch, (x,y) in enumerate(dataloader_train):
        # print(f"x: {x} type: {type(x)}\ny: {y} \n")
        x = x.to(device).float()
        y = y.to(device).float()
        # print("test() x.shape", x.shape)
        pred = model(x)
        loss = loss_fn(pred, y)
        losses.append(loss.item())
        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
        
        sys.stdout.write(f"\rTrain: {batch+1}/{n_batch} loss:{loss}")
        sys.stdout.flush()
        # break
    avg_loss = np.mean(losses)
    return avg_loss

def test(model, loss_fn, dataloader_test, device):
    model = model.to(device)
    N = len(dataloader_test.dataset)
    n_batch = int(N / dataloader_test.batch_size)
    losses = []
    model.eval()
    with torch.no_grad():
        for batch, (x,y) in enumerate(dataloader_test):
            x,y = x.to(device).float(), y.to(device).float()
            
            pred = model(x)
            loss = loss_fn(pred, y)
            losses.append(loss.item())
            
            sys.stdout.write(f"\rTest: {batch+1}/{n_batch} loss_test:{loss}")
            sys.stdout.flush()
            # break
    avg_loss = np.mean(losses)
    return avg_loss

def predict(model, dataloader_test, device, num_batches=4):
    model = model.to(device)
    N = len(dataloader_test.dataset)
    n_batch = int(N / dataloader_test.batch_size)
    model.eval()
    y_actual = []
    y_pred = []
    inputs = []
    ys = []
    with torch.no_grad():
        for batch, (x,y) in enumerate(dataloader_test):
#             ys.append(y)
#             inputs.append(x)
            x,y = x.to(device).float(), y.to(device).float()
            pred = model(x)
            if device == "cuda":
                y_actual.append(np.squeeze(y.detach().cpu().numpy()))
                y_pred.append(np.squeeze(pred.detach().cpu().numpy()))
            else:
                y_actual.append(np.squeeze(y.detach().numpy()))
                y_pred.append(np.squeeze(pred.detach().numpy()))
            inputs.append(x.detach().cpu().numpy())
            
            sys.stdout.write(f"\rPred: {batch+1}/{n_batch}")
            sys.stdout.flush()
            if batch+1 == num_batches:
                break
#     y_actual = [temp if len(temp.shape)>0 else np.array([temp])  for temp in y_actual]
#     y_pred = [temp if len(temp.shape)>0 else np.array([temp])  for temp in y_pred]
    y_actual = np.concatenate(y_actual)
    y_pred = np.concatenate(y_pred)
    inputs = np.concatenate(inputs)
#     ys = np.concatenate(ys)
    return inputs, y_actual, y_pred #, ys

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

losses_train = [1]
losses_test = [1]
plot_freq = 1

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    loss_train = train(model, loss_fn, optimizer, train_dl, device)
    losses_train.append(loss_train)
    print()
    loss_test = test(model, loss_fn, test_dl, device)
    losses_test.append(loss_test)
    print(f"{epoch}th epoch: train = {losses_train}, test = {losses_test}")
    if (epoch+1)%plot_freq == 0:
        ds_pred = DatasetImg(l_test[:num_examples], ab_test[:num_examples], input_transforms=input_transforms, output_transforms=output_transforms)
        dl_pred = DataLoader(ds_test, batch_size=4, shuffle=True)
        x_pred, y_pred, y_hat_pred = predict(model, dl_pred, device, num_batches=1)

        x_pred = transform_multiply(255.0)(x_pred)
        y_pred = transform_multiply(255.0)(y_pred)
        y_hat_pred = transform_multiply(255.0)(y_hat_pred)
    #     ys = transform_multiply(255.0)(ys)

        x_pred = to_channel_last(x_pred)
        y_pred = to_channel_last(y_pred)
        y_hat_pred = to_channel_last(y_hat_pred)
    #     ys = to_channel_last(ys)

        lab_pred = to_lab(x_pred, y_pred, channels_first=False)
        lab_hat_pred = to_lab(x_pred, y_hat_pred, channels_first=False)

        rgb_pred = lab2rgb(lab_pred.astype("uint8"))
        rgb_hat_pred = lab2rgb(lab_hat_pred.astype("uint8"))

        plot_image(epoch+1, 1, rgb_pred, figsize=(10,10), title="RGB Actual")
        plot_image(epoch+1, 2, x_pred, figsize=(10, 10), cmap="gray", title="Gray Scale")
        plot_image(epoch+1, 3, rgb_hat_pred, figsize=(10,10), title="RGB Pred")
        plt.show()
        
        plt.plot(losses_train, label="losses_train")
        plt.plot(losses_test, label="losses_test")
        plt.legend(loc="upper left")
        # plt.show()
        plt.savefig(f'/home/nol/woosuk/addColorToImg/res_imgs/res_epoch{epoch+1}.png')
