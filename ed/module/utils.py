import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

def plot_channels(img_batch, figsize=(8,3), cmap=None):
    if len(img_batch.shape)==3:
        img_batch = np.expand_dims(img_batch, axis=0)
    for img in img_batch:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1,3,1)
        plt.imshow(img[:,:,0].T, cmap=cmap)
        plt.title(f"ab-0")
        
        plt.subplot(1,3,2)
        plt.imshow(img[:,:,1].T, cmap=cmap)
        plt.title(f"ab-1")
        
        plt.subplot(1,3,3)
        plt.imshow(img[:,:,2].T, cmap=cmap)
        plt.title(f"l")
        
        plt.show()

def plot_image(epoch, j, img_batch, figsize=(8,3), cmap=None, title=None):
    if len(img_batch.shape)==3:
        img_batch = np.expand_dims(img_batch, axis=0)
    N = len(img_batch)
    fig = plt.figure(figsize=figsize)
    for i in range(N):
        img = img_batch[i]
#         img = np.transpose(img, [1,0,2])
        plt.subplot(1,N,i+1)
        plt.imshow(img, cmap=cmap)
    if title is not None:
        plt.title(f"{title}")
    plt.savefig(f'/home/nol/woosuk/addColorToImg/res_imgs/res_epoch{epoch+1}_{j}.png')
    # plt.show(f'/home/nol/woosuk/addColorToImg/res_imgs/res_epoch{epoch+1}_{i}.png')
    
    
# def plot_image(epoch, i, img_batch, figsize=(8,3), cmap=None, title=None):
#     if len(img_batch.shape)==3:
#         img_batch = np.expand_dims(img_batch, axis=0)
#     N = len(img_batch)
#     fig = plt.figure(figsize=figsize)
#     for i in range(N):
#         img = img_batch[i]
# #         img = np.transpose(img, [1,0,2])
#         plt.subplot(1,N,i+1)
#         plt.imshow(img, cmap=cmap)
#     if title is not None:
#         plt.title(f"{title}")
#     plt.savefig(f'/home/nol/woosuk/addColorToImg/res_imgs/res_epoch{epoch+1}_{i}.png')
#     # plt.show(f'/home/nol/woosuk/addColorToImg/res_imgs/res_epoch{epoch+1}_{i}.png')

def to_lab(l, ab, channels_first=True):
    if channels_first:
        if len(l.shape)==3:
            l = np.expand_dims(l, axis=1)
        lab = np.concatenate([l, ab], axis=1)
    else:
        if len(l.shape)==3:
            l = np.expand_dims(l, axis=3)
        lab = np.concatenate([l, ab], axis=3)
    return lab

def lab2rgb(lab):
    if len(lab.shape)==4:
        arr = []
        for img in lab:
            img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
            arr.append(img)
        arr = np.array(arr)
    else:
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return arr

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def rgb2lab(rgb):
    if len(rgb.shape)==4:
        arr = []
        for img in rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            arr.append(img)
        arr = np.array(arr)
    else:
        arr = cv2.cvtColor(rgb, cv2.COLOR_LAB2RGB)
    return arr

def to_channel_first(arr):
    if len(arr.shape)==4:
        arr = np.transpose(arr, [0,3,2,1])
    else:
        arr = np.transpose(arr, [2,1,0])
    return arr

def to_channel_last(arr):
    if len(arr.shape)==4:
        arr = np.transpose(arr, [0,3,2,1])
    else:
        arr = np.transpose(arr, [2,1,0])
    return arr



class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


    
def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")
        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")
