import os
import numpy as np
import math
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = img_size // 4 
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),    
            nn.Upsample(scale_factor=2),  
            nn.Conv2d(128, 128, 3, stride=1, padding=1), 
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Upsample(scale_factor=2),  
            nn.Conv2d(128, 64, 3, stride=1, padding=1), 
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),  
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),  
            *discriminator_block(16, 32), 
            *discriminator_block(32, 64), 
            *discriminator_block(64, 128), 
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4  
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity



def TrainSetLoader():
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(img_size),transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                #短边resize到这个int数，长边则根据对应比例调整
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader

def plot(gen_imgs):
    datas = gen_imgs.cpu().detach().numpy()[0:25].reshape(25,img_size, img_size)
    datas=255*(0.5*datas+0.5)
    plt.figure(figsize=(6,6))
    plt.axis('off')
    gs = GridSpec(5, 5)
    for index in range(25):
        ax = plt.subplot(gs[index])
        plt.imshow(datas[index],cmap='gray')
        plt.xticks([])  #去掉横坐标值
        plt.yticks([])  #去掉纵坐标值
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def weights_init_normal(m):
    # torch.nn.init.normal_(tensor, mean=0, std=1)
    # torch.nn.init.constant_(tensor, val)
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# parameters initialization
n_epochs = 200
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
img_size = 32
channels = 1
img_shape = (channels, img_size, img_size)  
# model initialization
cuda = True if torch.cuda.is_available() else False  # gpu
generator = Generator()
discriminator = Discriminator()
generator.apply(weights_init_normal)  # dfs 
discriminator.apply(weights_init_normal)
if cuda:
    generator.cuda()
    discriminator.cuda()
# tarin process initailization
loss = torch.nn.BCELoss()
if cuda:
    loss.cuda()
dataloader = TrainSetLoader()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))


for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        valid = torch.FloatTensor(imgs.size(0), 1).fill_(1.0).cuda()
        fake = torch.FloatTensor(imgs.size(0), 1).fill_(0.0).cuda()
        real_imgs = imgs.cuda()
        #  Train Generator
        optimizer_G.zero_grad()
        z = torch.rand(imgs.size(0),100).cuda()
        gen_imgs = generator(z)
        g_loss = loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()
        #  Train Discriminator
        optimizer_D.zero_grad()
        real_loss = loss(discriminator(real_imgs), valid)
        fake_loss = loss(discriminator(gen_imgs.detach()), fake) 
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
    print("Epoch %d  D loss: %f G loss: %f"
            % (epoch,d_loss.item(), g_loss.item())
        )
    if epoch%3 ==0:
      plot(gen_imgs)
