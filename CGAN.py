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

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100+num_class, 128, normalize=False), #100->128
            *block(128, 256), #128->256
            *block(256, 512), #256->512
            *block(512, 1024),#512->1024 
            nn.Linear(1024, int(np.prod(img_shape))),  # 1024->image shape 28*28*1
            nn.Tanh()  
        )

    def forward(self, z , label):
        label = label_emb(label).view(z.size(0),10)
        input_z = torch.cat((z,label),dim=1)
        img = self.model(input_z)
        img = img.view(img.size(0), *img_shape) 
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_class + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, img , label):
        label = label_emb(label).view(img.size(0),10)
        img_flat = torch.cat((img.view(img.size(0), -1),label),dim=1)
        validity = self.model(img_flat)
        return validity


def TrainSetLoader():
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader

def plot():
    z = torch.rand(10,100).cuda()
    labels = torch.LongTensor([0,1,2,3,4,5,6,7,8,9]).cuda()
    gen_imgs = generator(z,labels)
    datas = gen_imgs.cpu().detach().numpy().reshape(10,28,28)
    datas=255*(0.5*datas+0.5)
    plt.figure(figsize=(6,6))
    plt.axis('off')
    gs = GridSpec(1, 10)
    for index in range(10):
        ax = plt.subplot(gs[index])
        plt.imshow(datas[index],cmap='gray')
        plt.xticks([])  #去掉横坐标值
        plt.yticks([])  #去掉纵坐标值
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

# parameters initialization
n_epochs = 200
batch_size = 64
lr = 0.0002
b1 = 0.5
b2 = 0.999
img_size = 28
channels = 1
img_shape = (channels, img_size, img_size)  # 1*28*28
num_class = 10
# model initialization
cuda = True if torch.cuda.is_available() else False  # gpu
generator = Generator()
discriminator = Discriminator()
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
# embedding
label_emb = nn.Embedding(num_class,num_class).cuda()

for epoch in range(n_epochs):
    for i, (imgs, real_label) in enumerate(dataloader):
        valid = torch.FloatTensor(imgs.size(0), 1).fill_(1.0).cuda()
        fake = torch.FloatTensor(imgs.size(0), 1).fill_(0.0).cuda()
        label = torch.LongTensor(np.random.randint(0, num_class, imgs.size(0))).view(imgs.size(0),1).cuda()
        real_imgs = imgs.cuda()
        real_label = real_label.cuda()
        #  Train Generator
        optimizer_G.zero_grad()
        z = torch.rand(imgs.size(0),100).cuda()
        gen_imgs = generator(z,label)
        g_loss = loss(discriminator(gen_imgs,label), valid)
        g_loss.backward()
        optimizer_G.step()
        #  Train Discriminator
        optimizer_D.zero_grad()
        real_loss = loss(discriminator(real_imgs,real_label), valid)
        fake_loss = loss(discriminator(gen_imgs.detach(),label), fake) 
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
    print("Epoch %d  D loss: %f G loss: %f"
            % (epoch,d_loss.item(), g_loss.item())
        )
    if epoch%3 ==0:
      plot()
