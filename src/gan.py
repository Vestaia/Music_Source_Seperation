from torch import nn
import torch
import numpy as np
from time import time
import matplotlib.pyplot as plt
from IPython import display

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of channels
nc = 4

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 1025 x 1
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 1025 x 2
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 1025 x 4
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 1025 x 8
            nn.ConvTranspose2d( ngf, nc, 8, (1,1), 3, bias=False),
            # state size. (nc) x 1025 x 16
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 1025 x 16
            nn.Conv2d(nc, ndf, 7, (1,2), 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 1025 x 8
            nn.Conv2d(ndf, ndf * 2, 3, (1,2), 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 1025 x 4
            nn.Conv2d(ndf * 2, ndf * 4, 3, (1,2), 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 1025 x 2
            nn.Conv2d(ndf * 4, ndf * 8, 3, (1,2), 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 1025 x 1
            nn.Conv2d(ndf * 8, 4, 3, (1,2), 1, bias=False),
        )

    def forward(self, input):
        return self.main(input)


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        loss = torch.mean(torch.cosh(torch.log(torch.abs(y_t - y_prime_t) + 1)))
        return loss


def learn(netD, netG, optimD, optimG, data, target, batch_size=10):
    device = torch.device("cuda:0" if (next(netG.parameters()).is_cuda) else "cpu")
    D_losses = []
    G_losses = []
    D_real = []
    calc = LogCoshLoss().to(device)
    netD.train()
    netG.train()

    for batch in range(np.int(np.floor(len(data)/batch_size))):
        X = data[batch*batch_size:(batch+1)*batch_size].to(device)
        y = target[batch*batch_size:(batch+1)*batch_size].to(device)

        netD.zero_grad()
        output = netD(X)
        lossD_real = calc(output, y)
        lossD_real.backward()
        
        noise = torch.randn(batch_size, nz, 127, 1, device=device)
        fake = netG(noise)
        output = netD(fake.detach())
        zeros = torch.zeros_like(y)
        lossD_fake = calc(output, zeros)
        lossD_fake.backward()

        lossD = lossD_real + lossD_fake

        optimD.step()

        netG.zero_grad()
        output = netD(fake)
        ones = torch.ones_like(y)
        lossG = calc(output, ones)
        lossG.backward()
        optimG.step()
        
        D_real.append(lossD_real.item())
        D_losses.append(lossD.item())
        G_losses.append(lossG.item())


    return np.mean(D_losses), np.mean(G_losses), np.mean(D_real)


def test(netD, netG, data, target, batch_size=10):
    device = torch.device("cuda:0" if (next(netG.parameters()).is_cuda) else "cpu")
    G_losses = []
    D_losses = []
    D_real = []
    calc = LogCoshLoss().to(device)
    netD.eval()
    netG.eval()

    for batch in range(np.int(np.floor(len(data)/batch_size))):

        X = data[batch*batch_size:(batch+1)*batch_size].to(device)
        y = target[batch*batch_size:(batch+1)*batch_size].to(device)

        output = netD(X)
        lossD_real = calc(output, y).detach()
        
        noise = torch.randn(batch_size, nz, 127, 1, device=device)
        fake = netG(noise)
        output = netD(fake.detach())
        zeros = torch.zeros_like(y)
        lossD_fake = calc(output, zeros)

        lossD = lossD_real + lossD_fake

        output = netD(fake)
        ones = torch.ones_like(y)
        lossG = calc(output, ones).detach()
        
        D_real.append(lossD_real.item())
        D_losses.append(lossD.item())
        G_losses.append(lossG.item())

    return np.mean(D_losses), np.mean(G_losses), np.mean(D_real)

def train(netD, netG, optimG, optimD, data, target, batch_size=10, validation_size=None, epochs=1):

    if validation_size is None:
        validation_size = data.size(0) // 10

    #Scrambling data for training
    ind = np.random.permutation(data.size(0))

    #Splitting data to train and validation
    validationX = data[ind[:validation_size]]
    validationy = target[ind[:validation_size]]
    trainX = data[ind[validation_size:]]
    trainy = target[ind[validation_size:]]

    #Training
    print("Starting Training:")
    train_lossD = []
    train_lossG = []
    validation_lossD = []
    validation_lossG = []
    train_realD = []
    validation_realD = []
    plt.ion()
    start = time()
    epoch = 0

    for i in range(epochs):
        lossD,lossG,realD = learn(netD, netG, optimD, optimG, trainX, trainy, batch_size=batch_size)
        train_lossD.append(lossD)
        train_lossG.append(lossG)
        train_realD.append(realD)
        lossD,lossG,realD = test(netD, netG, validationX, validationy, batch_size=batch_size)
        validation_lossD.append(lossD)
        validation_lossG.append(lossG)
        validation_realD.append(realD)
        elapsed = time() - start

        print("Epoch:", i+1)
        print("Train lossD:", train_lossD[i])
        print("validation lossD:", validation_lossD[i])
        print("Train lossG:", train_lossG[i])
        print("validation lossG:", validation_lossG[i])
        print("Train realD:", train_realD[i])
        print("validation realD:", validation_realD[i])
        print("Time:", elapsed)
        plot_train(train_lossD,validation_lossD, train_lossG, validation_lossG, train_realD, validation_realD)
        epoch += 1


def plot_train(train_lossD, test_lossD, train_lossG, test_lossG, train_realD, test_realD):
    plt.clf()
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.plot(np.linspace(2,len(train_lossD),len(train_lossD)-1), train_lossD[1:], label="trainD")
    plt.plot(np.linspace(2,len(test_lossD),len(test_lossD)-1), test_lossD[1:], label="testD")
    plt.subplot(132)
    plt.plot(np.linspace(2,len(train_lossG),len(train_lossG)-1), train_lossG[1:], label="trainG")
    plt.plot(np.linspace(2,len(test_lossG),len(test_lossG)-1), test_lossG[1:], label="testG")
    plt.subplot(133)
    plt.plot(np.linspace(2,len(train_realD),len(train_realD)-1), train_realD[1:], label="trainR")
    plt.plot(np.linspace(2,len(test_realD),len(test_realD)-1), test_realD[1:], label="testR")
    plt.legend()
    display.display(plt.gcf())
    display.clear_output(wait=True)
