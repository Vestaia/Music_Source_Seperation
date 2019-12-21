
import os
# os.environ['OMP_NUM_THREADS'] = "1"
# os.environ['MKL_NUM_THREADS'] = "6"

import learn
import features

import torch
import torch.optim 
import torchvision.datasets as datasets
from torchsummary import summary
from torch_stft import STFT

from scipy.io.wavfile import write
import musdb

import preprocessing
import numpy as np
import gan

lr = 2e-4
beta1 = 0.5

cwd = os.getcwd()
path = cwd + "/musdb18/"
mus_train = musdb.DB(root=path, subsets='train')
mus_test = musdb.DB(root=path, subsets='test')

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

netD = gan.Discriminator().to(device)
netG = gan.Generator().to(device)
netD.apply(gan.weights_init)
netG.apply(gan.weights_init)
optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

summary(netD, (4, 1025, 17), batch_size=5)
summary(netG, (100, 127, 1), batch_size=5)


stft = STFT(filter_length=2048, hop_length=512)


track = mus_train.tracks[5]
mus = torch.FloatTensor(track.audio).T
voc = torch.FloatTensor(track.targets['vocals'].audio).T

mus_f, mus_p = stft.transform(mus)
voc_f, voc_p = stft.transform(voc)

for n in range(1):
    
    train, target = preprocessing.features(mus_f, mus_p, voc_f, voc_p, nsamples=4000, windowsize=17, random=False)
    train[:,0:2] = train[:,0:2]
    target[:,0:2] = target[:,0:2]

    gan.train(netD, netG, optimD, optimG, train[:2000], target[:2000], validation_size=100, batch_size=4, epochs=30)

# train, target = preprocessing.features(mus_f, mus_p, voc_f, voc_p, nsamples=4000, windowsize=31, random=False)
# train[:,0:2] = train[:,0:2]
# target[:,0:2] = target[:,0:2]

pred = preprocessing.run(netD, train[2000:])

source = preprocessing.istft(stft, train[2000:])
write("source.wav", 44100, source.T)

out = preprocessing.istft(stft, pred)
write("output.wav", 44100, out.T)

preprocessing.plot(pred,target)

'''
source = np.vstack((mus_l, mus_r)).T
out = np.vstack((librosa.istft(mus_l_stft), librosa.istft(mus_r_stft))).T

write("input.wav", 44100, source)
#write("target.wav", 44100, voc_l)
write("output.wav", 44100, out)
'''


'''for n in range(1):
    track = mus_train.tracks[5]

    mus = torch.FloatTensor(track.audio).T
    voc = torch.FloatTensor(track.targets['vocals'].audio).T

    mus_f, mus_p = stft.transform(mus)
    voc_f, voc_p = stft.transform(voc)

    train, target = preprocessing.features(mus_f, mus_p, voc_f, voc_p, nsamples=4000, windowsize=25)
    train[:,0:2] = torch.log10(1+train[:,0:2])
    target[:,0:2] = torch.log10(1+target[:,0:2])
    learn.train(model, optim, train[:2000], target[:2000], validation_size=200, batch_size=10, epochs=20)

pred = preprocessing.run(model, train[2000:])
pred[:,0:2] = 10**(pred[:,0:2]) - 1
train[:,0:2] = 10**(train[:,0:2]) - 1
target[:,0:2] = 10**(target[:,0:2]) - 1
source = preprocessing.istft(stft, train[2000:])
write("source.wav", 44100, source.T)
tar = preprocessing.istft(stft, target[2000:])
write("target.wav", 44100, tar.T)
out = preprocessing.istft(stft, pred)
write("output.wav", 44100, out.T)

preprocessing.plot(pred,target)
'''