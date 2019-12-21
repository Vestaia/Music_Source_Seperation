
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



cwd = os.getcwd()
path = cwd + "/musdb18/"
mus_train = musdb.DB(root=path, subsets='train')
mus_test = musdb.DB(root=path, subsets='test')

model = learn.Net().cuda()
optim = torch.optim.Adam(model.parameters(), lr=5e-5)

summary(model, (4, 1024, 25), batch_size=10)

stft = STFT(filter_length=4096, hop_length=512)


for n in range(50):
    track = mus_train.tracks[n%4 + 4]
    mus = torch.FloatTensor(track.audio).T
    voc = torch.FloatTensor(track.targets['vocals'].audio).T

    mus_f, mus_p = stft.transform(mus)
    voc_f, voc_p = stft.transform(voc)
    train, target = preprocessing.features(mus_f, mus_p, voc_f, voc_p, nsamples=4000, windowsize=25, random=True)
    # train[:,0:2] = torch.log(train[:,0:2]+1)
    # target[:,0:2] = torch.log(target[:,0:2]+1)
    learn.train(model, optim, train[:4000], target[:4000], validation_size=200, batch_size=10, epochs=1)

track = mus_train.tracks[7]
mus = torch.FloatTensor(track.audio).T
voc = torch.FloatTensor(track.targets['vocals'].audio).T
mus_f, mus_p = stft.transform(mus)
voc_f, voc_p = stft.transform(voc)
train, target = preprocessing.features(mus_f, mus_p, voc_f, voc_p, nsamples=4000, windowsize=25, random=False)
# train[:,0:2] = torch.log(train[:,0:2]+1)
# target[:,0:2] = torch.log(target[:,0:2]+1)

pred = preprocessing.run(model, train[2000:])

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