import torch
import sys
import numpy as np
from torch_stft import STFT
import matplotlib.pyplot as plt

def features(train_f_stft, train_p_stft, target_f_stft, target_p_stft, nsamples=sys.maxsize, windowsize=15, random=True):
    nsamples = np.min([nsamples,train_f_stft.shape[-1]-windowsize])
    train = torch.empty((nsamples, 2*train_f_stft.shape[0], train_f_stft.shape[1], windowsize))
    target = torch.empty((nsamples, 2*target_f_stft.shape[0], target_f_stft.shape[1], 1))

    if random:
        iterator = np.random.permutation(train.size(0))
    else:
        iterator = range(nsamples)
    for m in iterator[:nsamples]:
        train[m,0] = train_f_stft[0,:,m:m+windowsize]
        train[m,1] = train_f_stft[1,:,m:m+windowsize]
        train[m,2] = train_p_stft[0,:,m:m+windowsize]
        train[m,3] = train_p_stft[1,:,m:m+windowsize]

        target[m,0] = target_f_stft[0,:,m+windowsize // 2:m+windowsize // 2 + 1]
        target[m,1] = target_f_stft[1,:,m+windowsize // 2:m+windowsize // 2 + 1]
        target[m,2] = target_p_stft[0,:,m+windowsize // 2:m+windowsize // 2 + 1]
        target[m,3] = target_p_stft[1,:,m+windowsize // 2:m+windowsize // 2 + 1]

    return train, target

def run(model, data, size = None):
    if size is None:
        size = data.size(0)
    output = torch.zeros(size, 4, data.size(2), 1)
    for n in range(size):
        output[n] = model(data[n:n+1].cuda()).detach().cpu()
    return output.permute(3,1,2,0)[0]

def istft(stft, data):
    if(data.dim() == 4):
        data = data.permute(3,1,2,0)[0]
    out = stft.inverse(data[0:2], data[2:4]).numpy()
    return out

def plot(pred, target):
    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    plt.title('Prediction')
    plt.xlabel('Frames')
    plt.ylabel('FFT bin')
    plt.imshow(20*np.log10(1+pred[0]), aspect='auto', origin='lower')
    plt.subplot(212)
    plt.title('Target')
    plt.xlabel('Frames')
    plt.ylabel('FFT bin')
    plt.imshow(20*np.log10(1+target[:,0,:,0].T), aspect='auto', origin='lower')
    plt.tight_layout()
