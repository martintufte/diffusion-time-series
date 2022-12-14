# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 17:36:41 2022

@author: marti
"""

from unet_lightning import BLUP_Conv1d

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import matplotlib.pyplot as plt


def simple_conv():
    dim = 1
    dim_out = 1
    conv = nn.Conv1d(dim, dim_out, kernel_size=3, padding=1, bias=0)
    
    # manually set the weigths to test the implelmentation
    conv.weight = torch.nn.Parameter(torch.Tensor([[[-1, 0, 1]]]))
    
    return conv

def complex_conv():
    dim = 1
    dim_out = 1
    conv = BLUP_Conv1d(dim, dim_out, kernel_size=3, padding=1, bias=0)
    
    # manually set the weigths to test the implelmentation
    conv.weight = torch.nn.Parameter(torch.Tensor([[[-1, 0, 1]]]))
    
    return conv

if __name__=="__main__":
    print('Hellu World!')
    
    conv  = simple_conv()
    conv2 = complex_conv()
    
    start, end = 0, 4
    steps = 100
    x = torch.cos(torch.linspace(start, end, steps).view(1,1,-1) + 0.3)
    
    x_der = conv(x)
    x_der2 = conv2(x)
    
    plt.plot(x[0,0,:])
    plt.plot(x_der[0,0,:].detach())
    plt.plot(x_der2[0,0,:].detach())
    plt.show()