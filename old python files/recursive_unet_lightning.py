# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 14:36:03 2022

@author: martigtu@stud.ntnu.no
"""

###############################################################################
#                                                                             #
#                            Recursive Unet                                   #
#                                                                             #
###############################################################################


import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
import pytorch_lightning as pl

from einops import rearrange, reduce


# --- Utility functions ---

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def l2norm(t):
    return F.normalize(t, dim=-1)



# Defined by LucidRains
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g




class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)




# --- Sinusoidal positional embeddings ---

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered



# --- Building block modules ---

class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    Weight standardized convolutional layer (works wells with group normalization)
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        
        # normalize the weights of the convolution layer
        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.projection = WeightStandardizedConv1d(dim, dim_out, kernel_size=3, padding=1)
        self.group_norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.PReLU() # parameterized ReLU, max(0,x) + a*min(0,x)

    def forward(self, x, scale_shift=None):
        x = self.projection(x)
        x = self.group_norm(x)
        
        # adaptive GruopNorm
        if exists(scale_shift):
            scale, shift = scale_shift
            x = (1 + scale) * x + shift
            
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, time_emb_dim, groups=8):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(time_emb_dim, dim_out*2)
        )

        self.block1 = Block(dim_in, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)


    def forward(self, x, time_emb=None):
        scale_shift = None
        
        if exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h




class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim*3, kernel_size=1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, kernel_size=1),
            LayerNorm(dim)
        )


    def forward(self, x):
        b, c, h = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x -> b h c x', h=self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / h

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c x -> b (h c) x', x = h)
        
        return self.to_out(out)




class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim*3, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, kernel_size=1)


    def forward(self, x):
        b, c, h = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x -> b h c x', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h x d -> b (h d) x', x = h)
        
        return self.to_out(out)



# --- Modules ---

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv1d(dim, default(dim_out, dim), kernel_size=3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), kernel_size=4, stride=2, padding=1)


class UnetBlock(pl.LightningModule):
    def __init__(
        self,
        channels,
        downsampler,
        downs,
        middle,
        ups,
        upsampler,
        skip_connections,
    ):
        super().__init__()
        
        self.skip_connections = list(skip_connections)
        
        # Layers
        self.downsampler = downsampler
        self.downs       = downs
        self.middle      = middle
        self.ups         = ups
        self.upsampler   = upsampler
        
        # test Unet
        x = torch.randn((10, channels, 16))
        t = torch.rand((1,))
        
        o = self.forward(x, t)
        assert x.shape == o.shape, f'i/o shape mismatch! {x.shape} != {o.shape}'
        
    
    def forward(self, x, t, y=None):
        x = self.downsampler(x)
        
        connection_stack = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if self.skip_connections[i]:
                connection_stack.append(x.clone())
            
        x = self.middle(x, t, y)
            
        for i, up in enumerate(self.ups):
            if self.skip_connections[i]:
                x = torch.cat((x, connection_stack.pop()), dim=1)
            x = up(x)
                
        x = self.upsampler(x)
        
        return x


class MiddleWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        
        self.model = model

    def forward(self, x, t, y):
        return self.model(x)
        

class Unet(pl.LightningModule):
    def __init__(
        self,
        ts_length,
        n_classes,
        channels,
        out_channels = None,
        dim = 64,
        dim_multipliers = (1, 2, 4),
        skip_connections = (True, True, True),
        resnet_block_groups = 8
    ):
        super().__init__()
        
        self.ts_length = ts_length
        self.n_classes = n_classes
        
        # input/output channels
        self.channels = channels
        self.out_channels = default(out_channels, channels)
        
        # dimensions of the Unet
        self.dim = dim
        self.dims = [m*dim for m in dim_multipliers]
        self.mid_dim = self.dims[-1]
        self.depth = len(dim_multipliers)
        
        # resnet class (with number of groups and time emb dim defined)
        resnet_block = partial(ResnetBlock, time_emb_dim=1, groups=resnet_block_groups)

        # --- initial block; change number of channels ---
        self.init_sampler = nn.Conv1d(self.channels, self.dim, kernel_size=3, padding=1)
        self.init_block   = nn.ModuleList([resnet_block(self.dim, self.dim)])

        # --- middle block ---
        self.middle = MiddleWrapper(nn.Sequential(
            resnet_block(self.mid_dim, self.mid_dim),
            Residual(PreNorm(self.mid_dim, Attention(self.mid_dim))),
            resnet_block(self.mid_dim, self.mid_dim)
        ))
        
        # recursively define the middle
        if self.depth > 1:
            for i in range(self.depth-1, 0, -1):
                dim_io, dim, skip = self.dims[i-1], self.dims[i], skip_connections[i]
                
                self.middle = UnetBlock(
                    channels = dim_io,
                    downsampler = Downsample(dim_io, dim),
                    downs = nn.ModuleList([
                        resnet_block(dim, dim),
                        resnet_block(dim, dim),
                        Residual(PreNorm(dim, LinearAttention(dim)))
                    ]),
                    middle = self.middle,
                    ups = nn.ModuleList([
                        resnet_block(dim*(1 + skip), dim),
                        resnet_block(dim*(1 + skip), dim),
                        Residual(PreNorm(dim, LinearAttention(dim)))
                    ]),
                    upsampler = Upsample(dim, dim_io),
                    skip_connections = (skip, skip, False),
                )
        
        # --- final block ---
        self.final_block = nn.ModuleList([resnet_block(self.dim*(1 + skip_connections[0]), self.dim)])
        self.final_sampler = nn.Conv1d(self.dim, self.out_channels, kernel_size=1)
        
        # combine the initial block, the middle block and the final block
        self.model = UnetBlock(
            self.channels,
            self.init_sampler,
            self.init_block,
            self.middle,
            self.final_block,
            self.final_sampler,
            (skip_connections[0],)
        )
        

    def forward(self, x, t, y=None):
        return self.model(x, t, y)





def main():
    print('Hellu there!')
    
    rnet = Unet(
        ts_length = 128,
        n_classes = 2,
        in_channels = 1,
        out_channels = None,
        dim = 64,
        dim_multipliers = (1,),
        skip_connections = (True,),
        resnet_block_groups = 8    
    )
    
    x = torch.randn((10,1,128))
    
    print(rnet)
    
    print(rnet.forward(x, 0))
    
    
    

if __name__=='__main__':
    main()















