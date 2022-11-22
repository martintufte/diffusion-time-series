# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:17:20 2022

@author: martigtu@stud.ntnu.no
"""

###############################################################################
#                                                                             #
#                                   Unet                                      #
#                                                                             #
###############################################################################


import math
from functools import partial
import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

from einops import rearrange, reduce
from tqdm import tqdm

from math import pi, prod


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
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1)
    )




def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)




class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)




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




### Sinusoidal positional embeddings

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




### Building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv1d(dim, dim_out, kernel_size=3, padding=1)
        self.group_norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.PReLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.group_norm(x)
        
        # adaptive GroupNorm
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        
        return x




class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, emb_dim=None, groups=8):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.PReLU(),
            nn.Linear(emb_dim, dim_out*2)
        ) if exists(emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()


    def forward(self, x, cond_emb=None):
        scale_shift = None
        if exists(self.embedder) and exists(cond_emb):
            cond_emb = self.embedder(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1')
            scale_shift = cond_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)




class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim*3, kernel_size=1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )


    def forward(self, x):
        b, c, h = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x -> b h c x', h = self.heads), qkv)

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
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)


    def forward(self, x):
        b, c, h = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x -> b h c x', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h x d -> b (h d) x', x = h)
        
        return self.to_out(out)




# --- Unet model ---

class Unet(pl.LightningModule):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4),
        channels = 1,
        out_channels = None,
        n_classes = 1,
        resnet_block_groups = 8,
        learned_sinusoidal_cond = False,
        learned_sinusoidal_dim = 16,
        class_dim = 64,
        time_dim = 64
    ):
        super().__init__()

        # number of in/out channels
        self.channels = channels
        self.out_channels = default(out_channels, channels)
        self.n_classes    = n_classes
        self.uncond_class = n_classes
        
        # dimensions
        self.dims = [dim] + [dim*m for m in dim_mults]
        self.mid_dim = self.dims[-1]
        
        # time/class embedding dimensions
        self.time_dim = time_dim
        self.class_conditional = (self.n_classes > 1)
        self.class_dim = class_dim if self.class_conditional else 0
        self.emb_dim = self.time_dim + self.class_dim
        

        # --- time embeddings ---
        self.learned_sinusoidal_cond = learned_sinusoidal_cond
        self.learned_sinusoidal_dim = learned_sinusoidal_dim

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(self.learned_sinusoidal_dim)
            fourier_dim = self.learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        
        self.time_emb = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim)
        )
        
        
        # --- class embeddings ---
        self.class_emb = nn.Embedding(self.n_classes+1, self.class_dim) if self.class_conditional else nn.Identity()
        
        
        # --- layers ---
        
        # initial layer        
        self.init_block = nn.Conv1d(channels, dim, kernel_size=7, padding=3)
        
        # partially defined Resnet block
        resnet_block = partial(ResnetBlock, emb_dim=self.emb_dim, groups=resnet_block_groups)
        # in/out dim for each down/up layer
        in_out = list(zip(self.dims[:-1], self.dims[1:]))
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = (ind >= num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, kernel_size=3, padding = 1)
            ]))

        self.mid_block1 = resnet_block(self.mid_dim, self.mid_dim)
        self.mid_attn = Residual(PreNorm(self.mid_dim, Attention(self.mid_dim)))
        self.mid_block2 = resnet_block(self.mid_dim, self.mid_dim)

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, kernel_size=3, padding = 1)
            ]))

        self.final_res_block = resnet_block(dim*2, dim)
        self.final_conv_block = nn.Conv1d(dim, self.out_channels, kernel_size=1)


    def forward(self, z, t, y=None):
        '''
        z : Latent variable
        t : time
        y : class label
        '''
        
        # 0. time and class embedding
        t = torch.ceil(t * 1000) # Hack
        
        emb = self.time_emb(t)
        
        # if class conditional: concatenate the class embeddings
        if self.class_conditional:
            if exists(y):
                y = y.type(torch.int32)
                class_emb = self.class_emb(y)
            else:
                y = torch.Tensor([self.uncond_class]).type(torch.int32)
                y = y.repeat(z.shape[0]).to(self.device)
                class_emb = self.class_emb(y)
            emb = torch.concat((emb, class_emb), dim=1)
        
        # 1. initial block
        z = self.init_block(z)
        first_skip_connection = z.clone()
                
        # 2. down part
        skip_connections = []
        for res_block1, res_block2, attn, downsample in self.downs:
            z = res_block1(z, emb)
            skip_connections.append(z)
            z = res_block2(z, emb)
            z = attn(z)
            skip_connections.append(z)
            z = downsample(z)
        
        # 3. bottleneck
        z = self.mid_block1(z, emb)
        z = self.mid_attn(z)
        z = self.mid_block2(z, emb)
        
        # 4. up part
        for res_block1, res_block2, attn, upsample in self.ups:
            z = torch.cat((z, skip_connections.pop()), dim=1)
            z = res_block1(z, emb)
            z = torch.cat((z, skip_connections.pop()), dim=1)
            z = res_block2(z, emb)
            z = attn(z)
            z = upsample(z)
        
        # 5. final block
        z = torch.cat((z, first_skip_connection), dim=1)
        z = self.final_res_block(z, emb)
        z = self.final_conv_block(z)
        
        return z
