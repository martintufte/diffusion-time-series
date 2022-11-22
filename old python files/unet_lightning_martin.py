# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 10:19:41 2022

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
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.PReLU() # nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        
        return x




class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.PReLU(), # nn.SiLU()
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()


    def forward(self, x, time_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)




class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )


    def forward(self, x):
        b, c, h = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
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
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
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

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        dim_mults=(1, 2, 4),
        channels = 1,
        out_channels = None,
        self_condition = False,
        resnet_block_groups = 8,
        learned_sinusoidal_cond = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        self.out_channels = default(out_channels, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_channels, 1)


    def forward(self, x, t, y=None):
        # Hack
        t *= 1000
        
        # 1. initial block
        x = self.init_block(x)
        
        first_skip_connection = x.clone()

        emb = rearrange(self.time_emb(t), 'b -> b 1')
        #emb_y = self.class_emb(y)
        
        skip_connections = []
        
        # 2. down part
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, emb)
            skip_connections.append(x)
            x = block2(x, emb)
            x = attn(x)
            skip_connections.append(x)
            x = downsample(x)
        
        # 3. bottleneck
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)
        
        # 4. up part
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = block1(x, emb)
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = block2(x, emb)
            x = attn(x)
            x = upsample(x)

        
        # 5. final block
        x = torch.cat((x, first_skip_connection), dim=1)
        
        x = self.final_res_block(x, t)
        
        return self.final_conv(x)




# --- Unet model ---

class CondUnet(nn.Module):
    def __init__(
        self,
        ts_length:int,
        n_classes:int,
        dim:int = 64,
        dim_mults:tuple = (1, 2, 4),
        channels:int = 1,
        out_channels:int = None,
        resnet_block_groups:int = 8
    ):
        super().__init__()


        self.ts_length = ts_length  # time series length
        self.n_classes = n_classes  # number of classes
        self.channels = channels    # number of channels
        time_dim = 1                # time embedding dimension
        class_dim = n_classes       # class embedding dimension
        
        
        # time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(time_dim, 16),
            nn.GELU(),
            nn.Linear(16, time_dim)
        )
        
        # class embedding
        self.class_emb = nn.Embedding(self.n_classes, embedding_dim=class_dim)
        
        # dimensions of Unet
        dims = [dim] + [m*dim for m in dim_mults]
        mid_dim = dims[-1]
        
        # resnet class (with number of groups and time emb dim defined)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, groups=resnet_block_groups)

        
        # --- 1. initial block ---
        self.init_block = nn.Conv1d(channels, dim, kernel_size=7, padding=3)

        
        # --- 2. down part ---
        self.downs = nn.ModuleList([])
        
        for dim_in, dim_out in zip(dims[:-2], dims[1:-1]):
            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out)
            ]))
        self.downs.append(nn.ModuleList([
            resnet_block(mid_dim, mid_dim),
            resnet_block(mid_dim, mid_dim),
            Residual(PreNorm(mid_dim, LinearAttention(mid_dim))),
            nn.Identity(mid_dim, mid_dim)
        ]))
        
        
        # --- 3. bottleneck ---
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn   = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)
        
        
        # --- 4. up part ---
        self.ups = nn.ModuleList([])
        
        for dim_in, dim_out in zip(list(reversed(dims))[:-2], list(reversed(dims))[1:-1]):
            self.ups.append(nn.ModuleList([
                resnet_block(dim_in*2, dim_in), # multiply by two since we are concatenating the skip connection
                resnet_block(dim_in*2, dim_in),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in, dim_out)
            ]))
        self.ups.append(nn.ModuleList([
            resnet_block(dim*2, dim),
            resnet_block(dim*2, dim),
            Residual(PreNorm(dim, LinearAttention(dim))),
            nn.Identity(dim, dim)
        ]))


        # --- 5. final block ---
        self.out_dim = default(out_channels, channels)

        self.final_resnet_block = resnet_block(dim*2, dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, kernel_size=1)


    def forward(self, x, t, y=None):
        
        # 1. initial block
        x = self.init_block(x)
        
        first_skip_connection = x.clone()

        emb = rearrange(self.time_emb(t), 'b -> b 1')
        #emb_y = self.class_emb(y)
        
        skip_connections = []
        
        # 2. down part
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, emb)
            skip_connections.append(x)
            x = block2(x, emb)
            x = attn(x)
            skip_connections.append(x)
            x = downsample(x)
        
        # 3. bottleneck
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)
        
        # 4. up part
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = block1(x, emb)
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = block2(x, emb)
            x = attn(x)
            x = upsample(x)

        
        # 5. final block
        x = torch.cat((x, first_skip_connection), dim=1)
        
        x = self.final_res_block(x, t)
        
        return self.final_conv(x)
