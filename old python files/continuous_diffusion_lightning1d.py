# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 23:20:20 2022

@author: martigtu@stud.ntnu.no, adopted from denoising_diffusion_pytorch
"""


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


### Utility functions

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

def add_noise(x, beta=0.01):
    return np.sqrt(1-beta) * x + np.sqrt(beta) * np.random.normal(0, 1, x.shape)


def quasi_rand(shape:tuple, device=None):
    '''
    Return a quasi uniform distribution on [0, 1].
    
    u = [r + 0/k (mod 1), r + 1/k (mod 1), ... , r + (k-1)/k (mod 1)]
    '''
    if type(shape) == int:
        shape = (shape,)
    
    numel = prod(shape)
    
    return (torch.rand(1, device=device) + torch.linspace(0, 1-1/numel, numel, device=device).view(shape)) % 1
    
    

### Modules

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




### Unet model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4),
        channels = 1,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
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

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)


    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
        
        return self.final_conv(x)



###############################################################################
#                                                                             #
#                      VARIATIONAL DIFFUSION MODEL                            #
#                                                                             #
###############################################################################



def get_loss_fn(loss_type):
    """
    return loss function to be used in the Continuous Gaussian Diffusion Model
    """
    loss_fn = {'l1': F.l1_loss, 'l2': F.mse_loss}[loss_type]
    
    return loss_fn




class VDM(pl.LightningModule):
    def __init__(
        self,
        model:Unet,
        ts_length:int,
        loss_type:str    = 'l2',
        objective:str    = 'pred_noise',
        schedule:str     = 'cosine',
        train_lr:float   = 1e-5,
        adam_betas:tuple = (0.9, 0.99),
        quasi_rand:bool  = False
    ):
        super().__init__()
        
        ### assert inputs
        
        assert not (type(self) == VDM and model.channels != model.out_dim)
        assert not model.learned_sinusoidal_cond
        assert type(ts_length) == int,                  'ts_length must be an integer'
        assert loss_type in {'l1', 'l2'},               'loss_type must be either l1 or l2'
        assert objective in {'pred_noise', 'pred_x'},   'objective must be either pred_noise or pred_x0' # only pred_noise implemented
        assert schedule in {'linear', 'cosine'},        'beta_schedule must be either linear or cosine'  # only cosine implemented
        
        ### architecture
        self.model          = model                            # Unet model
        self.channels       = self.model.channels              # number of channels
        self.ts_length      = ts_length                        # time series length
        self.self_condition = self.model.self_condition        # conditional argument
        self.loss_type      = loss_type                        # loss type
        self.objective      = objective                        # prediction objective
        self.schedule       = schedule                         # signal-to-noise schedule
        self.train_lr       = train_lr                         # learning rate
        self.adam_betas     = adam_betas                       # adam parameters
        self.quasi_rand     = quasi_rand                       # quasi-random sampling of times
        self.loss_fn        = get_loss_fn(self.loss_type)      # loss function
        self.schedule_scale = 0.001                            # scale for cosine schedule
        
    
    
    ### Variance schedule
    
    def alpha(self, t):
        fn = self.schedule_scale + (1-2*self.schedule_scale) * torch.cos(t * pi/2)
        return fn
    
    def sigma(self, t):
        return self.schedule_scale + (1-2*self.schedule_scale) * torch.sin(t * pi/2)
        
    
    
    ### Forward process
    
    def q_sample(self, x, t=None, noise=None):
        """
        return a sample from q(z_t | x_0) = alpha_t*x + sigma_t * noise, noise ~ N(0,1)
        """
        
        # batch_size, n_channels, ts_length
        b, c, l = x.shape
        
        # time
        t = default(t, lambda: torch.rand((b), device=self.device))
        if t.numel() == 1:
           t = t * torch.ones((b), device=self.device) 
        
        # noise
        noise = default(noise, lambda: torch.randn_like(x))
        
        # alpha and sigma given time
        alpha_t = self.alpha(t).to(self.device)
        sigma_t = self.sigma(t).to(self.device)
        
        # cast to 3D tensors
        alpha_t = alpha_t.view(-1,1,1)
        sigma_t = sigma_t.view(-1,1,1)
        
        return alpha_t * x + sigma_t * noise
    
    
    def q_forward(self, z_s, s, t, noise=None):
        """
        return a sample from q(z_t | z_s) = alpha_(t|s)*x + sigma_(t|s) * noise, noise ~ N(0,1)
        """
        
        # batch_size, n_channels, ts_length
        b, c, l = z_s.shape
        
        # noise
        noise = default(noise, lambda: torch.randn_like(z_s))
        
        alpha_ts = self.alpha(t) / self.alpha(s)
        sigma_ts = torch.sqrt(self.sigma2(t) - alpha_ts * self.sigma2(s))
        
        # cast to 3D tensors and correct device
        alpha_ts = alpha_ts.view(-1,1,1).to(self.device)
        sigma_ts = sigma_ts.view(-1,1,1).to(self.device)
        
        return alpha_ts * z_s + sigma_ts * noise
    
    
    
    ### Model predictions
    
    def pred_start_from_noise(self, z_t, t, noise):
        """
        return x = (z_t - sigma_t * noise) / alpha_t
        """
        return (z_t - self.sigma(t).view(-1,1,1).to(self.device) * noise) / \
            self.alpha(t).view(-1,1,1).to(self.device)



    def pred_noise_from_start(self, z_t, t, x):
        """
        return noise = (z_t - x * alpha_t) / sigma_t
        """
        return (z_t - self.alpha(t).view(-1,1,1).to(self.device) * x) / \
            self.sigma(t).view(-1,1,1).to(self.device)



    def model_predictions(self, z_t, t, condition=None):
        """
        return the predicted noise and x
        """
        pred = self.model(z_t, t, condition)
        #pred = self.model(z_t, torch.ceil(t*100), condition)
        
        if self.objective == 'pred_noise':
            pred_noise = pred
            pred_x = self.pred_start_from_noise(z_t, t, pred_noise)

        elif self.objective == 'pred_x':
            pred_x = pred
            pred_noise = self.pred_noise_from_start(z_t, t, pred_x)

        return pred_noise, pred_x



    ### Sampling
    
    @torch.no_grad()
    def q_posterior(self, z_t, s, t, pred_noise):
        """
        return the mean and variance from
        q(z_s | z_t, x) = N(mean, var)
        
        mean = 1/alpha_(t|s) * (z_t + (1-e**lambda(t) - lambda(s)) * noise)
             = 1/alpha_(t|s) * (z_t + sigma_t * (1 - (alpha_t * sigma_s / (alpha_s * sigma_t))**2 ) * noise )
             
        var  = (1 - (alpha_t * sigma_s / (alpha_s * sigma_t))**2 ) sigma_s**2
        """
        
        # batch_size, num_channels, ts_length
        b, c, l = z_t.shape
        
        # change t to 1D if it is 0D
        #t = t * torch.ones((b)) if (t.numel() == 1) else t
        #s = s * torch.ones((b)) if (s.numel() == 1) else s
        
        alpha_ts   = self.alpha(t) / self.alpha(s)
        sigma_t   = self.sigma(t)
        sigma2_s  = self.sigma(s)**2
        expr      = 1 - sigma2_s * (alpha_ts / sigma_t)**2

        # change Tensors to correct shape and device
        for tensor in alpha_ts, sigma_t, sigma2_s, expr:
            tensor = tensor.view(-1,1,1).to(self.device)
        
        # calculate model mean
        posterior_mean = 1/alpha_ts * (z_t - sigma_t * expr * pred_noise) # Dont know why there is a minus sign here?
        
        # calculate model variance
        posterior_variance  = expr * sigma2_s
        
        return posterior_mean, posterior_variance


    @torch.no_grad()
    def p_sample(self, z, s, t, condition=None, guidance_weight=1.0):
        """
        single sample loop
        """
        b, *_ = z.shape
        
        if exists(condition):
            # conditional / unconditional sampling
            cond_noise, cond_x     = self.model_predictions(z, t, condition)
            uncond_noise, uncond_x = self.model_predictions(z, t, None)
            # classifier-free predictions
            pred_noise = (1-guidance_weight) * uncond_noise + guidance_weight * cond_noise
            pred_x = (1-guidance_weight) * uncond_x + guidance_weight * cond_x
        else:
            # normal (unconditional sampling)
            pred_noise, pred_x = self.model_predictions(z, t, None)

        # get conditional mean and variance
        model_mean, model_variance = self.q_posterior(z, s, t, pred_noise)
        
        eps = torch.randn_like(z) if t>0 else 0.0
        
        # sample x from a previous time step
        pred_z = model_mean + model_variance * eps
        
        # normalize variance
        pred_z /= torch.std(pred_z)
        
        return pred_z, pred_x


    @torch.no_grad()
    def sample(self, n_samples=1, sampling_steps=100, condition=None, guidance_weight=1.0):
        """ 
        ancestral sampling from the diffusion model
        """
        
        # time discretization
        tau = torch.linspace(1, 0, sampling_steps+1, device=self.device)
        
        # sample from prior N(0, I)
        shape = (n_samples, self.channels, self.ts_length)
        z = torch.randn(shape, device=self.device)
        
        
        for s, t in tqdm(zip(tau[1:], tau[:-1]), desc='Sampling', total=sampling_steps):
            # change from 0D to 1D
            s = s.view(1)
            t = t.view(1)
            
            # sample from p_theta(z_s | z_t, t, condition)
            z, _ = self.p_sample(z, s, t, condition, guidance_weight)
            
        return z
    


    ### Training    
    
    
    def p_losses(self, x, t=None, noise=None, condition=None):
        """
        calculate the batch loss
        """
        
        # batch_size, n_channels, ts_length
        b, c, l = x.shape
        
        # (quasi)random times
        if self.quasi_rand:    
            t = default(t, lambda: quasi_rand((b), device=self.device))
        else:
            t = default(t, lambda: torch.rand((b), device=self.device))
            
        # random noise
        noise = default(noise, lambda: torch.randn_like(x))

        # diffused sample
        z_t = self.q_sample(x, t, noise)
        
        # predict objective
        pred = self.model(z_t, t, condition)
        #pred = self.model(z_t, torch.ceil(t*100), condition)

        # target and weight
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x':
            target = x
        else:
            raise ValueError('unknown objective!')
        
        
        loss = self.loss_fn(pred, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        
        return loss.mean()
    
    
    
    def forward(self, x, *args, **kwargs):
        b, c, l = x.shape
        assert l == self.ts_length, f'length of time series must be {self.ts_length}'
        t = torch.rand((b), device=self.device)
        
        return self.p_losses(x, t)
    
    
    # --- Added for using PyTorch Lightning ---
    
    def training_step(self, batch, batch_idx):
        t = torch.rand([batch.shape[0]], device=self.device)
        loss = self.p_losses(batch, t)
        self.log("train/loss", loss)
        
        return loss


    def validation_step(self, batch, batch_idx):
        t = torch.rand([batch.shape[0]], device=self.device)
        val_loss = self.p_losses(batch, t)
        self.log("val/loss", val_loss)
        
        return val_loss


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.train_lr, betas=self.adam_betas)
        
        return optimizer
