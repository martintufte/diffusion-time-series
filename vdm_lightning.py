# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 14:03:40 2022

@author: martigtu@stud.ntnu.no
"""

###############################################################################
#                                                                             #
#                      VARIATIONAL DIFFUSION MODEL                            #
#                                                                             #
###############################################################################


from unet_lightning import Unet

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

from einops import reduce
from math import pi, prod
from random import random
from tqdm import tqdm


# --- Utility functions ---

def exists(x):
    ''' return True if x is not None '''
    return x is not None

def default(val, d):
    ''' returns val if it exists, else d '''
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    ''' identity function '''
    return t

def quasi_rand(shape:tuple, device=None):
    '''
    return a quasi uniform distribution on [0, 1].
    u = [r + 0/k (mod 1), r + 1/k (mod 1), ... , r + (k-1)/k (mod 1)]
    '''
    if type(shape) == int:
        shape = (shape,)
    
    numel = prod(shape)
    
    return (torch.rand(1, device=device) + torch.linspace(0, 1-1/numel, numel, device=device).view(shape)) % 1


def get_loss_fn(loss_type):
    """
    return loss function to be used in the Continuous Gaussian Diffusion Model
    """
    loss_fn = {'l1': F.l1_loss, 'l2': F.mse_loss}[loss_type]
    
    return loss_fn



# --- Variational Diffusion Model ---

class VDM(pl.LightningModule):
    def __init__(
        self,
        model:Unet,
        loss_type:str    = 'l2',
        objective:str    = 'pred_noise',
        train_lr:float   = 1e-5,
        adam_betas:tuple = (0.9, 0.99),
        cond_drop:float  = 0.1,
        quasi_rand:bool  = True
    ):
        super().__init__()
        
        # --- assert inputs ---
        assert not (type(self) == VDM and model.in_channels != model.out_channels), 'wrong number of channels in model'
        assert type(model.ts_length) == int, 'ts_length must be an integer'
        assert loss_type in {'l1', 'l2'}, 'loss_type must be either l1 or l2'
        assert objective in {'pred_noise', 'pred_x', 'pred_v'}, f'objective {objective} is not supported'
        
        # --- architecture ---
        self.model          = model                            # Unet model
        self.channels       = model.in_channels                # number of input channels
        self.ts_length      = model.ts_length                  # time series length
        self.loss_type      = loss_type                        # loss type
        self.objective      = objective                        # prediction objective
        self.train_lr       = train_lr                         # learning rate
        self.adam_betas     = adam_betas                       # adam parameters
        self.cond_drop      = cond_drop                        # conditional dorpout parameter
        self.quasi_rand     = quasi_rand                       # quasi-random sampling of times
        self.loss_fn        = get_loss_fn(self.loss_type)      # loss function
        self.schedule_scale = 0.001                            # scale for cosine schedule
        
    
    
    # --- Variance schedule ---
    
    def alpha(self, t):
        return self.schedule_scale + (1-2*self.schedule_scale) * torch.cos(t * pi/2)
    
    def sigma(self, t):
        return self.schedule_scale + (1-2*self.schedule_scale) * torch.sin(t * pi/2)
        
    
    
    # --- Forward Markovian diffusion process q ---
    
    def q_sample(self, x, t=None, noise=None):
        """
        return a sample from q(z_t | x_0) = alpha_t*x + sigma_t*noise, noise ~ N(0,1)
        """
        
        # batch_size, n_channels, ts_length
        b, c, l = x.shape
        
        # time
        t = default(t, lambda: torch.rand((b), device=self.device))
        if t.numel() == 1:
           t = t * torch.ones((b), device=self.device) 
        
        # noise
        noise = default(noise, lambda: torch.randn_like(x))
        
        # alpha and sigma at t
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        
        # cast to 3D tensors and correct device
        alpha_t = alpha_t.view(-1,1,1).to(self.device)
        sigma_t = sigma_t.view(-1,1,1).to(self.device)
        
        return alpha_t * x + sigma_t * noise
    
    
    def q_forward(self, z_s, s, t, noise=None):
        """
        return a sample from q(z_t | z_s) = alpha_(t|s)*x + sigma_(t|s) * noise, noise ~ N(0,1)
        """
        
        # batch_size, n_channels, ts_length
        b, c, l = z_s.shape
        
        # noise
        noise = default(noise, lambda: torch.randn_like(z_s))
        
        # alpha and sigma at s given t
        alpha_ts = self.alpha(t) / self.alpha(s)
        sigma_ts = torch.sqrt(self.sigma2(t) - alpha_ts * self.sigma2(s))
        
        # cast to 3D tensors and correct device
        alpha_ts = alpha_ts.view(-1,1,1).to(self.device)
        sigma_ts = sigma_ts.view(-1,1,1).to(self.device)
        
        return alpha_ts * z_s + sigma_ts * noise
    
    
    
    # --- Model predictions ---
    
    def pred_start_from_noise(self, z, t, noise):
        """
        return x = (z_t - sigma_t * noise) / alpha_t
        """
        # alpha and sigma at t
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        
        # cast to 3D tensors and correct device
        alpha_t = alpha_t.view(-1,1,1).to(self.device)
        sigma_t = sigma_t.view(-1,1,1).to(self.device)
        
        return (z - sigma_t * noise) / alpha_t


    def pred_noise_from_start(self, z, t, x):
        """
        return noise = (z_t - alpha_t * x) / sigma_t
        """
        # alpha and sigma at t
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        
        # cast to 3D tensors and correct device
        alpha_t = alpha_t.view(-1,1,1).to(self.device)
        sigma_t = sigma_t.view(-1,1,1).to(self.device)
        
        return (z - alpha_t * x) / sigma_t


    def model_predictions(self, z, t, condition=None):
        """
        return the predicted noise and predicted x
        """
        b, _, _ = z.shape
        
        if t.numel() == 1:
            t = t.repeat(b)
        
        pred = self.model(z, t, condition)
        
        if self.objective == 'pred_noise':
            pred_noise = pred
            pred_x = self.pred_start_from_noise(z, t, pred_noise)

        elif self.objective == 'pred_x':
            pred_x = pred
            pred_noise = self.pred_noise_from_start(z, t, pred_x)

        return pred_noise, pred_x



    # --- Sampling ---
    
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
        
        
        alpha_ts   = self.alpha(t) / self.alpha(s)
        sigma_t   = self.sigma(t)
        sigma2_s  = self.sigma(s)**2
        expr      = 1 - sigma2_s * (alpha_ts / sigma_t)**2

        # change Tensors to correct shape and device
        alpha_ts = alpha_ts.view(-1,1,1).to(self.device)
        sigma_t  = sigma_t.view(-1,1,1).to(self.device)
        sigma2_s = sigma2_s.view(-1,1,1).to(self.device)
        expr     = expr.view(-1,1,1).to(self.device)
        
        # calculate model mean
        posterior_mean = 1/alpha_ts * (z_t - sigma_t * expr * pred_noise)
        
        # calculate model variance
        posterior_variance  = expr * sigma2_s
        
        return posterior_mean, posterior_variance


    @torch.no_grad()
    def p_sample(self, z, s, t, condition=None, guidance_weight=1.0):
        """
        single sample loop p_theta(z_s | z_t)
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
        
        # Z-normalize the variance, but not the mean
        pred_z /= torch.std(pred_z)
        
        return pred_z, pred_x


    @torch.no_grad()
    def sample(self, n_samples=1, sampling_steps=100, condition=None, guidance_weight=1.0):
        """ 
        ancestral sampling from the diffusion model
        """
        
        # time discretization
        tau = torch.linspace(1, 0, sampling_steps+1, device=self.device).view(-1,1)
        
        
        # sample from prior N(0, I)
        z = torch.randn((n_samples, self.channels, self.ts_length), device=self.device)
        
        # sample from p_theta(z_s | z_t, t, condition)
        for s, t in tqdm(zip(tau[1:], tau[:-1]), desc='Sampling', total=sampling_steps):
            z, _ = self.p_sample(z, s, t, condition, guidance_weight)
            
        return z
    


    # --- Training ---   
    
    def p_losses(self, x, t=None, condition=None, noise=None):
        """
        calculate the batch loss
        """
        
        # batch_size, n_channels, ts_length
        b, c, l = x.shape
        
        # random time
        if self.quasi_rand:    
            t = default(t, lambda: quasi_rand((b), device=self.device))
        else:
            t = default(t, lambda: torch.rand((b), device=self.device))
            
        # random noise
        noise = default(noise, lambda: torch.randn_like(x, device=self.device))

        # diffused sample
        z_t = self.q_sample(x, t, noise)
        
        # predict objective with condition (w/ dropout)
        if random() < self.cond_drop:
            pred = self.model(z_t, t, None)
        else:
            pred = self.model(z_t, t, condition)
        
        
        # target
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x':
            target = x
        else:
            raise ValueError('unknown objective!')
        
        
        # calculate loss
        loss = self.loss_fn(pred, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        
        return loss.mean()

    
    # --- Overwriting PyTorch Lightning in-build methods ---
    
    def forward(self, x, *args, **kwargs):
        print('This is not how the variational diffusion model should be used! \
               Use the sample method instead!')
        
        return x
        
    
    def training_step(self, batch, batch_idx):
        X, Y = batch
        Y = Y.flatten() # nescessary if Y is 2D
        
        t = torch.rand([X.shape[0]], device=self.device)
        loss = self.p_losses(X, t, Y)
        self.log("train/loss", loss)
        
        return loss


    def validation_step(self, batch, batch_idx):
        X, Y = batch
        Y = Y.flatten() # nescessary if Y is 2D
        
        t = torch.rand([X.shape[0]], device=self.device)
        val_loss = self.p_losses(X, t, Y)
        self.log("val/loss", val_loss)
        
        return val_loss


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.train_lr, betas=self.adam_betas)
        
        return optimizer
