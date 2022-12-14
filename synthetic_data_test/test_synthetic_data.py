# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 17:53:48 2022

@author: martigtu@stud.ntnu.no
"""

# models
from unet_lightning import Unet
from vdm_lightning import VDM

# data
from synthetic_data import SyntheticData, SyntheticLabeledData

# metrics
from metrics import calculate_fid

# torch
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

# logger
import wandb
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from pathlib import Path

# plotting
import matplotlib.pyplot as plt

def plot(samples, title, i_low=0, i_high=10):
    for i in range(i_low, i_high):
        plt.plot(samples[i, 0, :].cpu())
    plt.title(title)
    plt.show()
    

if __name__()=='__main__':
    
    ### create synthetic time series data
    ts_length = 100
    batch_size = 64
    
    dataset_train1 = SyntheticData(('sin',), (80,), (1.0,), 5000, ts_length)
    dataset_val1   = SyntheticData(('sin',), (80,), (1.0,), 5000, ts_length)
    
    dataset_train2 = SyntheticData(('sin',), (40,), (1.0,), 5000, ts_length)
    dataset_val2   = SyntheticData(('sin',), (40,), (1.0,), 5000, ts_length)
    
    
    # combine the two different datasets
    dataset_train = SyntheticLabeledData([dataset_train1, dataset_train2])
    dataset_val = SyntheticLabeledData([dataset_val1, dataset_val2])
    
    loader_train  = DataLoader(dataset_train, batch_size=batch_size, num_workers=0, shuffle=True)
    loader_val    = DataLoader(dataset_val, batch_size=batch_size, num_workers=0, shuffle=True)
    
    
    ### unet model
    unet = Unet(
        ts_length    = ts_length,
        n_classes    = 2,
        dim          = 128,
        dim_mults    = (1, 2, 4),
        time_dim     = 32,
        class_dim    = 32,
        padding_mode = 'zeros'
    )
    
    ### diffusion model
    diffusion_model = VDM(
        unet,
        loss_type  = 'l2',
        objective  = 'pred_noise',
        train_lr   = 1e-5,
        adam_betas = (0.9, 0.99),
        cond_drop  = 0.1,
        quasi_rand = True
    )
    
    
    ### fit model
    trainer = Trainer(
        max_epochs = 10,
        log_every_n_steps = 10,
        accelerator = 'gpu',
        devices = 1,
        logger = WandbLogger(project='Synthetic', name=datetime.now().strftime('%D - %H:%M:%S'))
    )
    trainer.fit(diffusion_model, loader_train, loader_val)
    wandb.finish()
    
    
    ### sample new data
    n_samples = 10
    sampling_steps = 100
    condition = torch.Tensor([0]).type(torch.int32).repeat(n_samples)
    
    # conditional sampling
    guidance_strength = 1.0
    cond_samples = diffusion_model.sample(n_samples, sampling_steps, condition, 1+guidance_strength)
    plot(cond_samples, f"Conditional, steps={sampling_steps}, gs = {guidance_strength}")
    
    # unconditional sampling
    uncond_samples = diffusion_model.sample(n_samples, sampling_steps, None)
    plot(uncond_samples, f"Unconditional sampling, steps={sampling_steps}")

    
    ### calculate FID score
    
    # z1: ground truth, concatenation of the training and validation data
    # z2: generated samples
    
    z1 = torch.concat((dataset_train.data, dataset_val.data)).reshape(-1, ts_length)
    z2 = uncond_samples[0].reshape(-1, ts_length)
    
    fid = calculate_fid(z1, z2)
    print('FID score is {}.'.format(round(float(fid),4)))








