# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:03:34 2022

@author: martigtu@stud.ntnu.no
"""

# models
from denoising_diffusion_lightning1d import Unet, GaussianDiffusion1D

# data
from synthetic_data import SyntheticData
from preprocess_ucr import DatasetImporterUCR, UCRDataset

# metrics
from metrics import calculate_fid

# torch
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime
from pathlib import Path

# plotting
import matplotlib.pyplot as plt



if __name__()=='__main__':
    
    ### import UCR time series data
    dataset = 'Wafer' # TwoPatterns
    batch_size = 32
    cwdir = Path('C:/Users/marti/OneDrive/Dokumenter/9. semester/Prosjektoppgave/diffusion-time-series')

    # data pipeline
    dataset_importer = DatasetImporterUCR(cwdir, dataset, data_scaling=True)
    
    dataset_train = UCRDataset("train", dataset_importer)
    dataset_val = UCRDataset("test", dataset_importer)
    loader_train = DataLoader(dataset_train, batch_size, num_workers=0, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size, num_workers=0, shuffle=True)
    
    # time series length
    ts_length = dataset_importer.X_train.shape[2]
    
    
    
    ### unet model
    model = Unet(dim = ts_length, dim_mults = (1, 2, 4, 8))
    
    
    ### gaussian diffusion model
    diffusion_model = GaussianDiffusion1D(
        model,
        ts_length,
        diffusion_steps = 100,
        beta_schedule = 'cosine',
        loss_type = 'l1',
        train_lr = 1e-5
    )
    
    
    ### fit model
    trainer = Trainer(
        max_epochs = 10,
        log_every_n_steps = 10,
        accelerator = 'gpu',
        devices = 1,
        logger = WandbLogger(project=dataset, name=datetime.now().strftime('%D - %H:%M:%S'))
    )
    trainer.fit(diffusion_model, loader_train, loader_val)
    wandb.finish()
    
    
    
    
    ### sample new data (100 samples takes about 1 min)
    n_samples = 10
    
    samples = diffusion_model.sample(n_samples, return_all_steps=False)
    
    
    ### plot samples
    
    # plot ground truth
    for i in range(10):
        plt.plot(dataset_train.X[i,0,:].cpu())
    plt.show()
    
    # plot generated samples
    for i in range(10):
        plt.plot(samples[i, 0, :].cpu())
    plt.show()
    
    
    ### calculate FID score
    
    # z1: ground truth, concatenation of the training and validation data
    # z2: generated samples
    
    z1 = torch.concat((dataset_importer.X_train, dataset_importer.X_test)).reshape(-1, ts_length)
    z2 = samples.reshape(-1, ts_length)
    
    z3 = torch.randn((100, 152))
    
    fid = calculate_fid(z1, z2)
    print('FID score is {}.'.format(round(float(fid),4)))








cwdir = Path('C:/Users/marti/OneDrive/Dokumenter/9. semester/Prosjektoppgave/diffusion-time-series')

# data pipeline
dataset_importer = DatasetImporterUCR(cwdir, "Wafer", data_scaling=True)
dataset = UCRDataset("train", dataset_importer)
data_loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)

# get a mini-batch of samples
for batch in data_loader:
    x, y = batch
    break
print('x.shape:', x.shape)

# plot
n_samples = 5
c = 0
fig, axes = plt.subplots(n_samples, 1, figsize=(3.5, 1.7*n_samples))
for i, ax in enumerate(axes):
    ax.plot(x[i, c])
plt.tight_layout()
plt.show()