# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:38:47 2022

@author: martigtu@stud.ntnu.no
"""

# models
from unet_lightning import Unet
from vdm_lightning import VDM

# data
from preprocess_ucr import DatasetImporterUCR, UCRDataset

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



if __name__()=='__main__':
    ### import UCR time series data
    dataset = 'Wafer'
    batch_size = 64
    cwdir = Path('C:/Users/marti/OneDrive/Dokumenter/9. semester/Prosjektoppgave/diffusion-time-series')

    # data pipeline
    dataset_importer = DatasetImporterUCR(cwdir, dataset, data_scaling=True)
    
    # train and test set combined
    dataset_all   = UCRDataset("train/test", dataset_importer)
    loader_all    = DataLoader(dataset_all, batch_size, num_workers=0, shuffle=True)
    loader_val    = DataLoader(dataset_all, batch_size, num_workers=0, shuffle=False)
    
    # time series length and number of classes
    ts_length = dataset_importer.X_train.shape[2]
    n_classes = torch.unique(dataset_importer.Y_train).shape[0]
    
    
    ### unet model
    unet = Unet(
        dim       = ts_length,
        dim_mults = (1, 2, 4),
        n_classes = 2,
    )
    
    # test that the unet model works
    sample = dataset_importer.X_train[[0, 1, 0]]
    t = torch.Tensor([0.5, 0.6, 0.7])
    y = torch.Tensor([0, 1, 0]).type(torch.int32)
    
    # test conditional / unconditional unet progression
    unet(sample, t, None).shape
    unet(sample, t, y).shape
    
    ### diffusion model
    diffusion_model = VDM(
        unet,
        ts_length,
        loss_type  = 'l2',
        objective  = 'pred_noise',
        schedule   = 'cosine',
        train_lr   = 1e-5,
        adam_betas = (0.9, 0.99),
        cond_drop  = 0.1,
        quasi_rand = True
    )


    # test conditional / unconditional vdm progression loss    
    diffusion_model.p_losses(sample, t, condition=y, noise=None)

    
    ### fit model
    trainer = Trainer(
        max_epochs = 10,
        log_every_n_steps = 10,
        accelerator = 'gpu',
        devices = 1,
        logger = WandbLogger(project=dataset, name=datetime.now().strftime('%D - %H:%M:%S'))
    )
    trainer.fit(diffusion_model, loader_all, loader_all)
    wandb.finish()
    
    
    ### uncoditional sampling of new data
    n_samples = 10
    sampling_steps = 30
    
    samples = diffusion_model.sample(n_samples, sampling_steps)
    
    # plot generated samples
    for i in range(10):plt.plot(samples[i, 0, :].cpu())
    
    # plot ground truth (for reference)
    for i in range(10):plt.plot(dataset_all.X[i,0,:].cpu())
    
    
    ### conditional sampling of new data
    n_samples = 10
    sampling_steps = 30
    
    samples = diffusion_model.sample(n_samples, sampling_steps, condition=torch.Tensor([1,1,1,1,1,1,1,1,1,1]).type(torch.int32), guidance_weight=2.0)
    
    samples = diffusion_model.sample(n_samples, sampling_steps, condition=torch.Tensor([0,0,0,0,0,0,0,0,0,0]).type(torch.int32), guidance_weight=2.0)
    
    
    ### calculate FID score
    
    # X: ground truth, concatenation of the training and validation data
    # X_tilde: generated samples
    
    X = torch.concat((dataset_importer.X_train, dataset_importer.X_test)).reshape(-1, ts_length)
    X_tilde = samples.reshape(-1, ts_length)
    
    fid = calculate_fid(X_tilde, X)
    print('FID score is {}.'.format(round(float(fid),4)))
    
    
    
    
    
    
    
    ### old test that the diffusion model works!
    '''
    
    def plot_samples(samples):
        for sample in samples:
            for i in range(sample.shape[0]):
                plt.plot(sample[i,0,:].cpu())
        plt.show()
    
    # test q_sample (ok!)
    sample = dataset_importer.X_train[0:2]    
    diffused_sample = diffusion_model.q_sample(sample, t=torch.Tensor([0.1]))
        
    plot_samples([sample, diffused_sample])
    
    
    # test q_forward (ok!)
    sample = dataset_importer.X_train[[1]] 
    
    tau = torch.linspace(0, 1, 10)
    
    diffused_samples = [sample]
    for s, t in zip(tau[:-1], tau[1:]):
        sample = diffusion_model.q_forward(sample, s, t)
        diffused_samples.append(sample)
        
    plot_samples(diffused_samples[:5])
    
    
    # test q_reverse_distribution (ok!)
    s = torch.Tensor([0.1])
    t = torch.Tensor([0.2])
    
    sample = dataset_importer.X_train[0:2]    
    diffused_sample = diffusion_model.q_sample(sample, t)
    
    mean, var = diffusion_model.q_reverse_distribution(diffused_sample, s, t, torch.randn_like(sample))
    
    plot_samples([sample[[0]], sample[[1]], mean[[0]], mean[[1]]])
    plot_samples([sample[[0]], mean[[0]] + 1.96*torch.sqrt(var[0]), mean[[0]] - 1.96*torch.sqrt(var[0])])
    
    
    # test pred_x0_from_noise (ok!)
    sample = dataset_importer.X_train[[10]]
    noise = torch.randn_like(sample)
    t = torch.Tensor([0.2])
    
    sample_t = diffusion_model.q_sample(sample, t, noise)
    
    pred_sample = diffusion_model.pred_start_from_noise(sample_t, t, noise)
    
    plot_samples([sample, pred_sample]) # they match!
    
    # test pred_noise_from_x0 (ok!)
    sample = dataset_importer.X_train[[10]]
    noise = torch.randn_like(sample)
    t = torch.Tensor([0.2])
    
    sample_t = diffusion_model.q_sample(sample, t, noise)
    
    pred_noise = diffusion_model.pred_noise_from_start(sample_t, t, sample)
    
    plot_samples([noise, pred_noise]) # they match!
    
    # test model_predictions (maybe?)
    sample = dataset_importer.X_train[0:10]
    noise = torch.randn_like(sample)
    t = torch.Tensor([0.2])
    
    pred_noise, pred_x0 = diffusion_model.model_predictions(sample, t, cond=None)
    
    plot_samples([pred_noise.detach(), pred_x0.detach()])
    
    # test p_sample (maybe?)
    s = torch.Tensor([0.1])
    t = torch.Tensor([0.2])
    
    sample = dataset_importer.X_train[0:2]    
    sample_s = diffusion_model.q_sample(sample, s)
    sample_t = diffusion_model.q_forward(sample_s, s, t)
    
    plot_samples([sample, sample_s, sample_t])
    
    pred_sample_s, pred_sample = diffusion_model.p_sample(sample_t, s, t)
    
    plot_samples([sample_s, pred_sample_s])
    
    # test sample (maybe?)
    x = diffusion_model.sample(n_samples=2, sampling_steps=50)
    plot_samples([x])
    
    loss = diffusion_model.p_losses(x)
    
    # test p_losses (maybe?)
    sample = dataset_importer.X_train[0:10]
    loss = diffusion_model.p_losses(sample)
    
    
    # test backwards distribution
    sample = dataset_importer.X_train[[10]]
    noise = torch.randn_like(sample)
    t = torch.Tensor([0.5])
    s = torch.Tensor([0.])
    
    diffused_sample = diffusion_model.q_sample(sample, t)
    
    plot_samples([sample, diffused_sample])
    
    pred_noise, pred_x = diffusion_model.model_predictions(diffused_sample, t)
    
    mean, var = diffusion_model.q_posterior(diffused_sample, s, t, pred_noise)
    
    plot_samples([mean.detach(), diffused_sample])

    plot_samples([noise, pred_noise.detach()])
    plot_samples([sample, diffused_sample, pred_x.detach()])
    
    pred_x = (diffused_sample - diffusion_model.sigma(t).view(-1,1,1) * pred_noise) / \
        diffusion_model.alpha(t).view(-1,1,1)
        
        
    # own try at Unet
    unet = Unet(
        ts_length = ts_length,
        n_classes = n_classes,
        channels = 1,
        out_channels = 1,
        dim = 128,
        dim_multipliers = (1,2,4),
        skip_connections = (True, True, True),
        resnet_block_groups = 8  
    )
    '''

