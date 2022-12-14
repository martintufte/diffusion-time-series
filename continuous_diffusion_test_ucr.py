# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:38:47 2022

@author: martigtu@stud.ntnu.no
"""

# models
from unet_lightning import Unet # Denoising U-Net
from vdm_lightning import VDM # Variational Diffusion Model
#from supervised_FCN.models import fcn # Fully Convulational Network

# data
from preprocess_ucr import DatasetImporterUCR, UCRDataset, SynthethicUCRDataset

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
from plot import nice_plot


if __name__()=='__main__':
    
    ### data path
    path = Path('C:/Users/marti/OneDrive/Dokumenter/9. semester/Prosjektoppgave/diffusion-time-series')
    model_folder = 'saved_models'


    ### import UCR time series data
    dataset_name = 'TwoPatterns'
    dataset_importer = DatasetImporterUCR(path, dataset_name)
    ts_length        = dataset_importer.ts_length
    n_classes        = dataset_importer.n_classes
    le               = dataset_importer.label_encoder
    enc_dec          = dataset_importer.standard_scaler
    
    
    ### data loader
    batch_size  = 32
    train_dataset = UCRDataset("train", dataset_importer)
    train_loader  = DataLoader(train_dataset, batch_size, num_workers=0, shuffle=True)
    val_dataset = UCRDataset("test", dataset_importer)
    val_loader  = DataLoader(val_dataset, batch_size, num_workers=0, shuffle=False)
    
    ### unet model
    unet = Unet(
        ts_length    = ts_length,
        n_classes    = n_classes,
        dim          = 64,
        dim_mults    = (1, 2, 4, 8),
        time_dim     = 64,
        class_dim    = 32,
        padding_mode = 'replicate'
    )
    
    
    ### diffusion model
    diffusion_model = VDM(
        unet,
        loss_type  = 'l2',
        objective  = 'pred_noise',
        train_lr   = 1e-5,
        adam_betas = (0.9, 0.99),
        cond_drop  = 0.1
    )
    
    
    ### fit model
    logger = WandbLogger(
        name = datetime.now().strftime('%D - %H:%M:%S'),
        project = dataset_name,
        save_dir = model_folder
    )
    
    trainer = Trainer(
        default_root_dir = path.joinpath(model_folder),
        max_epochs = 2,
        log_every_n_steps = 10,
        accelerator = 'gpu',
        devices = 1,
        logger = logger
    )
    trainer.fit(diffusion_model, train_loader, val_loader)
    wandb.finish()
    
    
    ### sampling of new data
    n_samples = 7
    sampling_steps = 30
    condition = torch.Tensor([0,1,0,0,0,1,1]).type(torch.int32)
    guidance_strength = 2.0
    
    samples     = diffusion_model.sample(n_samples, sampling_steps, condition, guidance_strength)
    unc_samples = diffusion_model.sample(n_samples, sampling_steps, None)
    
    # plot generated samples
    nice_plot(samples, condition)
    
    for i in range(n_samples):
        nice_plot(samples[[i],:,:], condition[[i]])
    
    
    ### create synthetic dataset
    dataset_syn = SynthethicUCRDataset(diffusion_model, train_dataset, sampling_steps=10, guidance_weigth = 1.0)
    
    loader_syn = DataLoader(dataset_syn, batch_size=32, num_workers=0, shuffle=True)

    # get a mini-batch of samples
    for batch in loader_syn:
        x, y = batch
        break

    # plot
    n_samples = 5
    c = 0
    fig, axes = plt.subplots(n_samples, 1, figsize=(3.5, 1.7*n_samples))
    for i, ax in enumerate(axes):
        ax.plot(x[i, c])
    plt.tight_layout()
    plt.show()    
    
    
    
    
    ### calculate FID score
    
    # X: ground truth, concatenation of the training and validation data
    # X_tilde: generated samples
    
    X = torch.concat((dataset_importer.X_train, dataset_importer.X_test)).reshape(-1, ts_length)
    X_tilde = unc_samples.reshape(-1, ts_length)
    
    fid = calculate_fid(X_tilde, X)
    print('FID score is {}.'.format(round(float(fid),4)))
    
    # suspecting that the FID gives me too high results
    # try clipping the samples.
    unc_samples_clipped = torch.clip(unc_samples, min=-0.7, max=0.7)
    for i in range(10):plt.plot(unc_samples_clipped[i, 0, :].cpu())
    
    X = torch.concat((dataset_importer.X_train, dataset_importer.X_test)).reshape(-1, ts_length)
    X_tilde = unc_samples_clipped.reshape(-1, ts_length)
    
    fid = calculate_fid(X_tilde, X)
    print('FID score is {}.'.format(round(float(fid),4)))
    
    
    
    
    ### old tests that the diffusion model works!
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

