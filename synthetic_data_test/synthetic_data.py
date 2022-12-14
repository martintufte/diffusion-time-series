# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 01:44:00 2022

@author: martigtu@stud.ntnu.no
"""

import torch
from torch.utils.data import Dataset



class SyntheticData(Dataset):
    def __init__(self, kinds=("sin", "sin"), periods=(25, 70), weigths = (0.5, 0.5), n_samples=5000, length=128):
        super().__init__()
        
        self.n_samples = n_samples
        self.length = length
        self.kinds = kinds
        self.periods = periods
        self.weigths = weigths
        
        ts_combined = torch.zeros(self.n_samples, 1, self.length)
        for kind, period, weigth in zip(self.kinds, self.periods, self.weigths):
            if kind == "saw":
                R = period*torch.rand(self.n_samples).view(self.n_samples, 1, 1)
                ts = torch.arange(0, self.length).repeat(self.n_samples, 1, 1) + R
                ts %= period
            elif kind == "sin":
                R = period*torch.rand(self.n_samples).view(self.n_samples, 1, 1)
                ts = torch.sin((torch.arange(0, self.length).repeat(self.n_samples, 1, 1) + R) * 2 * torch.pi / period)
            elif kind == "abs sin":
                R = period*torch.rand(self.n_samples).view(self.n_samples, 1, 1)
                ts = torch.abs(torch.sin((torch.arange(0, self.length).repeat(self.n_samples, 1, 1) + R) * 2 * torch.pi / period))
            else:
                raise ValueError('Dataset type is not in "saw" or "sin".')
            
            ts_combined += ts * weigth
            
        # standardize
        # ts -= torch.mean(ts)
        # ts /= torch.std(ts)
        
        # transform to [-1, 1]
        ts_combined -= (ts_combined.max() + ts_combined.min())/2
        ts_combined *= 2/(ts_combined.max() - ts_combined.min())

        self.data  = ts_combined
        
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx]
    


class SyntheticLabeledData(Dataset):
    def __init__(self, datasets):
        self.n_samples = sum([dataset.n_samples for dataset in datasets])
        
        Ys = []
        for idx, dataset in enumerate(datasets):
            Ys.append(torch.Tensor([idx]).type(torch.int32).repeat(dataset.n_samples))
            
        self.X = torch.concat([dataset.data for dataset in datasets], dim=0)
        self.Y = torch.concat(Ys)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx, :], self.Y[idx]








