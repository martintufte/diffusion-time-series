# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:15:00 2022

@author: martigtu@stud.ntnu.no
"""

import torch
from numpy import cov
from scipy.linalg import sqrtm


def calculate_fid(z1, z2):
    '''
    Calculates the Fréchet Inception Distance between two normal distributions z1 and z2.
    '''
    # calculate mean and covariance statistics
    mu1, sigma1 = z1.mean(axis=0), torch.tensor( cov(z1, rowvar=False) )
    mu2, sigma2 = z2.mean(axis=0), torch.tensor( cov(z2, rowvar=False) )

    # calculate sum squared difference between means
    ssdiff = ((mu1 - mu2)**2).sum()

    # calculate sqrt of product between cov
    covmean = torch.tensor(sqrtm(torch.mm(sigma1, sigma2)))

    # check and correct imaginary numbers from sqrt
    if torch.is_complex(covmean):
        covmean = covmean.real

    # calculate score
    fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0*covmean)
    
    return fid



def calculate_is(z1, z2):
    '''
    Calculates the Inception Distance between two normal distributions z1 and z2.
    '''
    pass




if __name__ == '__main__':
    # define two representation vectors
    z1 = torch.rand(10, 2048)
    z2 = torch.rand(10, 2048)

    fid = calculate_fid(z1, z2)
    print('FID: {}'.format(round(float(fid),4)))
