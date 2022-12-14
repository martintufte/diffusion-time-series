# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 10:15:00 2022

@author: martigtu@stud.ntnu.no
"""

import torch
from numpy import cov
from scipy.linalg import sqrtm
import numpy as np
from numpy import asarray, expand_dims, log, mean, exp


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



# calculate the inception score for p(y|x)
def calculate_is(P_yx, n_split: int = 10, shuffle: bool = True, eps: float = 1e-16):
    '''
    Calculates the Inception Distance between two representation vectors.
    P_yx: (batch_size dim)
    '''
    if shuffle:
        np.random.shuffle(P_yx)  # in-place

    scores = list()
    n_part = int(np.floor(P_yx.shape[0] / n_split))
    for i in range(n_split):
        # retrieve p(y|x)
        ix_start, ix_end = i * n_part, i * n_part + n_part
        p_yx = P_yx[ix_start:ix_end]

        # calculate p(y)
        p_y = expand_dims(p_yx.mean(axis=0), 0)

        # calculate KL divergence using log probabilities
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))

        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)

        # average over images
        avg_kl_d = mean(sum_kl_d)

        # undo the log
        is_score = exp(avg_kl_d)

        # store
        scores.append(is_score)

    # average across images
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std

    

def validate_with_classifier(classifier, X_train, Y_train, X_tilde, Y_tilde, X_val, Y_val):
    '''
    Train a classifier on both X_train and X_tilde, and then validate on
    the validation set.
    '''
    pass



if __name__ == '__main__':
    # define two representation vectors
    z1 = torch.rand(10, 2048)
    z2 = torch.rand(10, 2048)

    fid = calculate_fid(z1, z2)
    print('FID: {}'.format(round(float(fid),4)))
