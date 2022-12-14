# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 16:43:18 2022

@author: martigtu@stud.ntnu.no
"""

import matplotlib.pyplot as plt


def nice_plot(samples, labels=None, max_plot=10, save_as=None):
    '''
    A function for plotting the time series in a nice format and saving them.
    '''
    n = min(samples.shape[0], max_plot)
    
    colors = ['grey', 'green']
        
    for i in range(n):
        c = labels[i].item()
        plt.plot(samples[i, 0, :].cpu(), label = c, linestyle = 'solid')
        
    plt.legend()
    plt.show()



if __name__ == '__main__':
    pass
