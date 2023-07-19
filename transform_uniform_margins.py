# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:44:18 2023

@author: A R Fogg
"""

import scipy

import matplotlib.pyplot as plt

def transform_from_data_scale_to_uniform_margins(data, plot=False):
    """

    Parameters
    ----------
    data : np.array
        
    plot : BOOL, optional
        If plot == True, plots the distributions of data in data 
        scale and on uniform margins. The default is False.

    Returns
    -------
    data_unif : np.array
        Data on uniform margins

    """
    
    # Transform the variables to uniform margins
    data_unif=scipy.stats.rankdata(data)/(data.size+1)

    if plot==True:
        
        fig,ax=plt.subplots(ncols=2,figsize=(8,4))
        
        ax[0].hist(data, bins=25, density=True, rwidth=0.8, color='cornflowerblue')
        ax[0].set_ylabel('Normalised Occurrence')
        ax[0].set_xlabel('Data in data scale')
        
        ax[1].hist(data_unif, bins=25, density=True, rwidth=0.8, color='darkorange')
        ax[1].set_ylabel('Normalised Occurrence')
        ax[1].set_xlabel('Data on uniform margins')
        
        plt.show()

    return data_unif

def transform_from_uniform_margins_to_data_scale(data,plot=False):
    
    print('hello')