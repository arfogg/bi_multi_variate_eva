# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:40:02 2023

@author: A R Fogg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from copulas.multivariate import GaussianMultivariate

def fit_copula_bivariate(x_extremes, y_extremes, x_name, y_name):
    """
    Function to fit a copula to x and y extremes

    Parameters
    ----------
    x_extremes : np.array
        X extremes on uniform margins.
    y_extremes : np.array
        Y extremes on uniform margins.
    x_name : string
        String descriptor for x values e.g. 'AE'.
    y_name : string
        String descriptor for y values e.g. 'AL'.

    Returns
    -------
    copula : copulas copula
        fitted copula

    """
    
    # First, check same number of extremes in x and y
    if np.array(x_extremes).size == np.array(y_extremes).size:
        print('Fitting copula to parsed extremes')
    else:
        print('ERROR: fit_copula_to_extremes.fit_copula_bivariate')
        print('x_extremes and y_extremes must have the same length')
        raise NameError('x_extremes and y_extremes must have the same length')
    
    # Format the extremes to how copulas wants them
    copula_df=pd.DataFrame({x_name:x_extremes,
                            y_name:y_extremes})
    
    # Initialise the copula - testing with GaussianMultivariate
    copula = GaussianMultivariate()
    # Fit copula to the extremes
    copula.fit(copula_df)
    
    return copula

def qualitative_copula_fit_check_bivariate(x_extremes, y_extremes, x_sample,
                                    y_sample, x_name, y_name):
    
    
    fig,ax=plt.subplots()
    
    # Plot detected extremes
    ax.plot(x_extremes, y_extremes,linewidth=0.0,marker='^', fillstyle='none', color='black', label='data')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    
    
    ax.plot(x_sample,y_sample, linewidth=0.0, marker='o', fillstyle='none', color='purple', label='copula sample')
    
    ax.legend()
    
    plt.show()
    
    csize=15
    fig, ax=plt.subplots(ncols=2, figsize=(18,7))
    
    # OBSERVED EXTREMES
    h_data=ax[0].hist2d(x_extremes, y_extremes, bins=[10,10], cmap='magma', cmin=1)
    
    # Some decor
    ax[0].set_xlabel(x_name, fontsize=csize)
    ax[0].set_ylabel(y_name, fontsize=csize)  
    for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
        label.set_fontsize(csize)
    ax[0].set_facecolor('darkgray')
    t_data=ax[0].text(0.06, 0.94, '(a)', transform=ax[0].transAxes, fontsize=csize,  va='top', ha='left')
    t_data.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0].set_title('Observed Extremes', fontsize=csize)
    
    # Colourbar
    cbar_data=fig.colorbar(h_data[3], ax=ax[0])
    cbar_data.set_label('Occurrence', fontsize=csize)
    cbar_data.ax.tick_params(labelsize=csize)

    
    return x_sample,y_sample
    

    # COPULA SAMPLE

    h_sample=ax[1].hist2d(x_sample, y_sample, bins=[10,10], cmap='magma', cmin=1)
    
    # Some decor
    ax[1].set_xlabel(x_name, fontsize=csize)
    ax[1].set_ylabel(y_name, fontsize=csize)  
    for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()):
        label.set_fontsize(csize)
    ax[1].set_facecolor('darkgray')
    t_data=ax[1].text(0.06, 0.94, '(b)', transform=ax[1].transAxes, fontsize=csize,  va='top', ha='left')
    t_data.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[1].set_title('Copula Sample (data scale)', fontsize=csize)
    
    # Colourbar
    cbar_sample=fig.colorbar(h_sample[3], ax=ax[1])
    cbar_sample.set_label('Occurrence', fontsize=csize)
    cbar_sample.ax.tick_params(labelsize=csize)
    
    # OBSERVED - COPULA
    
    
    
    