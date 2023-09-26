# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:40:39 2023

@author: A R Fogg
"""

#import copulas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import transform_uniform_margins

def calculate_return_period(copula, sample, block_size=pd.to_timedelta("365.2425D")):
    """
    Calculate the return period for a given list of sample x and y

    Parameters
    ----------
    copula : copulas copula
        Copula that has been fit to some data
    sample : pd.DataFrame
        Two columns with names copula.columns, containing x and y values to find the return period for.
    block_size : pd.Timedelta, optional
        Size over which block maxima have been found. The default is pd.to_timedelta("365.2425D").

    Returns
    -------
    return_period : np.array
        Return periods for given sample.

    """
    
    print('Calculating the return period over parsed copula and sample')
    
    # Calculate the CDF value for each point in sample
    CDF=copula.cumulative_distribution(sample)
    
    # Calculate the number of extremes in a year
    n_extremes_per_year=pd.to_timedelta("365.2425D")/block_size
    
    # Calculate the return period
    return_period=(1.0/n_extremes_per_year)*(1.0/(1-CDF))
    
    return return_period

def plot_return_period_as_function_x_y(copula,min_x,max_x,min_y,max_y,x_name,y_name,n_samples=1000,
                                       block_size=pd.to_timedelta("365.2425D")):
    
    # !!! should not be transformed empirically!!!
    # Create a sample
    sample=pd.DataFrame({x_name:transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(np.linspace(min_x,max_x,n_samples)),
                         y_name:transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(np.linspace(min_y,max_y,n_samples))})
    
    # Create sample grid
    xv, yv = np.meshgrid(sample[x_name], sample[y_name])
    
    # sample_grid=pd.DataFrame({x_name:xv.ravel(), y_name:yv.ravel()})
    # # Calculate the return period
    # return_period=calculate_return_period(copula, sample_grid, block_size=block_size)

    # # Initialise plotting window
    # fig,ax=plt.subplots()
    
    # # Plot return period as a function of x and y
    # ax.scatter(sample_grid[x_name], sample_grid[y_name], c=return_period, norm='linear')
    
    # plt.show()
    
    
    mid_point_x=(xv[1:,1:]+xv[:-1,:-1])/2
    mid_point_y=(yv[1:,1:]+yv[:-1,:-1])/2

    raveled_mid_point_x=mid_point_x.ravel()
    raveled_mid_point_y=mid_point_y.ravel()

    sample_grid=pd.DataFrame({x_name:raveled_mid_point_x, y_name:raveled_mid_point_y})
    return_period=calculate_return_period(copula, sample_grid, block_size=block_size)

    shaped_return_period=return_period.reshape(mid_point_x.shape)




    fig,ax=plt.subplots()

    print(xv.shape, yv.shape, shaped_return_period.shape)
    pcm=ax.pcolormesh(xv,yv,shaped_return_period, cmap='plasma', norm=colors.LogNorm(vmin=shaped_return_period.min(),
                  vmax=shaped_return_period.max()*0.60))
    
    fig.colorbar(pcm, ax=ax, extend='max', label='Return Period (units??)')
    
    # Some Decor
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    
    
    
    
    
    