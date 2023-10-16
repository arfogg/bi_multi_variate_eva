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
    
    # Calculate the return period (in years!)
    # See Coles 2001 textbook pages 81-82
    return_period=(1.0/n_extremes_per_year)*(1.0/(1-CDF))
    
    return return_period

def plot_return_period_as_function_x_y(copula,min_x,max_x,min_y,max_y,x_name,y_name,x_gevd_fit_params, y_gevd_fit_params,
                                       n_samples=1000,block_size=pd.to_timedelta("365.2425D")):
    
    # Create a sample
    sample_um=pd.DataFrame({x_name:transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(np.linspace(min_x,max_x,n_samples)),
                         y_name:transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(np.linspace(min_y,max_y,n_samples))})
    sample_ds=pd.DataFrame({x_name:np.linspace(min_x,max_x,n_samples),
                                    y_name:np.linspace(min_y,max_y,n_samples)})

    # Create sample grid
    xv_um, yv_um = np.meshgrid(sample_um[x_name], sample_um[y_name])    #uniform margins
    xv_ds, yv_ds = np.meshgrid(sample_ds[x_name], sample_ds[y_name])    #data scale
    # mesh grid on uniform margins for calculating, in data scale
    #   for plotting
      
    # Determine mid point of each pixel to calculate return
    #   period for
    mid_point_x_um=(xv_um[1:,1:]+xv_um[:-1,:-1])/2
    mid_point_y_um=(yv_um[1:,1:]+yv_um[:-1,:-1])/2

    # Reshape
    raveled_mid_point_x=mid_point_x_um.ravel()
    raveled_mid_point_y=mid_point_y_um.ravel()
    sample_grid=np.array([raveled_mid_point_x,raveled_mid_point_y]).T
    
    # Calculate return period
    return_period=calculate_return_period(copula, sample_grid, block_size=block_size)
    # Reshape for mesh grid
    shaped_return_period=return_period.reshape(mid_point_x_um.shape)

    # Initialise plot
    fig,ax=plt.subplots()
    
    # Plot return period as function of x and y in data scale
    pcm=ax.pcolormesh(xv_ds,yv_ds,shaped_return_period, cmap='plasma', norm=colors.LogNorm(vmin=shaped_return_period.min(),
                  vmax=shaped_return_period.max()*0.60))
    # Colourbar
    fig.colorbar(pcm, ax=ax, extend='max', label='Return Period (years)')
    
    # Some Decor
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
     