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

def plot_return_period_as_function_x_y(copula,min_x,max_x,min_y,max_y,x_name,y_name,x_gevd_fit_params, y_gevd_fit_params,
                                       n_samples=1000,block_size=pd.to_timedelta("365.2425D")):
    
    # !!! should not be transformed empirically!!!
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
    
    
    # sample_grid=pd.DataFrame({x_name:xv.ravel(), y_name:yv.ravel()})
    # # Calculate the return period
    # return_period=calculate_return_period(copula, sample_grid, block_size=block_size)

    # # Initialise plotting window
    # fig,ax=plt.subplots()
    
    # # Plot return period as a function of x and y
    # ax.scatter(sample_grid[x_name], sample_grid[y_name], c=return_period, norm='linear')
    
    # plt.show()
    
    
    mid_point_x_um=(xv_um[1:,1:]+xv_um[:-1,:-1])/2
    mid_point_y_um=(yv_um[1:,1:]+yv_um[:-1,:-1])/2
    # mid_point_x_ds=(xv_ds[1:,1:]+xv_ds[:-1,:-1])/2
    # mid_point_y_ds=(yv_ds[1:,1:]+yv_ds[:-1,:-1])/2


    raveled_mid_point_x=mid_point_x_um.ravel()
    raveled_mid_point_y=mid_point_y_um.ravel()

    #sample_grid=pd.DataFrame({x_name:raveled_mid_point_x, y_name:raveled_mid_point_y})
    sample_grid=np.array([raveled_mid_point_x,raveled_mid_point_y]).T
    return_period=calculate_return_period(copula, sample_grid, block_size=block_size)

    shaped_return_period=return_period.reshape(mid_point_x_um.shape)




    fig,ax=plt.subplots()

    #print(xv.shape, yv.shape, shaped_return_period.shape)
    pcm=ax.pcolormesh(xv_ds,yv_ds,shaped_return_period, cmap='plasma', norm=colors.LogNorm(vmin=shaped_return_period.min(),
                  vmax=shaped_return_period.max()*0.60))
    
    fig.colorbar(pcm, ax=ax, extend='max', label='Return Period (units??)')
    
    # Some Decor
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    
    # # Convert axes labels from uniform margins to data scale
    # xticks=transform_uniform_margins.transform_from_uniform_margins_to_data_scale(np.array(ax.get_xticks()),x_gevd_fit_params, plot=False)
    # yticks=transform_uniform_margins.transform_from_uniform_margins_to_data_scale(np.array(ax.get_yticks()),y_gevd_fit_params, plot=False)
    
    # ax.set_xticks(ax.get_xticks(),labels=xticks)
    # ax.set_yticks(ax.get_yticks(),labels=yticks)
    
    
    
    
    #THIS BELOW WORKED, BUT TICKS AREN'T EVENLY SPACED
    # THEY ARE EVENLY SPACED IN UNIFORM MARGINS
    # SO WE NEED TO DETERMINE BOX EDGES IN DATA SCALE THEN CONVERT
    
    # # Determine new axis labels, equally spaced in data scale
    # rounded_min_x=round(min_x*1.1,-1) if min_x>=10.0 else round(min_x*1.1,0)
    # rounded_max_x=round(max_x*0.9,-1) if max_x>=10.0 else round(max_x*0.9,0)
    # xticks_data_scale=np.linspace(rounded_min_x,rounded_max_x,5)
    # xticks_uniform_margins=transform_uniform_margins.transform_from_data_scale_to_uniform_margins_using_CDF(xticks_data_scale, x_gevd_fit_params, distribution=x_gevd_fit_params.distribution_name[0], plot=False)
    # print(xticks_data_scale)
    # print(xticks_uniform_margins)
    # ax.set_xticks(xticks_uniform_margins,labels=xticks_data_scale)
    