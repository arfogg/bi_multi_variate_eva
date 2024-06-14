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
import seaborn as sns

from matplotlib.transforms import TransformedBbox
from matplotlib.transforms import Bbox

import transform_uniform_margins
import plotting_utils

def calculate_return_period(copula, sample_grid, block_size=pd.to_timedelta("365.2425D")):
    """
    Calculate the return period for a given list of sample x and y

    Parameters
    ----------
    copula : copulas copula
        Copula that has been fit to some data
    sample_grid : pd.DataFrame
        Two columns with names same as copula, containing x and y values 
        to find the return period for.
    block_size : pd.Timedelta, optional
        Size over which block maxima have been found. The default 
        is pd.to_timedelta("365.2425D").

    Returns
    -------
    return_period : np.array
        Return periods for given sample.

    """
    
    print('Calculating the return period over parsed copula and sample')
    
    # Calculate the CDF value for each point in sample
    CDF=copula.cumulative_distribution(sample_grid)
    
    # Calculate the number of extremes in a year
    n_extremes_per_year=pd.to_timedelta("365.2425D")/block_size
    
    # Calculate the return period (in years!)
    # See Coles 2001 textbook pages 81-82
    return_period=(1.0/n_extremes_per_year)*(1.0/(1-CDF))
    
    # if np.isfinite(np.sum(return_period)) == False:
    #     breakpoint()
    
    return return_period

def estimate_return_period_ci(bs_copula_arr, n_bootstrap,
                              sample_grid, block_size=pd.to_timedelta("365.2425D"),
                              ci_percentiles=[2.5, 97.5]):
    
    print('banana')
    rp = np.full((sample_grid.shape[0], n_bootstrap), np.nan)
    for i in range(n_bootstrap):
        print('Bootstrap ',i)
        rp[:,i] = calculate_return_period(bs_copula_arr[i], sample_grid, 
                                          block_size=block_size)
        
    # Looping through each grid pixel
    ci = np.full([sample_grid.shape[0], 2], np.nan)
    n = np.full(sample_grid.shape[0], np.nan)
    for j in range(sample_grid.shape[0]):
        # First, select the Bootstraps where return_period is finite
        #   Infinite return_period means CDF->1, which indicates
        #   the bootstrap didn't contain the full extent of the actual
        #   observed extrema. Hence the copula CDF was saying "out 
        #   of bounds!" to the requested X/Y point. So, we use only
        #   finite return periods to calculate the CI, and retain the
        #   number of bootstraps contributing to each point.
        rp_clean_index, = np.where(np.isfinite(rp[j,:]))
        rp_clean = rp[j,rp_clean_index] if rp_clean_index.size > 0 else rp[j,:]
        ci[j,:] = np.percentile(rp_clean, ci_percentiles)
        n[j] = rp_clean.size    
    
    # print('nans in rp?',np.isnan(np.sum(rp)))
    # print('rp.shape', rp.shape)
    #     #print(sample_grid.shape[0], rp[:,i].shape)
    # # Can't allocate enough memory to do in one line 
    # #   for 1000 bootstraps
    # print('HELLO USING NANPERCENTILE')
    # ci = np.percentile(rp, ci_percentiles, axis=1)
    # print('nans in ci?',np.isnan(np.sum(ci)))
    # print('ci.shape', ci.shape)
    
    
    # quant = np.quantile(rp, [0.25, 0.975], axis=1)
    # print('nans in quant?',np.isnan(np.sum(quant)))
    # print('quant.shape', quant.shape)    
    
    return  rp, ci, n


def generate_sample_grid(min_x, max_x, min_y, max_y, 
                         x_name, y_name,
                         n_samples=1000):
    # Create a sample
    sample_um=pd.DataFrame({x_name:transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(
                                    np.linspace(min_x,max_x,n_samples)),
                            y_name:transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(
                                    np.linspace(min_y,max_y,n_samples))})
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
    mid_point_x_ds=(xv_ds[1:,1:]+xv_ds[:-1,:-1])/2
    mid_point_y_ds=(yv_ds[1:,1:]+yv_ds[:-1,:-1])/2
    
    # Reshape
    raveled_mid_point_x=mid_point_x_um.ravel()
    raveled_mid_point_y=mid_point_y_um.ravel()
    sample_grid=np.array([raveled_mid_point_x,raveled_mid_point_y]).T

    return sample_grid, xv_ds, yv_ds, mid_point_x_ds, mid_point_y_ds    

def plot_return_period_as_function_x_y(copula, min_x, max_x, min_y, max_y, 
                                       x_name, y_name, 
                                       x_gevd_fit_params, y_gevd_fit_params,
                                       x_label, y_label, 
                                       bs_copula_arr, n_bootstrap,
                                       sample_grid=None,
                                       xv_ds=None, yv_ds=None,
                                       mid_point_x_ds=None, mid_point_y_ds=None,
                                       return_period=None, ci=None, n=None,
                                       n_samples=1000,
                                       block_size=pd.to_timedelta("365.2425D"),
                                       contour_levels=[1/12,0.5,1.0,10.0],
                                       contour_colors=['white','white','white','black'],
                                       lower_ax_limit_contour_index=1,
                                       ci_percentiles=[2.5, 97.5],
                                       fontsize=15):
    """
    Function to plot the predicted return period as a function of
    the two input parameters.

    Parameters
    ----------
    copula : copulas copula
        Copula fitted to x and y extremes.
    min_x : float
        Minimum x value that the return period will be evaluated at.
    max_x : float
        Maximum x value that the return period will be evaluated at.
    min_y : float
        Minimum y value that the return period will be evaluated at.
    max_y : float
        Maximum y value that the return period will be evaluated at.
    x_name : string
        Name for x, used for pandas.DataFrame column names.
    y_name : string
        Name for y, used for pandas.DataFrame column names.
    x_gevd_fit_params : pandas.DataFrame
        Dataframe containing fit parameters for x, output from 
        fit_model_to_extremes.fit_gevd_or_gumbel.
    y_gevd_fit_params : pandas.DataFrame
        Dataframe containing fit parameters for y, output from 
        fit_model_to_extremes.fit_gevd_or_gumbel.
    x_label : string
        Name for x, used for labelling plots.
    y_label : string
        Name for y, used for labelling plots.
    bs_copula_arr : list of copulas copulae
        Python list of length n_bootstrap containing
        copulae for each bootstrapped set of extrema.
    n_bootstrap : int
        Number of bootstrapped extrema generated.
    n_samples : int, optional
        Number of points for x and y axes. So return period is evaluated 
        across n_samples x n_samples size grid. The default is 1000.
    block_size : pd.timedelta, optional
        Block size used in the block maxima extreme selection. The 
        default is pd.to_timedelta("365.2425D").
    contour_levels : list, optional
        Return period values at which contours will be drawn. The 
        default is [1/12,0.5,1.0,10.0].
    lower_ax_limit_contour_index : int, optional
        Used to decide the lower axes limits for x and y. The default is 1.
    ci_percentiles : list, optional
        Upper and lower percentiles for the confidence interval
        plots. The default is [2.5, 97.5].

    Returns
    -------
    fig : matplotlib figure
        Figure containing the return period plot.
    ax : matplotlib axes
        Axes within fig.

    """
    
    print('Creating a 2D return period plot with confidence intervals')
    # Adjust fontsize for all text
    plt.rcParams['font.size'] = fontsize
    
    # Create sample_grid etc if not parsed in
    if (sample_grid is None) | (xv_ds is None) | (yv_ds is None) | (mid_point_x_ds is None) | (mid_point_y_ds is None):
        sample_grid, xv_ds, yv_ds, mid_point_x_ds, mid_point_y_ds = generate_sample_grid(
                                 min_x, max_x, min_y, max_y, 
                                 x_name, y_name,
                                 n_samples=1000)
    
    # Calculate return period if not parsed
    if return_period is None:
        return_period=calculate_return_period(copula, sample_grid, block_size=block_size)

    # Reshape for mesh grid
    shaped_return_period=return_period.reshape(mid_point_x_ds.shape)

    # Calculate confidence intervals if needed
    if (ci is None) | (n is None):
        rp, ci, n = estimate_return_period_ci(bs_copula_arr, n_bootstrap,
                                      sample_grid, block_size=block_size,
                                      ci_percentiles=ci_percentiles)

    # Initialise plot
    fig,ax=plt.subplots(nrows=2, ncols=2, figsize=(16,12))
   


  
    
    
    # ----- RETURN PERIOD -----
    rp_cbar_norm = colors.LogNorm(vmin=np.quantile(shaped_return_period, 0.1),
                                  vmax=np.quantile(shaped_return_period, 0.999))
    # Plot return period as function of x and y in data scale
    pcm=ax[0,0].pcolormesh(xv_ds,yv_ds,shaped_return_period, cmap='plasma',
                         norm = rp_cbar_norm)
    
    # Contours
    contours = ax[0,0].contour(mid_point_x_ds, mid_point_y_ds, shaped_return_period, contour_levels, 
                           colors=contour_colors)    
    for c in contours.collections:
        c.set_clip_on(True)
    clabels = plotting_utils.contour_labels(contours, xpad=max_x*0.005, sides=['left', 'right'], 
                             color='black', ha='left', va='center')

    # Colourbar
    cbar=fig.colorbar(pcm, ax=ax[0,0], extend='max', label='Return Period (years)',
                      pad=0.06)
    cbar.add_lines(contours)  
    
    # Some Decor
    ax[0,0].set_xlabel(x_label)
    ax[0,0].set_ylabel(y_label)
    ax[0,0].set_title('(a) Return Period\n'+r'($\tau_{max}$ = '+str(round(np.nanmax(shaped_return_period),2))+' years)')

    # ----- LOWER QUANTILE -----

    # Plot lower quantile as function of x and y in data scale
    lower_quantile = ci[:,0].reshape(mid_point_x_ds.shape)
    lci_cbar_norm = colors.LogNorm(vmin=np.nanquantile(lower_quantile, 0.1),
                                  vmax=np.nanquantile(lower_quantile, 0.999))
    pcm_lq=ax[0,1].pcolormesh(xv_ds,yv_ds,lower_quantile, cmap='magma', 
                            norm = lci_cbar_norm)
    # Contours
    contours_lq = ax[0,1].contour(mid_point_x_ds, mid_point_y_ds, lower_quantile, contour_levels, 
                           colors=contour_colors)    
    for c in contours_lq.collections:
        c.set_clip_on(True)
    clabels = plotting_utils.contour_labels(contours_lq, xpad=max_x*0.005, sides=['left', 'right'], 
                             color='black', ha='left', va='center')    

    # Colourbar
    cbar=fig.colorbar(pcm_lq, ax=ax[0,1], extend='max', label='Return Period (years)')
    cbar.add_lines(contours_lq)
    # Some Decor
    ax[0,1].set_xlabel(x_label)
    ax[0,1].set_ylabel(y_label)
    
    ax[0,1].set_title('(b) '+str(ci_percentiles[1]-ci_percentiles[0])+'% Confidence Interval (lower)\n'+r'($\tau_{max}$ = '+str(round(np.nanmax(lower_quantile),2))+' years)')
    
    # ----- UPPER QUANTILE -----
    # Plot upper quantile as function of x and y in data scale
    upper_quantile = ci[:,1].reshape(mid_point_x_ds.shape)
    uci_cbar_norm = colors.LogNorm(vmin=np.nanquantile(upper_quantile, 0.1),
                                  vmax=np.nanquantile(upper_quantile, 0.999))
    pcm_uq=ax[1,1].pcolormesh(xv_ds,yv_ds,upper_quantile, cmap='magma',
                            norm = uci_cbar_norm)    
    # Contours
    contours_uq = ax[1,1].contour(mid_point_x_ds, mid_point_y_ds, upper_quantile, contour_levels, 
                           colors=contour_colors)    
    for c in contours_uq.collections:
        c.set_clip_on(True)
    clabels = plotting_utils.contour_labels(contours_uq, xpad=max_x*0.005, sides=['left', 'right'], 
                             color='black', ha='left', va='center')        
    # Colourbar
    cbar=fig.colorbar(pcm_uq, ax=ax[1,1], extend='max', label='Return Period (years)')
    cbar.add_lines(contours_uq)  
     
    # Some Decor
    ax[1,1].set_xlabel(x_label)
    ax[1,1].set_ylabel(y_label)
    ax[1,1].set_title('(d) '+str(ci_percentiles[1]-ci_percentiles[0])+'% Confidence Interval (upper)\n'+r'($\tau_{max}$ = '+str(round(np.nanmax(upper_quantile),2))+' years)')


    # ----- n -----
    # Plot number of bootstraps contributing as function of x and y in data scale
    number = n.reshape(mid_point_x_ds.shape)
    pcm_n=ax[1,0].pcolormesh(xv_ds,yv_ds,number, cmap='inferno_r')    
    
    # # Contours
    # contours_uq = ax[2].contour(mid_point_x_ds, mid_point_y_ds, upper_quantile, contour_levels, 
    #                        colors=contour_colors)    
    # for c in contours_uq.collections:
    #     c.set_clip_on(True)
    # clabels = plotting_utils.contour_labels(contours_uq, xpad=max_x*0.005, sides=['left', 'right'], 
    #                          color='black', ha='left', va='center')        
    # Colourbar
    cbar=fig.colorbar(pcm_n, ax=ax[1,0], label='Number of bootstraps')
    # cbar.add_lines(contours_uq)  
     
    # Some Decor
    ax[1,0].set_xlabel(x_label)
    ax[1,0].set_ylabel(y_label)
    ax[1,0].set_title('(c) Number of contributing bootstraps\nminimum = '+str(int(np.nanmin(n))))















    # Set x and y limits
    # Work out x and y limits based on parsed contour index
    xy_contour_limit=contours.allsegs[lower_ax_limit_contour_index][0]
    xlim=np.min(xy_contour_limit[:,0])*0.9
    ylim=np.min(xy_contour_limit[:,1])*0.9
    for i,a in np.ndenumerate(ax):
        a.set_xlim(left=xlim)
        a.set_ylim(bottom=ylim) 

    fig.tight_layout()
    #breakpoint()
    return fig, ax






def plot_return_period_as_function_x_y_3d(copula,min_x,max_x,min_y,max_y,x_name,y_name,x_gevd_fit_params, y_gevd_fit_params,
                                       x_label, y_label, n_samples=1000,block_size=pd.to_timedelta("365.2425D"),
                                       contour_levels=[1/12,0.5,1.0,10.0], lower_ax_limit_contour_index=1):
    """
    Function to plot a 3D visualisation of the return period plot.
    WARNING: this is a work-in-progress!

    Parameters
    ----------       
    copula : copulas copula
        Copula fitted to x and y extremes.
    min_x : float
        Minimum x value that the return period will be evalutated at.
    max_x : float
        Maximum x value that the return period will be evalutated at.
    min_y : float
        Minimum y value that the return period will be evalutated at.
    max_y : float
        Maximum y value that the return period will be evalutated at.
    x_name : string
        Name for x, used for pandas.DataFrame column names.
    y_name : string
        Name for y, used for pandas.DataFrame column names.
    x_gevd_fit_params : pandas.DataFrame
        Dataframe containing fit parameters for x, output from 
        fit_model_to_extremes.fit_gevd_or_gumbel.
    y_gevd_fit_params : pandas.DataFrame
        Dataframe containing fit parameters for y, output from 
        fit_model_to_extremes.fit_gevd_or_gumbel.
    x_label : string
        Name for x, used for labelling plots.
    y_label : string
        Name for y, used for labelling plots.
    n_samples : int, optional
        Number of points for x and y axes. So return period is evaluated 
        across n_samples x n_samples size grid. The default is 1000.
    block_size : pd.timedelta, optional
        Block size used in the block maxima extreme selection. The 
        default is pd.to_timedelta("365.2425D").
    contour_levels : list, optional
        Return period values at which contours will be drawn. The 
        default is [1/12,0.5,1.0,10.0].
    lower_ax_limit_contour_index : int, optional
        Used to decide the lower axes limits for x and y. The default is 1.

    Returns
    -------
    fig : matplotlib figure
        Figure containing 3D return period plot.
    ax : matplotlib axes
        Axes within fig.

    """
    print('WARNING: plot_return_period_as_function_x_y_3d is a work-in-progress!')
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
    mid_point_x_ds=(xv_ds[1:,1:]+xv_ds[:-1,:-1])/2
    mid_point_y_ds=(yv_ds[1:,1:]+yv_ds[:-1,:-1])/2

    # Reshape
    raveled_mid_point_x=mid_point_x_um.ravel()
    raveled_mid_point_y=mid_point_y_um.ravel()
    sample_grid=np.array([raveled_mid_point_x,raveled_mid_point_y]).T
    
    # Calculate return period
    return_period=calculate_return_period(copula, sample_grid, block_size=block_size)
    # Reshape for mesh grid
    shaped_return_period=return_period.reshape(mid_point_x_um.shape)

    # Initialise plot
    fig,ax=plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf = ax.plot_surface(mid_point_x_ds,mid_point_y_ds,shaped_return_period, cmap='plasma',
                           linewidth=0, antialiased=False)
    
    #print(np.min(shaped_return_period),np.max(shaped_return_period))
    
    ax.set_zscale('log')
    
    
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    ## A StrMethodFormatter is used automatically
    #ax.zaxis.set_major_formatter('{x:.02f}')
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()
    
    
    # CUT OUT THE CORNER OF HIGH VALUES AND HAVE A LOOK AT THE
    #   VARIABILITY ELSEWHERE
   
    
    # # Plot return period as function of x and y in data scale
    # #pcm=ax.pcolormesh(xv_ds,yv_ds,shaped_return_period, cmap='plasma', norm=colors.LogNorm(vmin=shaped_return_period.min(),
    # #              vmax=shaped_return_period.max()*0.10))
    # pcm=ax.pcolormesh(xv_ds,yv_ds,shaped_return_period, cmap='plasma', norm=colors.LogNorm(vmin=np.quantile(shaped_return_period, 0.1),
    #               vmax=np.quantile(shaped_return_period, 0.999)))

    # # Contours
    # contours=ax.contour(mid_point_x_ds,mid_point_y_ds,shaped_return_period, contour_levels, colors='white')
    # for c in contours.collections:
    #     c.set_clip_on(True)
    # clabels=ax.clabel(contours, inline=True, fmt="%0.1f")

    # Work out x and y limits based on parsed contour index
    # lower_ax_limit_contour_index=1
    # xy_contour_limit=contours.allsegs[lower_ax_limit_contour_index][0]
    # xlim=np.min(xy_contour_limit[:,0])*0.9
    # ylim=np.min(xy_contour_limit[:,1])*0.9
    # ax.set_xlim(left=xlim)
    # ax.set_ylim(bottom=ylim)
    
    # # Colourbar
    # cbar=fig.colorbar(pcm, ax=ax, extend='max', label='Return Period (years)')
    # cbar.add_lines(contours)  
    
    # # Some Decor
    # ax.set_xlabel(x_label)
    # ax.set_ylabel(y_label)
     
    return fig, ax


def plot_return_period_as_function_x_y_experimenting(copula,min_x,max_x,min_y,max_y,x_name,y_name,x_gevd_fit_params, y_gevd_fit_params,
                                       x_label, y_label, n_samples=1000,block_size=pd.to_timedelta("365.2425D"),
                                       contour_levels=[1/12,0.5,1.0,10.0], lower_ax_limit_contour_index=1):
    """
    Function with various experimental ways for showing the return
    period plot.
    WARNING: this is a work-in-progress!

    Parameters
    ----------    
    copula : copulas copula
        Copula fitted to x and y extremes.
    min_x : float
        Minimum x value that the return period will be evalutated at.
    max_x : float
        Maximum x value that the return period will be evalutated at.
    min_y : float
        Minimum y value that the return period will be evalutated at.
    max_y : float
        Maximum y value that the return period will be evalutated at.
    x_name : string
        Name for x, used for pandas.DataFrame column names.
    y_name : string
        Name for y, used for pandas.DataFrame column names.
    x_gevd_fit_params : pandas.DataFrame
        Dataframe containing fit parameters for x, output from 
        fit_model_to_extremes.fit_gevd_or_gumbel.
    y_gevd_fit_params : pandas.DataFrame
        Dataframe containing fit parameters for y, output from 
        fit_model_to_extremes.fit_gevd_or_gumbel.
    x_label : string
        Name for x, used for labelling plots.
    y_label : string
        Name for y, used for labelling plots.
    n_samples : int, optional
        Number of points for x and y axes. So return period is evaluated 
        across n_samples x n_samples size grid. The default is 1000.
    block_size : pd.timedelta, optional
        Block size used in the block maxima extreme selection. The 
        default is pd.to_timedelta("365.2425D").
    contour_levels : list, optional
        Return period values at which contours will be drawn. The 
        default is [1/12,0.5,1.0,10.0].
    lower_ax_limit_contour_index : int, optional
        Used to decide the lower axes limits for x and y. The default is 1.


    Returns
    -------
    None.

    """
    print('WARNING: plot_return_period_as_function_x_y_experimenting is a work-in-progress!')
    
    # Set fontsize
    csize=15
    
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
    mid_point_x_ds=(xv_ds[1:,1:]+xv_ds[:-1,:-1])/2
    mid_point_y_ds=(yv_ds[1:,1:]+yv_ds[:-1,:-1])/2

    # Reshape
    raveled_mid_point_x=mid_point_x_um.ravel()
    raveled_mid_point_y=mid_point_y_um.ravel()
    sample_grid=np.array([raveled_mid_point_x,raveled_mid_point_y]).T
    
    # Calculate return period
    return_period=calculate_return_period(copula, sample_grid, block_size=block_size)
    # Reshape for mesh grid
    shaped_return_period=return_period.reshape(mid_point_x_um.shape)


    # (1) 2D COLOUR PLOT WITH CONTOURS
    # Initialise plot
    fig, ax=plt.subplots(ncols=3, figsize=(27,7))

    # NO FIDDLING, AS IT IS
    # Plot return period as function of x and y in data scale
    pcm=ax[0].pcolormesh(xv_ds,yv_ds,shaped_return_period, cmap='plasma', norm=colors.LogNorm())

    # Contours
    contours=ax[0].contour(mid_point_x_ds,mid_point_y_ds,shaped_return_period, contour_levels, colors='white')
    for c in contours.collections:
        c.set_clip_on(True)
    clabels=ax[0].clabel(contours, inline=True, fmt="%0.1f")
    
    # Colourbar
    cbar=fig.colorbar(pcm, ax=ax[0], extend='max', label='Return Period (years)')
    cbar.add_lines(contours)  
    
    # Some Decor
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel(y_label)
    
    # Title
    ax[0].set_title('No axes limit, unsaturated cbar', fontsize=csize)


    # LIMIT AXES, SATURATE CBAR
    # Plot return period as function of x and y in data scale
    pcm=ax[1].pcolormesh(xv_ds,yv_ds,shaped_return_period, cmap='plasma', norm=colors.LogNorm(vmin=np.quantile(shaped_return_period, 0.1),
                  vmax=np.quantile(shaped_return_period, 0.999)))

    # Contours
    contours=ax[1].contour(mid_point_x_ds,mid_point_y_ds,shaped_return_period, contour_levels, colors='white')
    for c in contours.collections:
        c.set_clip_on(True)
    clabels=ax[1].clabel(contours, inline=True, fmt="%0.1f")

    # Work out x and y limits based on parsed contour index
    lower_ax_limit_contour_index=1
    xy_contour_limit=contours.allsegs[lower_ax_limit_contour_index][0]
    xlim=np.min(xy_contour_limit[:,0])*0.9
    ylim=np.min(xy_contour_limit[:,1])*0.9
    ax[1].set_xlim(left=xlim)
    ax[1].set_ylim(bottom=ylim)
    
    # Colourbar
    cbar=fig.colorbar(pcm, ax=ax[1], extend='max', label='Return Period (years)')
    cbar.add_lines(contours)  
    
    # Some Decor
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel(y_label)
    
    # Title
    ax[1].set_title('Limited axes limits (higher data shown),\nsaturated cbar 0.1-0.999 quantile', fontsize=csize)

    
    # GET RID OF VERY HIGH BIT
    # Determine corner to remove
    
    # Plot return period as function of x and y in data scale
    pcm=ax[2].pcolormesh(xv_ds,yv_ds,shaped_return_period, cmap='plasma', norm=colors.LogNorm(vmin=np.quantile(shaped_return_period, 0.1),
                  vmax=np.quantile(shaped_return_period, 0.99)))

    # Contours
    contours=ax[2].contour(mid_point_x_ds,mid_point_y_ds,shaped_return_period, contour_levels, colors='white')
    for c in contours.collections:
        c.set_clip_on(True)
    clabels=ax[2].clabel(contours, inline=True, fmt="%0.1f")

    # Work out x and y limits based on parsed contour index
    lower_ax_limit_contour_index=1
    xy_contour_limit=contours.allsegs[lower_ax_limit_contour_index][0]
    xlim=np.max(xy_contour_limit[:,0])*0.9
    ylim=np.max(xy_contour_limit[:,1])*0.9
    ax[2].set_xlim(right=xlim)
    ax[2].set_ylim(top=ylim)
    
    # Colourbar
    cbar=fig.colorbar(pcm, ax=ax[2], extend='max', label='Return Period (years)')
    cbar.add_lines(contours)  
    
    # Some Decor
    ax[2].set_xlabel(x_label)
    ax[2].set_ylabel(y_label)
    
    # Title
    ax[2].set_title('Limited axes limits (lower data shown),\nsaturated cbar 0.1-0.99 quantile', fontsize=csize)    



    # CONFUSION MATRIX-LIKE PLOT
    n_squares=15
    # Create a sample
    sample_um=pd.DataFrame({x_name:transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(np.linspace(min_x,max_x,n_squares)),
                         y_name:transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(np.linspace(min_y,max_y,n_squares))})
    sample_ds=pd.DataFrame({x_name:np.linspace(min_x,max_x,n_squares),
                                    y_name:np.linspace(min_y,max_y,n_squares)})

    # Create sample grid
    xv_um, yv_um = np.meshgrid(sample_um[x_name], sample_um[y_name])    #uniform margins
    xv_ds, yv_ds = np.meshgrid(sample_ds[x_name], sample_ds[y_name])    #data scale
    # mesh grid on uniform margins for calculating, in data scale
    #   for plotting
      
    # Determine mid point of each pixel to calculate return
    #   period for
    mid_point_x_um=(xv_um[1:,1:]+xv_um[:-1,:-1])/2
    mid_point_y_um=(yv_um[1:,1:]+yv_um[:-1,:-1])/2
    mid_point_x_ds=(xv_ds[1:,1:]+xv_ds[:-1,:-1])/2
    mid_point_y_ds=(yv_ds[1:,1:]+yv_ds[:-1,:-1])/2

    # Reshape
    raveled_mid_point_x=mid_point_x_um.ravel()
    raveled_mid_point_y=mid_point_y_um.ravel()
    raveled_mid_point_x_ds=mid_point_x_ds.ravel()
    raveled_mid_point_y_ds=mid_point_y_ds.ravel()
    
    sample_grid=np.array([raveled_mid_point_x,raveled_mid_point_y]).T
    
    # Calculate return period
    return_period=calculate_return_period(copula, sample_grid, block_size=block_size)
    # Reshape for mesh grid
    shaped_return_period=return_period.reshape(mid_point_x_um.shape)


    # Initialise figure 
    csize=20
    fig_cm,ax_cm=plt.subplots(figsize=(20,20))

    # Shape the data for the seaborn heatmap
    df=pd.DataFrame({x_label:np.round(raveled_mid_point_x_ds, decimals=2),y_label:np.round(raveled_mid_point_y_ds, decimals=2), 'Return Period':return_period})
    rect=df.pivot(index=y_label, columns=x_label, values='Return Period')
    
    # Plot the heatmap
    ax_cm=sns.heatmap(rect, annot=True, cmap='plasma', fmt='.2g', square=True, ax=ax_cm, 
                      norm=colors.LogNorm(vmin=np.quantile(shaped_return_period, 0.1), vmax=np.quantile(shaped_return_period, 0.99)), 
                      cbar_kws={'label':'Return Period (years)', 'extend':'max'},
                      annot_kws={'fontsize':csize})
    ax_cm.invert_yaxis()
    
    # Some decor
    ax_cm.tick_params(axis='both', labelsize=csize)
    ax_cm.xaxis.get_label().set_fontsize(csize)
    ax_cm.yaxis.get_label().set_fontsize(csize)
    
    cbar = ax_cm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=csize)
    ax_cm.figure.axes[-1].yaxis.label.set_size(csize) #cbar label fontsize

    #return return_period,raveled_mid_point_x,raveled_mid_point_y








    