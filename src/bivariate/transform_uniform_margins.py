# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:44:18 2023

@author: A R Fogg
"""

import scipy

import numpy as np
import matplotlib.pyplot as plt

from . import qq_plot
from . import return_period_plot_1d

def transform_from_data_scale_to_uniform_margins_empirically(data, plot=False):
    """
    Transform the data to uniform margins empirically
    by ranking the data

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

def transform_from_data_scale_to_uniform_margins_using_CDF(data, gevd_fitter, plot=False):
    """
    Transforms the data to uniform margins by plugging into the CDF
    (a probability integral transform)
    CDF Distribution is G(x) = some formula
    G(x) = u
    where u is on uniform margins, x is in data scale

    Citation for this equation Coles (2001) page 47
    

    Parameters
    ----------
    data : np.array
        Data in data scale.
    gevd_fitter : gevd_fitter class
        See gevd_fitter.py.
    plot : BOOL, optional
        If plot == True, plots the distributions of data in data
        scale and on uniform margins. The default is False.

    Returns
    -------
    data_unif : np.array
        Data transformed to uniform margins

    """
    
    data_unif = gevd_fitter.frozen_dist.cdf(data)
       
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
    

def transform_from_uniform_margins_to_data_scale(data_unif, gevd_fitter, plot=False):
    """
    Transform the data from uniform margins back to data scale
    using the CDF. 
    CDF Distribution is G(x) = some formula
        G(x) = u
    where u is on uniform margins, x is in data scale
    So we solve for x.

    Parameters
    ----------
    data_unif : np.array
        Data on uniform margins.
    gevd_fitter : gevd_fitter class
        See gevd_fitter.py.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    
    data = gevd_fitter.frozen_dist.ppf(data_unif)
    
    if plot==True:
        fig,ax=plt.subplots(ncols=2,figsize=(8,4))
        
        ax[0].hist(data, bins=25, density=True, rwidth=0.8, color='cornflowerblue')
        ax[0].set_ylabel('Normalised Occurrence')
        ax[0].set_xlabel('Data in data scale')
        
        ax[1].hist(data_unif, bins=25, density=True, rwidth=0.8, color='darkorange')
        ax[1].set_ylabel('Normalised Occurrence')
        ax[1].set_xlabel('Data on uniform margins')
        
        plt.show()
    
    return data

def plot_diagnostic(gevd_fitter, bootstrap_gevd_fit, data_tag, data_units_fm,
                    block_size, um_bins=np.linspace(0,1,11)):
    """
    Function to plot the PDF of extremes and the fitted distribution (left),
    and comparing the empirically and CDF determined data on uniform
    margins (right).

    Parameters
    ----------
    gevd_fitter : gevd_fitter class
        Object containing GEVD fitting information.
    bootstrap_gevd_fit : bootstrap_gevd_fit class
        see bootstrap_gevd_fit.py. Contains attributes
        listed below.
        bs_data : np.array
            Bootstrapped extrema of dimensions n_extrema x 
            n_ci_iterations.
        n_extrema : int
            Number of true extrema.
        n_ci_iterations : int
            Number of bootstraps.
        distribution_name : str
            Name of fitted distribution. Valid options
            'genextreme' and 'gumbel_r'.
        block_size : pd.Timedelta
            Size of block for maxima detection.
        periods : np.array
            Return periods for calculated levels.
        levels : np.array
            Return levels per period per bootstrap. Of shape
            periods.size x n_ci_iterations.
        shape_ : np.array
            Fitted shape parameter for each bootstrap.
        location : np.array
            Fitted location parameter for each bootstrap.
        scale : np.array
            Fitted scale parameter for each bootstrap.
    data_tag : string
        name of data to be put in figure captions etc
    data_units_fm : string
        units for data to be put in axes labels etc
    block_size : pd.Timedelta
        Size over which block maxima have been found,
        e.g. pd.to_timedelta("365.2425D").
    um_bins : np.array
        array defining the edges of the bins for the 
        uniform margins histograms

    Returns
    -------
    None.

    """
    # Initialise figure and axes
    fig,ax=plt.subplots(ncols=3, nrows=2, figsize=(16.5,10.5))
        
    # Plot normalised histogram of extremes
    ax[0,0].hist(gevd_fitter.extremes, bins=np.linspace(np.nanmin(gevd_fitter.extremes),np.nanmax(gevd_fitter.extremes),25), density=True, rwidth=0.8, color='darkgrey', label='extremes')
        
    # Initialise arrays
    model_x=np.linspace(np.nanmin(gevd_fitter.extremes),np.nanmax(gevd_fitter.extremes), 100)
    model_y = gevd_fitter.frozen_dist.pdf(model_x)
    
    # Plot the PDF against x
    ax[0,0].plot(model_x, model_y, color='darkmagenta', label=gevd_fitter.formatted_dist_name)
    
    # Some decor
    ax[0,0].set_ylabel('Normalised Occurrence')
    ax[0,0].set_xlabel(data_tag + ' in data scale ('+data_units_fm+')')    
    t=ax[0,0].text(0.06, 0.94, '(a)', transform=ax[0,0].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0,0].set_title(gevd_fitter.formatted_dist_name+' fit for '+data_tag)

    # Plot normalised histograms of different uniform margins data
    ax[0,1].hist(gevd_fitter.extremes_unif_CDF, bins=um_bins, density=True, rwidth=0.8, color='darkorange', label='using CDF')
    ax[0,1].hist(gevd_fitter.extremes_unif_empirical, bins=um_bins, density=True, rwidth=0.8, color='grey', alpha=0.5, label='empirical')
    
    # Some decor
    ax[0,1].set_ylabel('Normalised Occurrence')
    ax[0,1].set_xlabel(data_tag + ' on uniform margins')
    ax[0,1].legend(loc='upper right')
    t=ax[0,1].text(0.06, 0.94, '(b)', transform=ax[0,1].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0,1].set_title('Data on uniform margins')
    ax[0,0].legend(loc='upper right')
    
    # QQ plot comparing the extremes and their PDF
    ax[1,0]=qq_plot.qq_data_vs_model(ax[1,0], gevd_fitter.extremes, gevd_fitter.extremes_unif_empirical, gevd_fitter, 
                         marker='^', fillstyle='none', color='darkmagenta', title='', 
                         legend_pos='center left')
    
    # Some decor
    ax[1,0].set_xlabel('Extremes')
    ax[1,0].set_ylabel('Fitted '+gevd_fitter.formatted_dist_name+' distribution')
    t=ax[1,0].text(0.06, 0.94, '(d)', transform=ax[1,0].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))    
    
    # QQ plot comparing the uniform margins distributions
    ax[1,1]=qq_plot.qq_data_vs_data(gevd_fitter.extremes_unif_empirical, gevd_fitter.extremes_unif_CDF,
                                    ax[1,1], quantiles=np.linspace(0,100,26), 
                            legend_pos='center left', color='darkorange')
    
    # Some decor
    ax[1,1].set_xlabel('Empirical')
    ax[1,1].set_ylabel('CDF')
    t=ax[1,1].text(0.06, 0.94, '(e)', transform=ax[1,1].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))

    # Return period plot
    ax[0,2]=return_period_plot_1d.return_period_plot(gevd_fitter, bootstrap_gevd_fit, block_size, 
                                                     data_tag, data_units_fm,
                                                     ax[0,2], csize=15, line_color='darkmagenta')
    
    # Some decor
    t=ax[0,2].text(0.06, 0.94, '(c)', transform=ax[0,2].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))

    # Return values table
    ax[1,2]=return_period_plot_1d.return_period_table_ax(ax[1,2], gevd_fitter, block_size, data_units_fm, bootstrap_gevd_fit)
    
    # Some decor
    t=ax[1,2].text(0.03, 0.90, '(f)', transform=ax[1,2].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))    

    fig.tight_layout()

    plt.show()
    
    return fig, ax
    
def plot_copula_diagnostic(copula_x_sample, copula_y_sample, 
                           x_sample_data_scale, y_sample_data_scale,
                           x_gevd_fitter, y_gevd_fitter,
                           x_name, y_name, um_bins=np.linspace(0,1,11)):
    """
    Function to plot diagnostic to assess copula fit.

    Parameters
    ----------
    copula_x_sample : np.array or pd.Series
        Random sample of x from copula on uniform margins.
    copula_y_sample : np.array or pd.Series
        Random sample of y from copula on uniform margins.
    x_sample_data_scale : np.array or pd.Series
        Random sample of x transformed to data scale.
    y_sample_data_scale : np.array or pd.Series
        Random sample of y transformed to data scale.
    x_gevd_fitter : gevd_fitter class
        See gevd_fitter.py.
    y_gevd_fitter : gevd_fitter class
        See gevd_fitter.py.
    x_name : string
        String name for x. Used for labelling plots.
    y_name : string
        String name for y. Used for labelling plots.
    um_bins : np.array
        array defining the edges of the bins for the 
        uniform margins histograms

    Returns
    -------
    fig : matplotlib figure
        Figure containing copula diagnostic plot.
    ax : array of matplotlib axes
        Four axes within fig.

    """
    
    fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(16.5,10.5))
    
    # FOR X PARAMETER
    
    # Plot normalised histogram of copula sample on uniform margins
    ax[0,0].hist(copula_x_sample, bins=um_bins, density=True, rwidth=0.8, color='darkorange', label=x_name+' copula sample\n(uniform margins)')

    # Some decor
    ax[0,0].set_xlabel('Copula sample for '+x_name)
    ax[0,0].set_ylabel('Normalised Occurrence')
    ax[0,0].set_title('Copula sample on uniform margins')
    t=ax[0,0].text(0.06, 0.94, '(a)', transform=ax[0,0].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0,0].legend(loc='upper right') 
    
    # Plot normalised histogram of copula sample in data scale
    ax[0,1].hist(x_sample_data_scale, bins=25, density=True, rwidth=0.8, color='darkgray', label=x_name+' copula\nsample\n(data scale)')
    
    # Overplot distribution
    model_x=np.linspace(np.nanmin(x_sample_data_scale),np.nanmax(x_sample_data_scale), 100)
    model_y = x_gevd_fitter.frozen_dist.pdf(model_x)
    ax[0,1].plot(model_x, model_y, color='darkmagenta', label=x_gevd_fitter.formatted_dist_name)
    
    # Some decor
    ax[0,1].set_xlabel('Data scale for '+x_name)
    ax[0,1].set_ylabel('Normalised Occurrence')
    ax[0,1].set_title('Copula sample vs '+x_gevd_fitter.formatted_dist_name+' (data scale)')
    t=ax[0,1].text(0.06, 0.94, '(b)', transform=ax[0,1].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0,1].legend(loc='upper right')
    
    # QQ plot comparing Copula sample in data scale with GEVD fit
    ax[0,2]=qq_plot.qq_data_vs_model(ax[0,2], x_sample_data_scale, copula_x_sample,
                        x_gevd_fitter, marker='^', fillstyle='none',
                        color='darkmagenta', title='Copula sample vs '+x_gevd_fitter.formatted_dist_name+' (QQ)', 
                        legend_pos='center left')
    ax[0,2].set_xlabel('Copula sample in data scale')
    ax[0,2].set_ylabel(x_gevd_fitter.formatted_dist_name+' fitted to observations')
    t=ax[0,2].text(0.06, 0.94, '(c)', transform=ax[0,2].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    
    # FOR Y PARAMETER
    # Plot normalised histogram of copula sample on uniform margins
    ax[1,0].hist(copula_y_sample, bins=um_bins, density=True, rwidth=0.8, color='darkorange', label=y_name+' copula sample\n(uniform margins)')

    # Some decor
    ax[1,0].set_xlabel('Copula sample for '+y_name)
    ax[1,0].set_ylabel('Normalised Occurrence')
    ax[1,0].set_title('Copula sample on uniform margins')
    t=ax[1,0].text(0.06, 0.94, '(d)', transform=ax[1,0].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[1,0].legend(loc='upper right')

    # Plot normalised histogram of copula sample in data scale
    ax[1,1].hist(y_sample_data_scale, bins=25, density=True, rwidth=0.8, color='darkgray', label=y_name+' copula\nsample\n(data scale)')
    
    # Overplot distribution
    model_x=np.linspace(np.nanmin(y_sample_data_scale),np.nanmax(y_sample_data_scale), 100)
    model_y = y_gevd_fitter.frozen_dist.pdf(model_x)
    ax[1,1].plot(model_x,model_y, color='darkmagenta', label=y_gevd_fitter.formatted_dist_name)
    
    # Some decor
    ax[1,1].set_xlabel('Data scale for '+y_name)
    ax[1,1].set_ylabel('Normalised Occurrence')
    ax[1,1].set_title('Copula sample vs '+y_gevd_fitter.formatted_dist_name+' (data scale)')
    t=ax[1,1].text(0.06, 0.94, '(e)', transform=ax[1,1].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[1,1].legend(loc='upper right')    
    
    # QQ plot comparing Copula sample in data scale with GEVD fit
    ax[1,2]=qq_plot.qq_data_vs_model(ax[1,2], y_sample_data_scale, copula_y_sample,
                        y_gevd_fitter, marker='^', fillstyle='none',
                        color='darkmagenta', title='Copula sample vs '+y_gevd_fitter.formatted_dist_name+' (QQ)', 
                        legend_pos='center left')
    ax[1,2].set_xlabel('Copula sample in data scale')
    t=ax[1,2].text(0.06, 0.94, '(f)', transform=ax[1,2].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    
    fig.tight_layout()
    
    plt.show()
    
    return fig, ax