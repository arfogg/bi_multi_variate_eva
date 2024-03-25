# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:44:18 2023

@author: A R Fogg
"""

import scipy

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import genextreme
from scipy.stats import gumbel_r

import qq_plot
import return_period_plot_1d

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

def transform_from_data_scale_to_uniform_margins_using_CDF(data, fit_params, distribution='genextreme', plot=False):
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
    fit_params : df
        For distribution='genextreme', must contain parameters scale, shape_, location.
    distribution : TYPE, optional
        DESCRIPTION. The default is 'genextreme'.
    plot : BOOL, optional
        If plot == True, plots the distributions of data in data
        scale and on uniform margins. The default is False.

    Returns
    -------
    data_unif : np.array
        Data transformed to uniform margins

    """
    
    if distribution=='genextreme':
        print('Transforming from data scale to uniform margins for GEVD distribution')
        # data_unif=np.full(data.size,np.nan)
        # for i in range(data.size):
            # 1- G(x) = u
            # u = data on uniform margins
            
            # G(x) IS the CDF, so we don't need to do 1- at the beginning
                        
            # # Citation for this equation Coles (2001) page 47
            # data_unif[i]=np.exp(-1.0 * (1.0 + (fit_params.shape_ * ( (data[i]-fit_params.location)/(fit_params.scale) )))**(-1.0/fit_params.shape_)  )
        data_unif=genextreme.cdf(data, fit_params.shape_, loc=fit_params.location, scale=fit_params.scale)    
            
    
    elif distribution=='gumbel_r':
        print('Transforming from data scale to uniform margins for the Gumbel distribution')
        # data_unif=np.full(data.size,np.nan)
        # for i in range(data.size):
        #     # 1- G(x) = u
        #     # u = data on uniform margins
        #     # G(x) IS the CDF, so we don't need to do 1- at the beginning
        #     #data_unif[i]=1.0 - np.exp( -1.0*np.exp( -1.0*( (data[i]-fit_params.location)/(fit_params.scale) ) ) )
            
        #     # Citation for this equation, Coles (2001) page 48
        #     data_unif[i]=np.exp( -1.0*np.exp( -1.0*( (data[i]-fit_params.location)/(fit_params.scale) ) ) )
        data_unif=gumbel_r.cdf(data,loc=fit_params.location, scale=fit_params.scale)
            
    else:
        print('ERROR: distribution "'+distribution+'" not implemented yet or incorrect spelling')
        raise NameError(distribution+' distribution not implemented yet or incorrect spelling')
    
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
    

def transform_from_uniform_margins_to_data_scale(data_unif,fit_params, plot=False):
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
    fit_params : pd.DataFrame
        df containing tags including distribution_name,
        shape_, scale, location
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    
    data=np.full(data_unif.size,np.nan)
    
    if fit_params.distribution_name[0]=='genextreme':
        # For the GEVD distribution
        print('Transforming data from uniform margins to data scale for GEVD distribution')    
        
        for i in range(data.size):
            data[i]=( (fit_params.scale) / ( fit_params.shape_ * (-np.log(data_unif[i])) ** fit_params.shape_ ) )-(fit_params.scale/fit_params.shape_)+(fit_params.location)
    elif fit_params.distribution_name[0]=="gumbel_r":
        # For the Gumbel distribution
        print('Transforming data from uniform margins to data scale for Gumbel distribution')
        for i in range(data.size):
            data[i]=fit_params.location - (fit_params.scale * np.log(-1.0*np.log(data_unif[i])))
        
    
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

def estimate_pdf(x_data,fit_params):
    """
    Function to estimate the values of the PDF for GEVD
    and Gumbel distributions

    Parameters
    ----------
    x_data : np.array
        X values at which PDF will be calculated
    fit_params : pd.DataFrame
        df containing tags including distribution_name,
        shape_, scale, location

    Returns
    -------
    pdf : np.array
        Value of PDF along x_data

    """

    # Calculate the PDF at values of x
    if fit_params.distribution_name[0]=='genextreme':
        print('Estimating PDF for GEVD distribution')
        pdf=genextreme.pdf(x_data, fit_params.shape_, loc=fit_params.location, scale=fit_params.scale)
    elif fit_params.distribution_name[0]=='gumbel_r':
        print('Estimating PDF for Gumbel distribution')
        pdf=gumbel_r.pdf(x_data, loc=fit_params.location, scale=fit_params.scale)
        
    return pdf

def plot_diagnostic(data,data_unif_empirical,data_unif_cdf,fit_params,data_tag, data_units_fm, block_size, um_bins=np.linspace(0,1,11)):
    """
    Function to plot the PDF of extremes and the fitted distribution (left),
    and comparing the empirically and CDF determined data on uniform
    margins (right).

    Parameters
    ----------
    data : np.array
        Detected extremes in data scale.
    data_unif_empirical : np.array
        Detected extremes converted to uniform margins
        empirically.
    data_unif_cdf : np.array
        Detected extremes converted to uniform margins
        using the CDF.
    fit_params : pandas.DataFrame
        df containing tags including distribution_name,
        shape_, scale, location
    data_tag : string
        name of data to be put in figure captions etc
    data_units_fm : string
        units for data to be put in axes labels etc
    block_size : pd.Timedelta
        Size over which block maxima have been found, e.g. pd.to_timedelta("365.2425D").
    um_bins : np.array
        array defining the edges of the bins for the 
        uniform margins histograms

    Returns
    -------
    None.

    """
    # Initialise figure and axes
    fig,ax=plt.subplots(ncols=3, nrows=2, figsize=(13,8))
        
    # Plot normalised histogram of extremes
    ax[0,0].hist(data, bins=np.linspace(np.nanmin(data),np.nanmax(data),25), density=True, rwidth=0.8, color='darkgrey', label='extremes')
        
    # Initialise arrays
    model_x=np.linspace(np.nanmin(data),np.nanmax(data), 100)
    model_y=estimate_pdf(model_x,fit_params)
    
    # Plot the PDF against x
    ax[0,0].plot(model_x,model_y, color='darkmagenta', label=fit_params.formatted_dist_name[0])
    
    # Some decor
    ax[0,0].set_ylabel('Normalised Occurrence')
    ax[0,0].set_xlabel(data_tag + ' in data scale ('+data_units_fm+')')    
    t=ax[0,0].text(0.06, 0.94, '(a)', transform=ax[0,0].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0,0].set_title(fit_params.formatted_dist_name[0]+' fit assessment for '+data_tag)

    # Plot normalised histograms of different uniform margins data
    ax[0,1].hist(data_unif_cdf, bins=um_bins, density=True, rwidth=0.8, color='darkorange', label='using CDF')
    ax[0,1].hist(data_unif_empirical, bins=um_bins, density=True, rwidth=0.8, color='grey', alpha=0.5, label='empirical')
    
    # Some decor
    ax[0,1].set_ylabel('Normalised Occurrence')
    ax[0,1].set_xlabel(data_tag + ' on uniform margins')
    ax[0,1].legend(loc='upper right')
    t=ax[0,1].text(0.06, 0.94, '(b)', transform=ax[0,1].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0,1].set_title('Comparison of data on uniform margins')
    ax[0,0].legend(loc='upper right')
    
    # QQ plot comparing the extremes and their PDF
    ax[1,0]=qq_plot.qq_data_vs_model(ax[1,0], data, data_unif_empirical, fit_params, 
                         marker='^', fillstyle='none', color='darkmagenta', title='', 
                         legend_pos='center left')
    
    # Some decor
    ax[1,0].set_xlabel('Extremes')
    ax[1,0].set_ylabel('Fitted '+fit_params.formatted_dist_name[0]+' distribution')
    t=ax[1,0].text(0.06, 0.94, '(d)', transform=ax[1,0].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))    
    
    # QQ plot comparing the uniform margins distributions
    ax[1,1]=qq_plot.qq_data_vs_data(data_unif_empirical, data_unif_cdf, ax[1,1], quantiles=np.linspace(0,100,26), 
                            legend_pos='center left', color='darkorange')
    
    # Some decor
    ax[1,1].set_xlabel('Empirical')
    ax[1,1].set_ylabel('CDF')
    t=ax[1,1].text(0.06, 0.94, '(e)', transform=ax[1,1].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    
    
    
    # Return period plot
    ax[0,2]=return_period_plot_1d.return_period_plot(data, fit_params, block_size, data_tag, data_units_fm, ax[0,2], csize=15)

    # TEMPORARY LABEL - REMOVE WHEN COMPLETE
    ax[0,2].text(0.5,0.5,'needs CI shade', transform=ax[0,2].transAxes, va='center', ha='center', fontsize=20)
    
    # Some decor
    t=ax[0,2].text(0.06, 0.94, '(c)', transform=ax[0,2].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))

    
    # Return values table
    ax[1,2].text(0.5,0.5,'needs CI', transform=ax[1,2].transAxes, va='center', ha='center', fontsize=20)
    ax[1,2]=return_period_plot_1d.return_period_table_ax(ax[1,2], fit_params, block_size, data_units_fm)
    
    # Some decor
    t=ax[1,2].text(0.03, 0.90, '(f)', transform=ax[1,2].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))    
    
    
    
    fig.tight_layout()

    plt.show()
    
    return fig, ax
    
def plot_copula_diagnostic(copula_x_sample, copula_y_sample, x_sample_data_scale, y_sample_data_scale,
                           x_fit_params, y_fit_params, x_name, y_name, um_bins=np.linspace(0,1,11)):
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
    x_fit_params : pandas.DataFrame
        pandas.DataFrame of the format output by 
        fit_model_to_extremes.fit_gevd_or_gumbel for x.
    y_fit_params : pandas.DataFrame
        pandas.DataFrame of the format output by 
        fit_model_to_extremes.fit_gevd_or_gumbel for y.
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
    
    
    fig, ax=plt.subplots(nrows=2,ncols=3, figsize=(11,7))
    
    # FOR X PARAMETER
    
    # Plot normalised histogram of copula sample on uniform margins
    ax[0,0].hist(copula_x_sample, bins=um_bins, density=True, rwidth=0.8, color='darkorange', label=x_name+' copula sample\n(uniform margins)')

    # Some decor
    ax[0,0].set_xlabel('Copula sample for '+x_name)
    ax[0,0].set_ylabel('Normalised Occurrence')
    ax[0,0].set_title('Copula sample on uniform margins')
    t=ax[0,0].text(0.06, 0.94, '(b)', transform=ax[0,0].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0,0].legend(loc='upper right') 
    
    # Plot normalised histogram of copula sample in data scale
    ax[0,1].hist(x_sample_data_scale, bins=25, density=True, rwidth=0.8, color='deepskyblue', label=x_name+' copula\nsample\n(data scale)')
    
    # Overplot distribution
    model_x=np.linspace(np.nanmin(x_sample_data_scale),np.nanmax(x_sample_data_scale), 100)
    model_y=estimate_pdf(model_x,x_fit_params)
    ax[0,1].plot(model_x,model_y, color='darkmagenta', label=x_fit_params.formatted_dist_name[0])
    
    # Some decor
    ax[0,1].set_xlabel('Data scale for '+x_name)
    ax[0,1].set_ylabel('Normalised Occurrence')
    ax[0,1].set_title('Copula sample vs '+x_fit_params.formatted_dist_name[0]+' (data scale)')
    t=ax[0,1].text(0.06, 0.94, '(a)', transform=ax[0,1].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0,1].legend(loc='upper right')
    
    # QQ plot comparing Copula sample in data scale with GEVD fit
    ax[0,2].text(0.5,0.5,'QQ TBC', transform=ax[0,2].transAxes, va='center', ha='center')

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
    ax[1,1].hist(y_sample_data_scale, bins=25, density=True, rwidth=0.8, color='deepskyblue', label=y_name+' copula\nsample\n(data scale)')
    
    # Overplot distribution
    model_x=np.linspace(np.nanmin(y_sample_data_scale),np.nanmax(y_sample_data_scale), 100)
    model_y=estimate_pdf(model_x,y_fit_params)
    ax[1,1].plot(model_x,model_y, color='darkmagenta', label=y_fit_params.formatted_dist_name[0])
    
    # Some decor
    ax[1,1].set_xlabel('Data scale for '+y_name)
    ax[1,1].set_ylabel('Normalised Occurrence')
    ax[1,1].set_title('Copula sample vs '+y_fit_params.formatted_dist_name[0]+' (data scale)')
    t=ax[1,1].text(0.06, 0.94, '(c)', transform=ax[1,1].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[1,1].legend(loc='upper right')    
    
    # QQ plot comparing Copula sample in data scale with GEVD fit
    ax[1,2].text(0.5,0.5,'QQ TBC', transform=ax[1,2].transAxes, va='center', ha='center')

    
    fig.tight_layout()
    
    plt.show()
    
    return fig, ax