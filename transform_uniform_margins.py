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
            #data[i]=( (fit_params.scale) / ( fit_params.shape_ * (-np.log(1-data_unif[i])) ** fit_params.shape_ ) )-(fit_params.scale/fit_params.shape_)+(fit_params.location)
            data[i]=( (fit_params.scale) / ( fit_params.shape_ * (-np.log(data_unif[i])) ** fit_params.shape_ ) )-(fit_params.scale/fit_params.shape_)+(fit_params.location)
    elif fit_params.distribution_name[0]=="gumbel_r":
        # For the Gumbel distribution
        print('Transforming data from uniform margins to data scale for Gumbel distribution')
        for i in range(data.size):
            #!!! need to get this eqn/derivation checked !!!
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
    # PDF of distributions from wikipedia
    if fit_params.distribution_name[0]=='genextreme':
        print('Estimating PDF for GEVD distribution')
        pdf=genextreme.pdf(x_data, fit_params.shape_, loc=fit_params.location, scale=fit_params.scale)
    elif fit_params.distribution_name[0]=='gumbel_r':
        print('Estimating PDF for Gumbel distribution')
        pdf=gumbel_r.pdf(x_data, loc=fit_params.location, scale=fit_params.scale)
        
    return pdf

def plot_diagnostic(data,data_unif_empirical,data_unif_cdf,fit_params,data_tag):
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

    Returns
    -------
    None.

    """
    # Initialise figure and axes
    fig,ax=plt.subplots(ncols=2,figsize=(8,4))
        
    # Plot normalised histogram of extremes
    ax[0].hist(data, bins=25, density=True, rwidth=0.8, color='deepskyblue', label='extremes')
        
    # Initialise arrays
    model_x=np.linspace(np.nanmin(data),np.nanmax(data), 100)
    model_y=estimate_pdf(model_x,fit_params)
    
    # Plot the PDF against x
    ax[0].plot(model_x,model_y, color='darkmagenta', label=fit_params.distribution_name[0])
    
    # Some decor
    ax[0].set_ylabel('Normalised Occurrence')
    ax[0].set_xlabel('Data in data scale')    
    ax[0].legend(loc='upper right')
    t=ax[0].text(0.06, 0.94, '(a)', transform=ax[0].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0].set_title(fit_params.distribution_name[0]+' fit assessment for '+data_tag)

    # Plot normalised histograms of different uniform margins data
    ax[1].hist(data_unif_cdf, bins=25, density=True, rwidth=0.8, color='darkorange', label='using CDF')
    ax[1].hist(data_unif_empirical, bins=25, density=True, rwidth=0.8, color='grey', alpha=0.5, label='empirical')
    
    # Some decor
    ax[1].set_ylabel('Normalised Occurrence')
    ax[1].set_xlabel('Data on uniform margins '+data_tag)
    ax[1].legend(loc='upper right')
    t=ax[1].text(0.06, 0.94, '(b)', transform=ax[1].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[1].set_title('Comparison of data on uniform margins')
    
    fig.tight_layout()

    plt.show()
    
    return fig, ax
    
def plot_copula_diagnostic(copula_x_sample, copula_y_sample, x_sample_data_scale, y_sample_data_scale, x_fit_params, y_fit_params, x_name, y_name):
    
    fig, ax=plt.subplots(nrows=2,ncols=2, figsize=(7,7))
    
    # FOR X PARAMETER
    # Plot normalised histogram of copula sample in data scale
    ax[0,0].hist(x_sample_data_scale, bins=25, density=True, rwidth=0.8, color='deepskyblue', label='x copula sample\n(data scale)')
    
    # Overplot distribution
    model_x=np.linspace(np.nanmin(x_sample_data_scale),np.nanmax(x_sample_data_scale), 100)
    model_y=estimate_pdf(model_x,x_fit_params)
    ax[0,0].plot(model_x,model_y, color='darkmagenta', label=x_fit_params.distribution_name[0])
    
    # Some decor
    ax[0,0].set_xlabel('Data scale for '+x_name)
    ax[0,0].set_ylabel('Normalised Occurrence')
    ax[0,0].set_title('Copula sample vs '+x_fit_params.distribution_name[0]+' (data scale)')
    t=ax[0,0].text(0.06, 0.94, '(a)', transform=ax[0,0].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0,0].legend(loc='upper right')
    
    # Plot normalised histogram of copula sample on uniform margins
    ax[0,1].hist(copula_x_sample, bins=25, density=True, rwidth=0.8, color='darkorange', label='copula sample')

    # Some decor
    ax[0,1].set_xlabel('Copula sample for '+x_name)
    ax[0,1].set_ylabel('Normalised Occurrence')
    ax[0,1].set_title('Copula sample on uniform margins')
    t=ax[0,1].text(0.06, 0.94, '(b)', transform=ax[0,1].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0,1].legend(loc='upper right')    

    # FOR Y PARAMETER
    # Plot normalised histogram of copula sample in data scale
    ax[1,0].hist(y_sample_data_scale, bins=25, density=True, rwidth=0.8, color='deepskyblue', label='y copula sample\n(data scale)')
    
    # Overplot distribution
    model_x=np.linspace(np.nanmin(y_sample_data_scale),np.nanmax(y_sample_data_scale), 100)
    model_y=estimate_pdf(model_x,y_fit_params)
    ax[1,0].plot(model_x,model_y, color='darkmagenta', label=y_fit_params.distribution_name[0])
    
    # Some decor
    ax[1,0].set_xlabel('Data scale for '+y_name)
    ax[1,0].set_ylabel('Normalised Occurrence')
    ax[1,0].set_title('Copula sample vs '+y_fit_params.distribution_name[0]+' (data scale)')
    t=ax[1,0].text(0.06, 0.94, '(c)', transform=ax[1,0].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[1,0].legend(loc='upper right')    
    
    # Plot normalised histogram of copula sample on uniform margins
    ax[1,1].hist(copula_y_sample, bins=25, density=True, rwidth=0.8, color='darkorange', label='copula sample')

    # Some decor
    ax[1,1].set_xlabel('Copula sample for '+y_name)
    ax[1,1].set_ylabel('Normalised Occurrence')
    ax[1,1].set_title('Copula sample on uniform margins')
    t=ax[1,1].text(0.06, 0.94, '(d)', transform=ax[1,1].transAxes, va='top', ha='left')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[1,1].legend(loc='upper right')
    
    fig.tight_layout()
    
    plt.show()
    
    return fig, ax