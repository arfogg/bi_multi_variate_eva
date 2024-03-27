# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:40:20 2023

@author: A R Fogg

based on R code sent by Dr Daire Healy
"""

import scipy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.gridspec import GridSpec

import transform_uniform_margins

    
def plot_extremal_dependence_coefficient(x_data, y_data, x_bs_um, y_bs_um, bootstrap_n_iterations,
                                         x_name, y_name, x_units, y_units, csize=17):
    """
    Function to create a diagnostic plot to determine whether a pair of 
    variables are asymptotically dependent.

    Parameters
    ----------
    x_data : np.array or pd.Series
        Timeseries of x parameter.
    y_data : np.array or pd.Series
        Timeseries of y parameter.
    x_bs_um : np.array
        Bootstrapped dataset of x_data, as in output from 
        bootstrap_data.iterative_bootstrap_um. Of shape x_data.size x 
        bootstrap_n_iterations.
    y_bs_um : np.array
        Bootstrapped dataset of y_data, as in output from 
        bootstrap_data.iterative_bootstrap_um. Of shape y_data.size x 
        bootstrap_n_iterations.
    bootstrap_n_iterations : int
        Number of iterations used to generate the bootstrapped
        dataset. So, x_bs_um will be of shape x_data.size x 
        bootstrap_n_iterations.
    x_name : string
        String name for axes labelling for x.
    y_name : string
        String name for axes labelling for y.
    x_units : string
        String name for axes labelling for x units.
    y_units : string
        String name for axes labelling for y units.
    csize : int, optional
        Fontsize for text on output plot. The default is 17.

    Returns
    -------
    fig : matplotlib figure
        Figure containing ax_data, ax_data_unif, ax_edc.    
    ax_data : matplotlib axes
        Axes containing a 2D histogram comparing x and y in data scale.
    ax_data_unif : matplotlib axes
        Axes containing a 2D histogram comparing x and y on uniform margins.
    ax_edc : matplotlib axes
        Axes showing the extremal dependence coefficient as a function of quantile.
    np.min(chi) : float
        Minimum value of the extremal dependence coefficient.

    """

    # Makes fig.tight_layout() and colorbar work together
    mpl.rcParams['figure.constrained_layout.use'] = False
    
    # Define plotting window for diagnostic plot
    fig=plt.figure(figsize=(15,11))
    gs=GridSpec(2,2,figure=fig)

    # Define three axes to plot the data on
    ax_data=fig.add_subplot(gs[0,0])
    ax_data_unif=fig.add_subplot(gs[0,1])
    ax_edc=fig.add_subplot(gs[1,:])
    
    # Plot the original input data
    h_data=ax_data.hist2d(x_data,y_data, bins=50, density=True, norm='log')
    cb_data=fig.colorbar(h_data[3],ax=ax_data)
    # Formatting
    ax_data.set_xlabel(x_name+' '+x_units, fontsize=csize)
    ax_data.set_ylabel(y_name+' '+y_units, fontsize=csize)
    for label in (ax_data.get_xticklabels() + ax_data.get_yticklabels()):
        label.set_fontsize(csize)
    cb_data.ax.tick_params(labelsize=csize)
    cb_data.set_label("Normalised occurrence", fontsize=csize)
    t_data=ax_data.text(0.06, 0.94, '(a)', transform=ax_data.transAxes, fontsize=csize,  va='top', ha='left')
    t_data.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))

    # Transform the variables into uniform margins
    x_unif=transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(x_data,plot=False)
    y_unif=transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(y_data,plot=False)
    
    # Plot uniform margins data
    h_unif=ax_data_unif.hist2d(x_unif, y_unif, bins=50, density=True)
    cb_unif=fig.colorbar(h_unif[3],ax=ax_data_unif)
    # Formatting
    ax_data_unif.set_xlabel(x_name+" on uniform margins", fontsize=csize)
    ax_data_unif.set_ylabel(y_name+" on uniform margins", fontsize=csize)
    for label in (ax_data_unif.get_xticklabels() + ax_data_unif.get_yticklabels()):
        label.set_fontsize(csize)
    cb_unif.ax.tick_params(labelsize=csize)
    cb_unif.set_label("Normalised occurrence", fontsize=csize)
    
    t_unif=ax_data_unif.text(0.06, 0.94, '(b)', transform=ax_data_unif.transAxes, fontsize=csize,  va='top', ha='left')
    t_unif.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    
    # Calculate the extremal dependence coefficient (chi) over quantiles u 
    quantiles=np.linspace(0,0.99,100)
    chi=calculate_extremal_dependence_coefficient(quantiles,x_unif,y_unif)
    
    # Calculate errors on chi
    print('---------------------TEMP REDUCTION TO N BOOTSTRAP FOR TESTING-------------------------------')
    bootstrap_n_iterations=10
    chi_lower_bound, chi_upper_bound, bootstrap_chi = calculate_upper_lower_quartile_chi(quantiles, x_bs_um, y_bs_um, bootstrap_n_iterations)

    markers=[".",",","o","v","^","<",">","1","2","3"]
    for i in range(bootstrap_n_iterations):
        ax_edc.plot(quantiles,bootstrap_chi[:,i], label=i, marker=markers[i], linewidth=0.0)
    ax_edc.legend()

    # Plot chi as a function of quantiles
    ax_edc.plot(quantiles,chi, color='orange')
    # Plot error shade
    ax_edc.fill_between(quantiles, chi_lower_bound, chi_upper_bound, alpha=0.5, color='grey')
    # Formatting
    ax_edc.set_xlabel("Quantiles", fontsize=csize)
    ax_edc.set_ylabel("Extremal Dependence Coefficient, $\chi$", fontsize=csize)
    for label in (ax_edc.get_xticklabels() + ax_edc.get_yticklabels()):
        label.set_fontsize(csize)
    t=ax_edc.text(0.95, 0.95, '$\chi _{q=1}$ = '+str(round(chi[-1],3)), transform=ax_edc.transAxes, fontsize=csize,  va='top', ha='right')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    t_edc=ax_edc.text(0.03, 0.94, '(c)', transform=ax_edc.transAxes, fontsize=csize,  va='top', ha='left')
    t_edc.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))    

    fig.tight_layout()
    
    return fig, ax_data, ax_data_unif, ax_edc, chi[-1], chi
        
def calculate_extremal_dependence_coefficient(quantiles, x_unif, y_unif):
    """
    Calculate the extremal dependence coefficient for 
    two parameters x and y provided on uniform margins.

    Parameters
    ----------
    quantiles : np.array
        Quantiles to calculate chi over.
    x_unif : np.array
        X data on uniform margins.
    y_unif : np.array
        Y data on uniform margins.

    Returns
    -------
    chi : np.array
        Extremal dependence coefficient as a function
        of input quantiles.

    """

    chi=[]
    for i in range(quantiles.size):
        top,=np.where((x_unif>quantiles[i]) & (y_unif>quantiles[i]))
        bottom,=np.where(x_unif>quantiles[i])
        chi.append( (top.size)/(bottom.size) )
        
    return chi

def calculate_upper_lower_quartile_chi(quantiles, x_bs_um, y_bs_um, bootstrap_n_iterations):
    
    # based on bootstrapping
    
    print('Estimating chi over '+str(bootstrap_n_iterations)+' bootstrapped iterations - may be slow')
    bts_chi=np.full((quantiles.size,bootstrap_n_iterations), np.nan)
    for i in range(bootstrap_n_iterations):
        print(i)
        bts_chi[:,i]=calculate_extremal_dependence_coefficient(quantiles, x_bs_um[:,i], y_bs_um[:,i])
    
    lower_q=np.full(quantiles.size, np.nan)
    upper_q=np.full(quantiles.size, np.nan)
    for j in range(quantiles.size):
        lower_q[j], upper_q[j]=np.percentile(bts_chi[j,:],[2.5,97.5])
    
    return lower_q, upper_q, bts_chi
        
        
        
        
        
        
        
        
        
        
        
        