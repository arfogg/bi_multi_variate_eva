# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:40:20 2023

@author: A R Fogg

based on R code sent by Daire Healy (Maynooth)
"""

import scipy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.gridspec import GridSpec

import transform_uniform_margins

    
def plot_extremal_dependence_coefficient(x_data,y_data, x_name, y_name, x_units, y_units, csize=17):
    """
    Function to create a diagnostic plot to determine whether a pair of 
    variables are asymptotically dependent.

    Parameters
    ----------
    x_data : np.array or pd.Series
        Timeseries of x parameter.
    y_data : np.array or pd.Series
        Timeseries of y parameter.
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
    # # TEST DATA
    # # Making the same function as in Daire's code
    # mean = [0, 0]
    # cov = [[1, 0.95], [0.95, 1]]
    # x_data, y_data = np.random.multivariate_normal(mean, cov, 10000).T
    
    # Makes fig.tight_layout() and colorbar work together
    mpl.rcParams['figure.constrained_layout.use'] = False
    
    
    # Define plotting window for generic test plotting
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
    
    # Transform the variables into "uniform" - ask Daire
    #x_unif=scipy.stats.rankdata(x_data)/(x_data.size+1)
    x_unif=transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(x_data,plot=False)
    #y_unif=scipy.stats.rankdata(y_data)/(y_data.size+1)
    y_unif=transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(y_data,plot=False)
    # PLOT X_UNIF AS FUNCTION OF X_DATA TO GET VISUALISATION OF EMPIRICAL CDF
    
    
    # Plot out these uniform data
    
    # Notes from meeting
    #   put it onto uniform margins
    #   put everything onto same marginal distributions - e.g. if one variable
    #   is gaussian and one not - comparing them will get biased information and results are over/underestimating dependence
    #   so put the data into it's CDF and then you get out a uniform distribution
    #   ^ so do this seperately to y and x -> so you get "data transformed through prob integral transform onto uniform margins"
    
    h_unif=ax_data_unif.hist2d(x_unif, y_unif, bins=50, density=True)
    cb_unif=fig.colorbar(h_unif[3],ax=ax_data_unif)
    # Formatting
    ax_data_unif.set_xlabel(x_name+" on uniform margins", fontsize=csize)
    ax_data_unif.set_ylabel(y_name+" on uniform margins", fontsize=csize)
    for label in (ax_data_unif.get_xticklabels() + ax_data_unif.get_yticklabels()):
        label.set_fontsize(csize)
    cb_unif.ax.tick_params(labelsize=csize)
    cb_unif.set_label("Normalised occurrence", fontsize=csize)
    
    # Calculate the "extremal dependence coefficient", called chi
    #    for a range of quantiles u (from 0 to 1)
    # if chi -> 0 as u -> 1, then x and y are asymptotically independent
    #   otherwise they are asymtotically dependent
    u=np.linspace(0,0.99,100)  # bins of width 0.01
    chi=[]
    for i in range(u.size):
        top,=np.where((x_unif>u[i]) & (y_unif>u[i]))
        bottom,=np.where(x_unif>u[i])
        chi.append( (top.size)/(bottom.size) )

    ax_edc.plot(u,chi, color='orange')
    # Formatting
    ax_edc.set_xlabel("Quantiles", fontsize=csize)
    ax_edc.set_ylabel("Extremal Dependence Coefficient, $\chi$", fontsize=csize)
    for label in (ax_edc.get_xticklabels() + ax_edc.get_yticklabels()):
        label.set_fontsize(csize)
    
    #t=ax_edc.text(0.95, 0.95, '$\chi _{min}$ = '+str(round(np.min(chi),3)), transform=ax_edc.transAxes, fontsize=csize,  va='top', ha='right')
    t=ax_edc.text(0.95, 0.95, '$\chi _{q=1}$ = '+str(round(chi[-1],3)), transform=ax_edc.transAxes, fontsize=csize,  va='top', ha='right')
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))

    
    fig.tight_layout()
    
    return fig, ax_data, ax_data_unif, ax_edc, chi[-1]
        
