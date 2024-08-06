# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:35:16 2024

@author: A R Fogg

Based on the private code found on github
https://github.com/arfogg/qq_plot
"""

import numpy as np

from scipy.stats import linregress

def qq_data_vs_data(x_dist, y_dist, ax, quantiles=np.linspace(0,100,101),
            marker='^', fillstyle='none', color='darkcyan', title='', 
            fit_linear=False, legend_pos='upper left'):
    """
    Takes in two datasets and plots a Quantile-Quantile
    plot comparing the two distributions.

    Parameters
    ----------
    x_dist : np.array or list 
        Set of observations forming x distribution. Will
        be on the x axis.
    y_dist : TYPE
        Set of observations forming y distribution. Will
        be on the y axis.
    ax : matplotlib axis object
        Axis object to do the plotting on.
    quantiles : np.array, optional
        Quantiles to evaluate the distributions at for 
        the QQ plot. The default is np.linspace(0,100,101).
    marker : string, optional
        Markerstyle for the points, feeds into ax.plot. For 
        valid options see matplotlib.markers. The default is '^'.
    fillstyle : string, optional
        Fillstyle for the markers. The default is 'none'.
    color : string, optional
        Color for the markers. The default is 'darkcyan'.
    title : string, optional
        Title for the axis. The default is ''.
    fit_linear : bool, optional
        If fit_linear==True, a linear fit to the quantiles 
        is calculated and overplotted. The default is False.
    legend_pos : string, optional
        Position to place the legend. The default is 'upper 
        left'.

    Returns
    -------
    ax
        Matplotlib axis object containing the plotting.

    """
    
    print('Welcome to the Quantile-Quantile plotting program')
    

    # Calculate the quantiles
    x_q=np.nanpercentile(x_dist, quantiles)
    y_q=np.nanpercentile(y_dist, quantiles)

    # Plot the quantiles
    ax.plot(x_q,y_q, linewidth=0.0,
               marker=marker, fillstyle=fillstyle, color=color, label='QQ')

    
    # Draw a y=x line
    min_value = min([min(ax.get_xlim()), min(ax.get_ylim())])
    max_value = max([max(ax.get_xlim()), max(ax.get_ylim())])
    
    ax.plot( [min_value, max_value], [min_value, max_value], linewidth=1.0, linestyle='--', color='black', label='y=x')
    
    ax.set_title(title)
    
    # Fit a linear trend to the data
    if fit_linear:
        lin_fit=linregress(x_q,y_q)
        ax.plot(x_q, lin_fit.intercept + lin_fit.slope*x_q, color=color, linewidth=1.0, 
                label=str(float('%.4g' % lin_fit.slope))+'x + '+  str(float('%.4g' % lin_fit.intercept)))
        ax.legend(loc=legend_pos)
        print('Returning QQ axis')
        return ax
    else:
        ax.legend(loc=legend_pos)
        print('Returning Q-Q axis')
        return ax
    
    
def qq_data_vs_model(ax, extrema_ds, extrema_um_empirical, gevd_fitter, 
                     marker='^', fillstyle='none', color='darkcyan', title='', 
                     legend_pos='center left'):
    """
    Function to generate a QQ plot comparing a GEVD model with
    detected extrema.

    Parameters
    ----------
    ax : matplotlib axis object
        Axis to draw the QQ plot onto.
    extrema_ds : np.array
        Detected extrema in data scale.
    extrema_um_empirical : np.array
        Detected extrema on uniform margins
        (transformed empirically).
    gevd_fitter : gevd_fitter class
        Object containing fitting information.
    marker : string, optional
        Markerstyle for the points, feeds into ax.plot. For 
        valid options see matplotlib.markers. The default is '^'.
    fillstyle : string, optional
        Fillstyle for the markers. The default is 'none'.
    color : string, optional
        Color for the markers. The default is 'darkcyan'.
    title : string, optional
        Title for the axis. The default is ''.
    legend_pos : string, optional
        Position to place the legend. The default is 'center 
        left'.        

    Returns
    -------
    ax : matplotlib axis object
        Axis containing the QQ plot.

    """
    
    print('Welcome to the Quantile-Quantile plotting program')
    
    model_qq = gevd_fitter.frozen_dist.ppf(extrema_um_empirical)

    ax.plot(extrema_ds, model_qq, linewidth=0.0,
               marker=marker, fillstyle=fillstyle, color=color, label='QQ')
    
    # Draw a y=x line
    min_value = min([min(ax.get_xlim()), min(ax.get_ylim())])
    max_value = max([max(ax.get_xlim()), max(ax.get_ylim())])
    
    ax.plot( [min_value, max_value], [min_value, max_value], linewidth=1.0, linestyle='--', color='black', label='y=x')
    
    # Decor
    ax.set_title(title)
    ax.legend(loc=legend_pos)
    
    return ax