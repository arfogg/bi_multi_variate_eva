# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:40:02 2023

@author: A R Fogg

Functions to fit copulae to data.
"""

import numpy as np
import matplotlib.pyplot as plt

from copulas.bivariate import Bivariate


def fit_copula_bivariate(x_extremes, y_extremes, x_name, y_name):
    """
    Function to fit a copula to x and y extremes

    Parameters
    ----------
    x_extremes : np.array
        X extremes on uniform margins.
    y_extremes : np.array
        Y extremes on uniform margins.
    x_name : string
        String descriptor for x values e.g. 'AE'.
    y_name : string
        String descriptor for y values e.g. 'AL'.

    Returns
    -------
    copula : copulas copula
        fitted copula

    """

    # First, check same number of extremes in x and y
    if np.array(x_extremes).size == np.array(y_extremes).size:
        print('Fitting copula to parsed extremes')
    else:
        print('ERROR: fit_copula_to_extremes.fit_copula_bivariate')
        print('x_extremes and y_extremes must have the same length')
        raise NameError('x_extremes and y_extremes must have the same length')

    # Format the extremes to how copulas wants them
    copula_arr = np.array([x_extremes, y_extremes]).T

    # Initialise the copula - using Gumbel
    #   (options are clayton, frank, gumbel or independence)
    copula = Bivariate(copula_type='gumbel')

    # Fit the copula to the extremes
    copula.fit(copula_arr)

    return copula


def qualitative_copula_fit_check_bivariate(x_extremes, y_extremes,
                                           x_sample, y_sample,
                                           x_name, y_name):
    """
    Function to do a qualitative diagnostic plot for copula fit.

    Parameters
    ----------
    x_extremes : np.array or pd.Series
        Observed x extremes (magnitude).
    y_extremes : np.array or pd.Series
        Observed y extremes (magnitude).
    x_sample : np.array or pd.Series
        Random copula sample (x) in data scale.
    y_sample : np.array or pd.Series
        Random copula sample (y) in data scale.
    x_name : string
        Name for x.
    y_name : string
        Name for y.

    Returns
    -------
    fig : matplotlib figure
        Diagnostic figure.
    ax : array of matplotlib axes
        Axes from fig.

    """

    fontsize = 15
    fig, ax = plt.subplots(ncols=3, figsize=(27, 7))

    # OBSERVED EXTREMES
    h_data = ax[0].hist2d(x_extremes, y_extremes, bins=[10, 10],
                          cmap='magma', cmin=1)

    # Some decor
    ax[0].set_xlabel(x_name, fontsize=fontsize)
    ax[0].set_ylabel(y_name, fontsize=fontsize)
    for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
        label.set_fontsize(fontsize)
    ax[0].set_facecolor('darkgray')
    t_data = ax[0].text(0.06, 0.94, '(a)', transform=ax[0].transAxes,
                        fontsize=fontsize,  va='top', ha='left')
    t_data.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[0].set_title('Observed Extremes', fontsize=fontsize)

    # Colourbar
    cbar_data = fig.colorbar(h_data[3], ax=ax[0])
    cbar_data.set_label('Occurrence', fontsize=fontsize)
    cbar_data.ax.tick_params(labelsize=fontsize)

    # COPULA SAMPLE
    h_sample = ax[1].hist2d(x_sample, y_sample, bins=[10, 10], cmap='magma',
                            cmin=1)

    # Some decor
    ax[1].set_xlabel(x_name, fontsize=fontsize)
    ax[1].set_ylabel(y_name, fontsize=fontsize)
    for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()):
        label.set_fontsize(fontsize)
    ax[1].set_facecolor('darkgray')
    t_data = ax[1].text(0.06, 0.94, '(b)', transform=ax[1].transAxes,
                        fontsize=fontsize,  va='top', ha='left')
    t_data.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[1].set_title('Copula Sample (data scale)', fontsize=fontsize)

    # Colourbar
    cbar_sample = fig.colorbar(h_sample[3], ax=ax[1])
    cbar_sample.set_label('Occurrence', fontsize=fontsize)
    cbar_sample.ax.tick_params(labelsize=fontsize)

    # Both both sets of component-wise maxima as a scatter plot
    ax[2].plot(x_extremes, y_extremes, marker='^', color="indigo",
               label='Observations', fillstyle='none', linewidth=0.)
    ax[2].plot(x_sample, y_sample, marker='*', color="darkgoldenrod",
               label="Copula Sample", fillstyle='none', linewidth=0.)

    # Some decor
    ax[2].set_xlabel(x_name, fontsize=fontsize)
    ax[2].set_ylabel(y_name, fontsize=fontsize)
    for label in (ax[2].get_xticklabels() + ax[2].get_yticklabels()):
        label.set_fontsize(fontsize)
    t_data = ax[2].text(0.06, 0.94, '(c)', transform=ax[2].transAxes,
                        fontsize=fontsize,  va='top', ha='left')
    t_data.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))
    ax[2].set_title('Comparison', fontsize=fontsize)
    ax[2].legend(fontsize=fontsize)

    # Make axes limits sample as for panel0 for panels 1 and 2
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())
    ax[2].set_xlim(ax[0].get_xlim())
    ax[2].set_ylim(ax[0].get_ylim())

    return fig, ax


def qualitative_copula_fit_check_bivariate_scatter(x_extremes, y_extremes,
                                                   x_sample, y_sample,
                                                   x_name, y_name):
    """
    Function to do a qualitative diagnostic plot for copula fit.

    Parameters
    ----------
    x_extremes : np.array or pd.Series
        Observed x extremes (magnitude).
    y_extremes : np.array or pd.Series
        Observed y extremes (magnitude).
    x_sample : np.array or pd.Series
        Random copula sample (x) in data scale.
    y_sample : np.array or pd.Series
        Random copula sample (y) in data scale.
    x_name : string
        Name for x.
    y_name : string
        Name for y.

    Returns
    -------
    fig : matplotlib figure
        Diagnostic figure.
    ax : array of matplotlib axes
        Axes from fig.

    """

    fontsize = 15
    fig, ax = plt.subplots(figsize=(10, 10))

    # Both both sets of component-wise maxima as a scatter plot
    ax.plot(x_extremes, y_extremes, marker='^', color="indigo",
            label=str(round(len(x_extremes))) + ' Observations',
            fillstyle='none', linewidth=0., markersize=fontsize)
    ax.plot(x_sample, y_sample, marker='*', color="darkgoldenrod",
            label=str(round(len(x_sample))) + " Copula Samples",
            fillstyle='none', linewidth=0., markersize=fontsize)

    # Some decor
    ax.set_xlabel(x_name, fontsize=fontsize)
    ax.set_ylabel(y_name, fontsize=fontsize)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(fontsize)

    ax.set_title('Component-wise maxima', fontsize=fontsize)
    ax.legend(fontsize=fontsize)

    return fig, ax
