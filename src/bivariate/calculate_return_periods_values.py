# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:40:39 2023

@author: A R Fogg

Functions to calculate return periods.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from . import transform_uniform_margins
from . import plotting_utils


def calculate_return_period(copula, sample_grid,
                            block_size=pd.to_timedelta("365.2425D")):
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
        Return periods for given sample in years.

    """

    print('Calculating the return period over parsed copula and sample')

    # Calculate the CDF value for each point in sample
    CDF = copula.cumulative_distribution(sample_grid)

    # Calculate the number of extremes in a year
    n_extremes_per_year = pd.to_timedelta("365.2425D")/block_size

    # Calculate the return period (in years!)
    # See Coles 2001 textbook pages 81-82
    return_period = (1.0/n_extremes_per_year)*(1.0/(1-CDF))

    return return_period


def estimate_return_period_ci(bs_copula_arr, n_bootstrap,
                              sample_grid,
                              block_size=pd.to_timedelta("365.2425D"),
                              ci_percentiles=[2.5, 97.5]):
    """
    Function to estimate the Confidence Interval over the 2D
    return period matrix using bootstraps.

    Parameters
    ----------
    bs_copula_arr : list
        List of len n_bootstrap containing copulas.bivariate objects for each
        individual bootstrap.
    n_bootstrap : int
        Number of bootstraps parsed.
    sample_grid : pd.DataFrame
        Two columns with names same as copula, containing x and y values
        to find the return period for.
    block_size : pd.Timedelta, optional
        Size over which block maxima have been found. The default
        is pd.to_timedelta("365.2425D").
    ci_percentiles : list, optional
        Upper and lower percentiles for the confidence interval
        plots. The default is [2.5, 97.5].

    Returns
    -------
    rp : np.array
        Array containing the calculated return period across all bootstraps.
    ci : np.array
        Confidence interval on each rp value.

    """
    # Initialise empty return period matrix
    rp = np.full((sample_grid.shape[0], n_bootstrap), np.nan)
    for i in range(n_bootstrap):
        print('Bootstrap ', i)
        rp[:, i] = calculate_return_period(bs_copula_arr[i], sample_grid,
                                           block_size=block_size)

    # Looping through each grid pixel
    print('Calculating confidence interval')
    ci = np.full(sample_grid.shape, np.nan)
    for j in range(sample_grid.shape[0]):
        # First, select the Bootstraps where return_period is finite
        #   Infinite return_period means CDF->1, which indicates
        #   the bootstrap didn't contain the full extent of the actual
        #   observed extrema. Hence the copula CDF was saying "out
        #   of bounds!" to the requested X/Y point. So, we use only
        #   finite return periods to calculate the CI, and retain the
        #   number of bootstraps contributing to each point.
        rp_clean_index, = np.where(np.isfinite(rp[j, :]))
        rp_clean = rp[j, rp_clean_index] if rp_clean_index.size > 0 \
            else rp[j, :]
        ci[j, :] = np.percentile(rp_clean, ci_percentiles)

    return rp, ci


def generate_sample_grid(min_x, max_x, min_y, max_y,
                         x_name, y_name,
                         n_samples=1000):
    """
    Generate x vs y grid for 2D return period plot.

    Parameters
    ----------
    min_x : float
        Minimum value for x.
    max_x : float
        Maximum value for x.
    min_y : float
        Minimum value for y.
    max_y : float
        Maximum value for y.
    x_name : string
        Tag for x values.
    y_name : string
        Tag for y values.
    n_samples : int, optional
        Number of points in x and y direction for the matrix. The default
        is 1000.

    Returns
    -------
    sample_grid : np.array
        X and y points (on uniform margins) for the plotting grid. Raveled
        onto 2 column matrix.
    xv_ds : np.array
        X points for pcolormesh grid in data scale.
    yv_ds : np.array
        Y points for pcolormesh grid in data scale.
    mid_point_x_ds : np.array
        Mid x point of bin to be used to calculate return period.
    mid_point_y_ds : np.array
        Mid y point of bin to be used to calculate return period.

    """

    # Create a sample
    sample_um = \
        pd.DataFrame({x_name:
                      transform_uniform_margins.
                      transform_from_data_scale_to_uniform_margins_empirically(
                                    np.linspace(min_x, max_x, n_samples)),
                      y_name:
                      transform_uniform_margins.
                      transform_from_data_scale_to_uniform_margins_empirically(
                                    np.linspace(min_y, max_y, n_samples))})
    sample_ds = pd.DataFrame({x_name: np.linspace(min_x, max_x, n_samples),
                              y_name: np.linspace(min_y, max_y, n_samples)})

    # Create sample grid
    #   On uniform margins
    xv_um, yv_um = np.meshgrid(sample_um[x_name], sample_um[y_name])
    #   In data scale
    xv_ds, yv_ds = np.meshgrid(sample_ds[x_name], sample_ds[y_name])
    # mesh grid on uniform margins for calculating, in data scale
    #   for plotting

    # Determine mid point of each pixel to calculate return
    #   period for
    mid_point_x_um = (xv_um[1:, 1:]+xv_um[:-1, :-1])/2
    mid_point_y_um = (yv_um[1:, 1:]+yv_um[:-1, :-1])/2
    mid_point_x_ds = (xv_ds[1:, 1:]+xv_ds[:-1, :-1])/2
    mid_point_y_ds = (yv_ds[1:, 1:]+yv_ds[:-1, :-1])/2

    # Reshape
    raveled_mid_point_x = mid_point_x_um.ravel()
    raveled_mid_point_y = mid_point_y_um.ravel()
    sample_grid = np.array([raveled_mid_point_x, raveled_mid_point_y]).T

    return sample_grid, xv_ds, yv_ds, mid_point_x_ds, mid_point_y_ds


def plot_return_period_as_function_x_y(copula,
                                       min_x, max_x, min_y, max_y,
                                       x_name, y_name,
                                       x_label, y_label,
                                       bs_copula_arr, n_bootstrap,
                                       n_samples=1000,
                                       block_size=pd.to_timedelta("365.2425D"),
                                       sample_grid=None,
                                       xv_ds=None, yv_ds=None,
                                       mid_point_x_ds=None,
                                       mid_point_y_ds=None,
                                       return_period=None, ci=None,
                                       contour_levels=[1/12, 0.5, 1.0, 10.0],
                                       contour_colors=['white', 'white',
                                                       'white', 'black'],
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
    x_label : string
        Name for x, used for labelling plots.
    y_label : string
        Name for y, used for labelling plots.
    bs_copula_arr : list of copulas copulae
        Python list of length n_bootstrap containing
        copulas.bivariate object for each bootstrapped set of extrema.
    n_bootstrap : int
        Number of bootstrapped extrema generated.
    n_samples : int, optional
        Number of points for x and y axes. So return period is evaluated
        across n_samples x n_samples size grid. The default is 1000.
    block_size : pd.timedelta, optional
        Block size used in the block maxima extreme selection. The
        default is pd.to_timedelta("365.2425D").
    sample_grid : np.array, optional
        X and y points (on uniform margins) for the plotting grid. Raveled
        onto 2 column matrix. The default is None - in this case, the parameter
        is calculated.
    xv_ds : np.array, optional
        X points for pcolormesh grid in data scale. The default is None - in
        this case, the parameter is calculated.
    yv_ds : np.array, optional
        Y points for pcolormesh grid in data scale. The default is None - in
        this case, the parameter is calculated.
    mid_point_x_ds : np.array, optional
        Mid x point of bin to be used to calculate return period. The default
        is None - in this case, the parameter is calculated.
    mid_point_y_ds : np.array, optional
        Mid y point of bin to be used to calculate return period. The default
        is None - in this case, the parameter is calculated.
    return_period : np.array, optional
        Return period matrix. The default is None - in this case, the parameter
        is calculated.
    ci : np.array, optional
        Confidence interval matrix. The default is None - in this case, the
        parameter is calculated.
    contour_levels : list, optional
        Return period values at which contours will be drawn. The
        default is [1/12,0.5,1.0,10.0].
    contour_colors : list, optional
        List of string names for contour colors. The default is ['white',
        'white', 'white', 'black'].
    lower_ax_limit_contour_index : int, optional
        Used to decide the lower axes limits for x and y. The default is 1.
    ci_percentiles : list, optional
        Upper and lower percentiles for the confidence interval
        plots. The default is [2.5, 97.5].
    fontsize : float, optional
        Fontsize to be applied to all labels. The default is 15.

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
    if (sample_grid is None) | (xv_ds is
                                None) | (yv_ds is
                                         None) | (mid_point_x_ds is
                                                  None) | (mid_point_y_ds is
                                                           None):
        sample_grid, xv_ds, yv_ds, mid_point_x_ds, mid_point_y_ds = \
                            generate_sample_grid(
                                 min_x, max_x, min_y, max_y,
                                 x_name, y_name,
                                 n_samples=1000)

    # Calculate return period if not parsed
    if return_period is None:
        return_period = calculate_return_period(copula, sample_grid,
                                                block_size=block_size)

    # Reshape for mesh grid
    shaped_return_period = return_period.reshape(mid_point_x_ds.shape)

    # Calculate confidence intervals if needed
    if (ci is None):
        rp, ci = estimate_return_period_ci(bs_copula_arr, n_bootstrap,
                                           sample_grid, block_size=block_size,
                                           ci_percentiles=ci_percentiles)

    # Initialise plot
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))

    # ----- RETURN PERIOD -----
    rp_cbar_norm = colors.LogNorm(vmin=np.quantile(shaped_return_period, 0.1),
                                  vmax=np.quantile(shaped_return_period,
                                                   0.999))

    # Plot return period as function of x and y in data scale
    pcm = ax[0].pcolormesh(xv_ds, yv_ds, shaped_return_period, cmap='plasma',
                           norm=rp_cbar_norm)

    # Contours
    contours = ax[0].contour(mid_point_x_ds, mid_point_y_ds,
                             shaped_return_period, contour_levels,
                             colors=contour_colors)
    for c in contours.collections:
        c.set_clip_on(True)
    clabels = plotting_utils.contour_labels(contours, xpad=max_x*0.005,
                                            sides=['left', 'right'],
                                            color='black', ha='left',
                                            va='center')

    # Colourbar
    cbar = fig.colorbar(pcm, ax=ax[0], extend='max',
                        label='Return Period (years)', pad=0.06)
    cbar.add_lines(contours)

    # Some Decor
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel(y_label)
    ax[0].set_title('(a) Return Period\n'+r'($\tau_{max}$ = ' +
                    str(round(np.nanmax(shaped_return_period), 2))+' years)')

    # ----- LOWER QUANTILE -----
    # Plot lower quantile as function of x and y in data scale
    lower_quantile = ci[:, 0].reshape(mid_point_x_ds.shape)
    lci_cbar_norm = colors.LogNorm(vmin=np.nanquantile(lower_quantile, 0.1),
                                   vmax=np.nanquantile(lower_quantile, 0.999))
    pcm_lq = ax[1].pcolormesh(xv_ds, yv_ds, lower_quantile, cmap='magma',
                              norm=lci_cbar_norm)

    # Contours
    contours_lq = ax[1].contour(mid_point_x_ds, mid_point_y_ds, lower_quantile,
                                contour_levels, colors=contour_colors)
    for c in contours_lq.collections:
        c.set_clip_on(True)
    clabels = plotting_utils.contour_labels(contours_lq, xpad=max_x*0.005,
                                            sides=['left', 'right'],
                                            color='black', ha='left',
                                            va='center')

    # Colourbar
    cbar = fig.colorbar(pcm_lq, ax=ax[1], extend='max',
                        label='Return Period (years)')
    cbar.add_lines(contours_lq)

    # Some Decor
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel(y_label)

    ax[1].set_title('(b) ' + str(ci_percentiles[1]-ci_percentiles[0]) +
                    '% Confidence Interval (lower)\n' + r'($\tau_{max}$ = '
                    + str(round(np.nanmax(lower_quantile), 2)) + ' years)')

    # ----- UPPER QUANTILE -----
    # Plot upper quantile as function of x and y in data scale
    upper_quantile = ci[:, 1].reshape(mid_point_x_ds.shape)
    uci_cbar_norm = colors.LogNorm(vmin=np.nanquantile(upper_quantile, 0.1),
                                   vmax=np.nanquantile(upper_quantile, 0.999))
    pcm_uq = ax[2].pcolormesh(xv_ds, yv_ds, upper_quantile, cmap='magma',
                              norm=uci_cbar_norm)
    # Contours
    contours_uq = ax[2].contour(mid_point_x_ds, mid_point_y_ds, upper_quantile,
                                contour_levels, colors=contour_colors)
    for c in contours_uq.collections:
        c.set_clip_on(True)
    clabels = plotting_utils.contour_labels(contours_uq, xpad=max_x*0.005,
                                            sides=['left', 'right'],
                                            color='black', ha='left',
                                            va='center')
    # Colourbar
    cbar = fig.colorbar(pcm_uq, ax=ax[2], extend='max',
                        label='Return Period (years)')
    cbar.add_lines(contours_uq)

    # Some Decor
    ax[2].set_xlabel(x_label)
    ax[2].set_ylabel(y_label)
    ax[2].set_title('(c) ' + str(ci_percentiles[1]-ci_percentiles[0]) +
                    '% Confidence Interval (upper)\n' + r'($\tau_{max}$ = '
                    + str(round(np.nanmax(upper_quantile), 2)) + ' years)')

    # Set x and y limits
    # Work out x and y limits based on parsed contour index
    xy_contour_limit = contours.allsegs[lower_ax_limit_contour_index][0]
    xlim = np.min(xy_contour_limit[:, 0])*0.9
    ylim = np.min(xy_contour_limit[:, 1])*0.9
    for i, a in np.ndenumerate(ax):
        a.set_xlim(left=xlim)
        a.set_ylim(bottom=ylim)

    fig.tight_layout()

    return fig, ax
