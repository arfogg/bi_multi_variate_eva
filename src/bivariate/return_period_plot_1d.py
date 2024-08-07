# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:31:24 2024

@author: A R Fogg
"""

import numpy as np
import pandas as pd

from scipy.stats import genextreme
from scipy.stats import gumbel_r
from scipy import stats


def return_period_plot(gevd_fitter, bootstrap_gevd_fit, block_size, data_tag, data_units_fm, ax,
                       csize=15, line_color='darkcyan', ci_percentiles=[2.5, 97.5]):
    """
    Function to generate a return period plot.

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
    block_size : pd.Timedelta
        Size over which block maxima have been found,
        e.g. pd.to_timedelta("365.2425D").
    data_tag : string
        name of data to be put in figure captions etc
    data_units_fm : string
        units for data to be put in axes labels etc
    ax : Matplotlib axes object
        Axes to do the plotting on.
    csize : int, optional
        Fontsize for the labels etc. The default is 15.
    line_color : string, optional
        Named Matplotlib color for the fitted model
        line. The default is 'darkcyan'.
    ci_percentiles : list, optional
        Pair of floats which define the upper and lower
        percentiles of the confidence interval shade. 
        The default is [2.5, 97.5].

    Returns
    -------
    ax : Matplotlib axes object
        Return period plot.

    """
       
    print('Creating a Return Period plot')
    
    # Plot observed extremes as a function of their return period
    empirical_return_period=calculate_return_period_empirical(gevd_fitter.extremes, block_size)
    ax.plot(empirical_return_period, gevd_fitter.extremes, linewidth=0.0, marker='^', fillstyle='none', color='black', label='Observations')   
    
    # Overplot the model return value as a function of period
    model_data=np.linspace(np.nanmin(gevd_fitter.extremes)*0.8,np.nanmax(gevd_fitter.extremes)*1.2,201)
    model_return_period=calculate_return_period_CDF(model_data, gevd_fitter, block_size)
    ax.plot(model_return_period, model_data, linewidth=1.5, color=line_color, label=gevd_fitter.formatted_dist_name)
    
    # Overplot confidence interval
    plot_ind,=np.where(bootstrap_gevd_fit.periods <= np.max(model_return_period))
    percentiles=np.percentile(bootstrap_gevd_fit.levels[plot_ind,], ci_percentiles, axis=1).T
    ax.plot(bootstrap_gevd_fit.periods[plot_ind], percentiles, color='grey', linestyle="--", linewidth=1.0)
    ax.fill_between(bootstrap_gevd_fit.periods[plot_ind],percentiles[:,0],percentiles[:,1],color='grey',alpha=0.5,
                    label=str(ci_percentiles[1]-ci_percentiles[0])+'% CI')
    
    # Some decor
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel(str(data_tag)+' observed at least once\nper return period ('+str(data_units_fm)+')')
    ax.set_xscale('log')
    ax.legend(loc='lower right')  
    ax.set_title('Return Period Plot')

    return ax

def return_period_table_ax(ax, gevd_fitter, block_size, data_units_fm, bootstrap_gevd_fit,
                           periods=np.array([2, 5, 10, 15, 20, 25, 50, 100]), 
                           ci_percentiles=[2.5, 97.5]):
    """
    Function to generate a table of return levels, at
    given periods, on the input axis object.

    Parameters
    ----------
    ax : matplotlib axis object
        The axes to plot the return period table onto.
    gevd_fitter : gevd_fitter class
        Object containing GEVD fitting information.
    block_size : pd.Timedelta
        Size chosen for block maxima selection.
    data_units_fm : string
        Units for data to be put in axes labels etc.
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
    periods : np.array, optional
        Array of return periods (in years) to evaluate
        the return level at. 
        The default is np.array([2, 5, 10, 15, 20, 25, 50, 100]).
    ci_percentiles : list, optional
        Pair of floats which define the upper and lower
        percentiles of the confidence interval shade. 
        The default is [2.5, 97.5].

    Returns
    -------
    ax : matplotlib axis object
        Axis containing the table of values.

    """
    print('Creating a table of return periods and values')
    
    # Calculate return level
    levels = calculate_return_value_CDF(periods, gevd_fitter, block_size)
    
    # Confidence intervals
    period_ind=np.full(periods.size, np.nan)
    for i in range(period_ind.size):
        period_ind[i],=np.where(bootstrap_gevd_fit.periods == periods[i])
    period_ind=period_ind.astype('int')
    percentiles=np.percentile(bootstrap_gevd_fit.levels[period_ind,], ci_percentiles, axis=1).T
    
    table_df=pd.DataFrame({"period\n(years)":periods,
                          "level\n("+data_units_fm+")":["%.2f" % v for v in levels],
                          "-"+str("%.0f" % (ci_percentiles[1]-ci_percentiles[0]))+"% CI":["%.2f" % v for v in percentiles[:,0]],
                          "+"+str("%.0f" % (ci_percentiles[1]-ci_percentiles[0]))+"% CI":["%.2f" % v for v in percentiles[:,1]]
                          })
    
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc='center')
    table.scale(1,2)
    ax.axis('off')
    
    return ax


def calculate_return_period_empirical(data, block_size):
    """
    Function to calculate the return period of provided
    extrema based on exceedance probability.
    
    Pr_exceedance = 1 - (rank / (n + 1) )
    rank - the ranking of the ordered data
    n - number of data points
    
    tau = 1 / (Pr_exceedance * n_ex_per_year)
    
    !!!!!!!!!!!!!!! for theory.

    Parameters
    ----------
    data : np.array
        Extrema / y values for return period
        plot.
    fit_params : pd.DataFrame
        Contains columns as returned by
        fit_model_to_extremes.fit_gevd_or_gumbel.
    block_size : pd.Timedelta
        Size chosen for block maxima selection.

    Returns
    -------
    tau : np.array
        Return period for data in years.

    """
    
    # Calculate the number of extrema per year based on block_size
    extrema_per_year=pd.to_timedelta("365.2425D")/block_size
    
    # Rank the data
    rank=stats.rankdata(data)
    
    # Calculate exceedance probability
    exceedance_prob=1.-((rank)/(data.size + 1.0))
    
    # Calculate Return Period
    tau=(1.0/(exceedance_prob*extrema_per_year))

    return tau

def calculate_return_period_CDF(data, gevd_fitter, block_size):
    
    """
    Function to calculate the return period of provided
    extrema based on CDF model.
    
    See Coles 2001 pg 81-82 for theory.

    Parameters
    ----------
    data : np.array
        Extrema / y values for return period
        plot.
    gevd_fitter : gevd_fitter class
        Object containing GEVD fitting information.
    block_size : pd.Timedelta
        Size chosen for block maxima selection.

    Returns
    -------
    tau : np.array
        Return period for data in years.

    """
    
    # Calculate the number of extrema per year based on block_size
    extrema_per_year=pd.to_timedelta("365.2425D")/block_size
       
    tau = 1.0 / ( (extrema_per_year) * (1.0 - gevd_fitter.frozen_dist.cdf(data)) )

    return tau

def calculate_return_value_CDF(periods, gevd_fitter, block_size):
    """
    Function to calculate the return levels based on a list of
    periods and GEVD/Gumbel fit.

    Parameters
    ----------
    periods : np.array
        The return periods (in years) to evaluate the return
        levels over.
    gevd_fitter : gevd_fitter class
        Object containing GEVD fitting information.
    block_size : pd.Timedelta
        Size chosen for block maxima selection.

    Returns
    -------
    levels : np.array
        Calculated return levels.

    """
    
    extrema_per_year=pd.to_timedelta("365.2425D")/block_size
    
    return_periods=periods*extrema_per_year
    
    levels = gevd_fitter.frozen_dist.ppf(1.-(1./return_periods))
    
    return levels

def return_level_bootstrapped_data(bootstrap_gevd_fit, n_bootstrap, return_periods):
    """
    

    Parameters
    ----------
    bs_data : np.array
        Bootstrapped extrema. Of shape number
        of extrema x n_bootstrap.
    n_bootstrap : int
        Number of desired bootstraps. 
    distribution_name : string
        Distribution to fit to for bootstraps. 
    block_size : pd.Timedelta
        Size over which block maxima have been found,
        e.g. pd.to_timedelta("365.2425D").
    return_periods : np.array
        Return periods to calculate return levels at.
    true_fit_params : gevd_fitter class
        See gevd_fitter.py.

    Returns
    -------
    levels : np.array
        Return levels at parsed return periods. Of
        shape return_periods.size x n_bootstrap
    shape_ : np.array
        Shape parameters for GEVD/Gumbel fits.
    location : np.array
        Location parameters for GEVD/Gumbel fits.
    scale : np.array
        Scale parameters for GEVD/Gumbel fits.

    """

    # # For GEVD distribution
    # if distribution_name == 'genextreme':
    #     shape_=np.full(n_bootstrap, np.nan)
    #     location=np.full(n_bootstrap, np.nan)
    #     scale=np.full(n_bootstrap, np.nan)        
    #     # Fit GEVD for each bootstrap iteration
    #     for i in range(n_bootstrap):
    #         shape_[i],location[i],scale[i]=genextreme.fit(bs_data[:,i],
    #                                                       true_fit_params.shape_,
    #                                                       loc=true_fit_params.location,
    #                                                       scale=true_fit_params.scale)      
    
    # # For Gumbel distribution
    # elif distribution_name == 'gumbel_r':
    #     shape_=np.full(n_bootstrap, 0.0)
    #     location=np.full(n_bootstrap, np.nan)
    #     scale=np.full(n_bootstrap, np.nan)        
    #     # Fit Gumbel for each bootstrap iteration
    #     for i in range(n_bootstrap):
    #         location[i],scale[i]=gumbel_r.fit(bs_data[:,i],
    #                                           loc=true_fit_params.location,
    #                                           scale=true_fit_params.scale)
    
    
    shape_=np.full(n_bootstrap, np.nan)
    location=np.full(n_bootstrap, np.nan)
    scale=np.full(n_bootstrap, np.nan) 
    levels=np.full((return_periods.size, n_bootstrap), np.nan)
    
    for i in range(n_bootstrap):
        # Store the params
        shape_[i] = bootstrap_gevd_fit.gevd_fitter_arr[i].shape_
        location[i] = bootstrap_gevd_fit.gevd_fitter_arr[i].location
        scale[i] = bootstrap_gevd_fit.gevd_fitter_arr[i].scale
        
        # For each bootstrap iteration, calculate the return level
        #   as a function of the parsed return periods, given the 
        #   iterated fitting parameters
        levels[:,i]=calculate_return_value_CDF(return_periods, 
                                               bootstrap_gevd_fit.gevd_fitter_arr[i], 
                                               bootstrap_gevd_fit.block_size)
        # levels[:,i]=calculate_return_value_CDF(return_periods, 
        #                                        pd.DataFrame({'distribution_name':distribution_name,
        #                                                      'shape_':shape_[i],
        #                                                      'location':location[i],
        #                                                      'scale':scale[i]}, index=[0]),
        #                                        block_size)

    return levels, shape_, location, scale
        