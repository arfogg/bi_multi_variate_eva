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


def return_period_plot(data, bs_dict, fit_params, block_size, data_tag, data_units_fm, ax,
                       csize=15, line_color='darkcyan', ci_percentiles=[2.5, 97.5]):
    """
    Function to generate a return period plot.

    Parameters
    ----------
    data : np.array
        Observed extrema.
    bs_dict : dictionary
        Containing keys listed below.
        n_iterations = number of bootstrap iterations
        bs_data = bootstrapped extrema, of shape 
            number of extrema x n_iterations
        n_ci_iterations = number of iterations for
            calculation of confidence interval
        periods = return periods to evaluate level at
            for confidence interval calculation
        levels = return levels across n_ci_iterations
            of model fits
        distribution_name = 'genextreme' or 'gumbel_r'
        shape_ = array of shape parameters from fitting
        location = array of location parameters from 
            fitting
        scale = array of scale parameters from fitting
    fit_params : pandas.DataFrame
        df containing tags including distribution_name,
        shape_, scale, location
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
    empirical_return_period=calculate_return_period_empirical(data, block_size)
    ax.plot(empirical_return_period, data, linewidth=0.0, marker='^', fillstyle='none', color='black', label='Observations')   
    
    # Overplot the model return value as a function of period
    model_data=np.linspace(np.nanmin(data)*0.8,np.nanmax(data)*1.2,201)
    model_return_period=calculate_return_period_CDF(model_data, fit_params, block_size)
    ax.plot(model_return_period, model_data, linewidth=1.5, color=line_color, label=fit_params.formatted_dist_name[0])
    
    # Overplot confidence interval
    plot_ind,=np.where(bs_dict['periods'] <= np.max(model_return_period))
    percentiles=np.percentile(bs_dict['levels'][plot_ind,], ci_percentiles, axis=1).T
    ax.plot(bs_dict['periods'][plot_ind], percentiles, color='grey', linestyle="--", linewidth=1.0)
    ax.fill_between(bs_dict['periods'][plot_ind],percentiles[:,0],percentiles[:,1],color='grey',alpha=0.5,label=str(ci_percentiles[1]-ci_percentiles[0])+'% CI')
    
    # Some decor
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel(str(data_tag)+' observed at least once\nper return period ('+str(data_units_fm)+')')
    ax.set_xscale('log')
    ax.legend(loc='lower right')  
    ax.set_title('Return Period Plot')

    return ax

def return_period_table_ax(ax, fit_params, block_size, data_units_fm,
                           periods=np.array([2, 5, 10, 15, 20, 25, 50, 100]), ci=0.95):
    """
    Function to generate a table of return levels, at
    given periods, on the input axis object.

    Parameters
    ----------
    ax : matplotlib axis object
        The axes to plot the return period table onto.
    fit_params : pd.DataFrame
        Contains columns as returned by
        fit_model_to_extremes.fit_gevd_or_gumbel.
    block_size : pd.Timedelta
        Size chosen for block maxima selection.
    data_units_fm : string
        Units for data to be put in axes labels etc.
    periods : np.array, optional
        Array of return periods (in years) to evaluate
        the return level at. 
        The default is np.array([2, 5, 10, 15, 20, 25, 50, 100]).
    ci : float, optional
        Decimal confidence interval for errors (i.e. 0.95 is 
        95% confidence interval). The default is 0.95.

    Returns
    -------
    ax : matplotlib axis object
        Axis containing the table of values.

    """
    print('Creating a table of return periods and values')
    
    levels = calculate_return_value_CDF(periods, fit_params, block_size)
    
    table_df=pd.DataFrame({"period\n(years)":periods,
                          "level\n("+data_units_fm+")":["%.2f" % v for v in levels],
                          "-"+str("%.0f" % (ci*100))+"% CI":np.full(levels.size,"TBC"),
                          "+"+str("%.0f" % (ci*100))+"% CI":np.full(levels.size,"TBC")
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
    # NEED TO CHECK THE 1- WITH DÁIRE
    
    # Calculate Return Period
    tau=(1.0/(exceedance_prob*extrema_per_year))

    return tau

def calculate_return_period_CDF(data, fit_params, block_size):
    
    """
    Function to calculate the return period of provided
    extrema based on CDF model.
    
    See Coles 2001 pg 81-82 for theory.

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
       
    if fit_params.distribution_name[0]=='genextreme':
        tau = 1.0 / ( (extrema_per_year) * (1.0 - genextreme.cdf(data, fit_params.shape_, loc=fit_params.location, scale=fit_params.scale)) )
    
    elif fit_params.distribution_name[0]=='gumbel_r':
        tau = 1.0 / ( (extrema_per_year) * (1.0 - gumbel_r.cdf(data, loc=fit_params.location, scale=fit_params.scale)) )

    return tau

def calculate_return_value_CDF(periods, fit_params, block_size):
    """
    Function to calculate the return levels based on a list of
    periods and GEVD/Gumbel fit.

    Parameters
    ----------
    periods : np.array
        The return periods (in years) to evaluate the return
        levels over.
    fit_params : pd.DataFrame
        Contains columns as returned by
        fit_model_to_extremes.fit_gevd_or_gumbel..
    block_size : pd.Timedelta
        Size chosen for block maxima selection..

    Returns
    -------
    levels : np.array
        Calculated return levels.

    """
    
    extrema_per_year=pd.to_timedelta("365.2425D")/block_size
    
    return_periods=periods*extrema_per_year
    
    # !!! maths here needs to be checked with Dáire
    if fit_params.distribution_name[0]=='genextreme':
        levels=genextreme.ppf(1.-(1./return_periods), fit_params.shape_, 
                              loc=fit_params.location, scale=fit_params.scale)
    elif fit_params.distribution_name[0]=='gumbel_r':
        levels=gumbel_r.ppf(1.-(1./return_periods), 
                              loc=fit_params.location, scale=fit_params.scale)
    
    return levels

def return_level_bootstrapped_data(bs_data, n_bootstrap, distribution_name, block_size,
                                   return_periods):
    
    

        

    if distribution_name == 'genextreme':
        shape_=np.full(n_bootstrap, np.full)
        location=np.full(n_bootstrap, np.full)
        scale=np.full(n_bootstrap, np.full)        

        for i in range(n_bootstrap):
            shape_[i],location[i],scale[i]=genextreme.fit(bs_data[:,i])
            
    
    elif distribution_name == 'gumbel_r':
        shape_=np.full(n_bootstrap, 0.0)
        location=np.full(n_bootstrap, np.full)
        scale=np.full(n_bootstrap, np.full)        

        for i in range(n_bootstrap):
            location[i],scale[i]=gumbel_r.fit(bs_data[:,i])
    
    levels=np.full((return_periods.size, n_bootstrap), np.nan)
    for i in range(n_bootstrap):
        levels[:,i]=calculate_return_value_CDF(return_periods, 
                                               pd.DataFrame({'distribution_name':distribution_name,
                                                             'shape_':shape_[i],
                                                             'location':location[i],
                                                             'scale':scale[i]}, index=[0]),
                                               block_size)

    return levels, shape_, location, scale
        