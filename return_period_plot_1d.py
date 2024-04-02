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


def return_period_plot(data, bs_data, fit_params, block_size, data_tag, data_units_fm, ax,
                       n_ci_bootstrap=100, csize=15, line_color='darkcyan', ci=0.95):
    
    #data = extrema
    # bs_data = data.size * n_ci_bootstrap
    
    print('Creating a Return Period plot')
    
    #options
    # % CI
    
    
    
    

    # observed_return_values=pyextremes.get_return_periods(ts=eva_model.data, extremes=eva_model.extremes, 
    #         extremes_method=eva_model.extremes_method, extremes_type=eva_model.extremes_type,
    #         block_size=eva_model.extremes_kwargs.get("block_size", None), return_period_size='365.2425D' )
    # return_period=np.linspace(observed_return_values.loc[:, "return period"].min(),
    #         observed_return_values.loc[:, "return period"].max(),100,)    
    # modeled_return_values = eva_model.get_summary(return_period=return_period, return_period_size='365.2425D',alpha=0.95)

    # ax_model[0,0].plot(observed_return_values['return period'], observed_return_values[tag], linewidth=0.0, marker='^', fillstyle='none', color='black', label='Observations')
    # ax_model[0,0].plot(modeled_return_values.index, modeled_return_values['return value'], linewidth=2.0, color='coral', label='Model')
    # ax_model[0,0].fill_between(modeled_return_values.index, modeled_return_values['lower ci'], modeled_return_values['upper ci'], color='grey', alpha=0.5, label='95% CI')
    
  
    
    
    # Plot observed extremes as a function of their return period
    empirical_return_period=calculate_return_period_empirical(data, block_size)
    ax.plot(empirical_return_period, data, linewidth=0.0, marker='^', fillstyle='none', color='black', label='Observations')   
    
    # Overplot the model return value as a function of period
    model_data=np.linspace(np.nanmin(data)*0.8,np.nanmax(data)*1.2,201)
    model_return_period=calculate_return_period_CDF(model_data, fit_params, block_size)
    ax.plot(model_return_period, model_data, linewidth=1.5, color=line_color, label=fit_params.formatted_dist_name[0])
    
    # Overplot confidence interval
    # bs_data=np.full((data.size, n_ci_bootstrap), np.nan)
    bs_return_period=np.full(bs_data.shape,np.nan)
    for i in range(n_ci_bootstrap):
    #     bs_data[:,i]=np.sort(np.random.choice(data, size=data.size, replace=True))
        
        bs_return_period[:,i]=calculate_return_period_empirical(bs_data[:,i],block_size)
        ax.plot(bs_return_period[:,i], bs_data[:,i], color='magenta', alpha=0.1)
        
         
   
    # ax.fill_between(x,lower_ci,upper_ci,color='grey',alpha='0.5',label=str(ci*100.)+'% CI')
    
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