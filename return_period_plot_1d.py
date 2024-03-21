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


def return_period_plot(data, fit_params, block_size, data_tag, ax, csize=15):
    
    #data = extrema
    
    
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
#    return_period=calculate_return_period(data, fit_params, block_size)
    empirical_return_period=calculate_return_period_empirical(data, block_size)
    ax.plot(empirical_return_period, data, linewidth=0.0, marker='^', fillstyle='none', color='black', label='Observations')
     
     
    
    
    
    # Overplot the model return value as a function of period
    return_period=np.linspace(0,ax.get_xlim()[1],500)
    #return_value=
    
    
    # Overplot confidence interval
    
    
    # Some decor
    ax.set_xlabel('Return Period (years)', fontsize=csize)
    #ax.set_ylabel(str(tag)+' observed at least once\nper return period (nT)', fontsize=csize)
    ax.set_xscale('log')
    ax.legend(fontsize=csize, loc='lower right')  
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(csize) 


    return ax

def calculate_return_period_empirical(data, block_size):
    """
    Function to calculate the return period of provided
    extrema based on exceedance probability.
    
    !!!!!!!!!!!!!!!See Coles 2001 pg XXXXX for theory.

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
    exceedance_prob=(rank)/(data.size + 1.0)
    
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