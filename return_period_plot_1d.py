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


def return_period_plot(data, fit_params, block_size, data_tag, data_units_fm, ax, csize=15, line_color='darkcyan', ci=0.95):
    
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
    empirical_return_period=calculate_return_period_empirical(data, block_size)
    ax.plot(empirical_return_period, data, linewidth=0.0, marker='^', fillstyle='none', color='black', label='Observations')   
    
    # Overplot the model return value as a function of period
    model_data=np.linspace(np.nanmin(data)*0.8,np.nanmax(data)*1.2,201)
    model_return_period=calculate_return_period_CDF(model_data, fit_params, block_size)
    ax.plot(model_return_period, model_data, linewidth=1.5, color=line_color, label=fit_params.formatted_dist_name[0])
    
    # Overplot confidence interval
    # ax.fill_between(x,lower_ci,upper_ci,color='grey',alpha='0.5',label=str(ci*100.)+'% CI')
    
    # Some decor
    ax.set_xlabel('Return Period (years)')
    ax.set_ylabel(str(data_tag)+' observed at least once\nper return period ('+str(data_units_fm)+')')
    ax.set_xscale('log')
    ax.legend(loc='lower right')  
    ax.set_title('Return Period Plot')

    return ax

def return_period_table_ax(ax, periods=[2, 5, 10, 15, 20, 25, 50, 100], ci=0.95):
    
    print('Creating a table of return periods and values')
    
    # # Plot a table of return values
    # summary = eva_model.get_summary(
    #     return_period=[2, 5, 10,15,20, 25, 50, 100],
    #     alpha=0.95 )
    # summary=summary.reset_index()

    # # Format the DF for the table
    # summary=summary.round()
    # summary=summary.rename(columns={"return period": "period",
    #                         "return value": "value",
    #                         "lower ci": "-95% CI",
    #                         "upper ci": "+95% CI"})

    # summary_new=pd.DataFrame({"period":summary['period'],
    #                           "value":summary['value'],
    #                           "-95% CI":summary['value'] - summary['-95% CI'],
    #                           "+95% CI":summary['+95% CI'] - summary['value']
    #                           })


    # table=ax_model[1,1].table(cellText=summary_new.values, colLabels=summary_new.columns, loc='center')#, fontsize=csize+2)
    # table.auto_set_font_size(False) # stop auto font size
    # table.set_fontsize(csize)       # increase font size
    # table.scale(1,3)    # don't increase cell width (1) but increase height x3
    # ax_model[1,1].axis('off')
    # t=ax_model[1,1].text(0.06,0.94,'(d)', transform=ax_model[1,1].transAxes, fontsize=csize, va='top', ha='left')
    # t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='grey'))

    
    
    
    return ax
    
    
    

def calculate_return_period_empirical(data, block_size):
    """
    Function to calculate the return period of provided
    extrema based on exceedance probability.
    
    Pr_exceedance = 1 - (rank / (n + 1) )
    rank - the ranking of the ordered data
    n - number of data points
    
    tau = 1 / (Pr_exceedance * n_ex_per_year)
    
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
    # Calculate return values
    
    print('banana')
    
    
    
    
    return