# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 09:21:15 2023

@author: A R Fogg
"""

import pyextremes

import pandas as pd
import numpy as np

def fit_gevd_or_gumbel(extremes_df,df_data_tag,df_time_tag='datetime',
                       fitting_type='Emcee', distribution='genextreme'):
    """
    Function to fit a GEVD or GUMBEL distribution to preselected extremes.

    Parameters
    ----------
    extremes_df : pd.DataFrame
        DataFrame containing a column with pd.Timestamp and a data column to 
        extract extremes from
    df_data_tag : string
        Tag describing the column of data which extremes should be extracted from.
    df_time_tag : TYPE, optional
        Tag describing the column of data which contains pd.Timestamp. The default 
        is 'datetime'.
    fitting_type : string, optional
        Fitting method to use for fitting the model. The default
        is 'Emcee', other valid option is 'MLE'.
    distribution : string, optional
        Distribution to fit extremes to. The default is 'genextreme', other option
        is 'gumbel_r'. These are suitable for block_maxima selected extremes only.

    Raises
    ------
    NameError
        DESCRIPTION.

    Returns
    -------
    fit_params : pd.DataFrame
        DataFrame containing info on the fitting GEVD or Gumbel distribution

    """
    
    # First, check the parsed columns exist in data
    if set([df_data_tag,df_time_tag]).issubset(extremes_df.columns):
        print('Extracting extremes for ',df_data_tag)
    else:
        print('ERROR: detect_extremes.find_block_maxima')
        print('Either '+df_data_tag+' or '+df_time_tag+' does not exist in dataframe')
        print('Exiting...')
        raise NameError(df_data_tag+' or '+df_time_tag+' does not exist in dataframe')

    
    # Convert to a pandas series with datetime as the index
    extremes_series=pd.Series(data=extremes_df[df_data_tag].values,index=extremes_df[df_time_tag])

    # Initialise the EVA class, with parsed extremes
    print('Initialising EVA class with parsed extremes')
    eva=pyextremes.EVA.from_extremes(extremes_series)
    
    # Fit a model to the extremes
    eva.fit_model(model=fitting_type)
    
    # Extract fit parameters
    fit_params=pd.DataFrame(eva.model.fit_parameters, index=[0])
    fit_params.rename(columns={'c':'shape_', 'loc':'location'}, inplace=True)
    
    # Caculate confidence intervals on the fit parameters
    if eva.distribution.name == 'gumbel_r':
        fit_params['shape_']=0.0
        location_quantiles=np.quantile(eva.model.trace[:,:,0].flatten(), [0.025, 0.975])
        scale_quantiles=np.quantile(eva.model.trace[:,:,1].flatten(), [0.025, 0.975])
        
        fit_params['shape_lower_ci_width']=np.nan
        fit_params['shape_upper_ci_width']=np.nan
        
        fit_params['location_lower_ci_width']=fit_params.location-location_quantiles[0]
        fit_params['location_upper_ci_width']=location_quantiles[1]-fit_params.location
        
        fit_params['scale_lower_ci_width']=fit_params.scale-scale_quantiles[0]
        fit_params['scale_upper_ci_width']=scale_quantiles[1]-fit_params.scale

    else:
        # Calculate the 95% confidence intervals on fit params
        shape_quantiles=np.quantile(eva.model.trace[:,:,0].flatten(), [0.025, 0.975])
        location_quantiles=np.quantile(eva.model.trace[:,:,1].flatten(), [0.025, 0.975])
        scale_quantiles=np.quantile(eva.model.trace[:,:,2].flatten(), [0.025, 0.975])
        
        fit_params['shape_lower_ci_width']=fit_params.shape_-shape_quantiles[0]
        fit_params['shape_upper_ci_width']=shape_quantiles[1]-fit_params.shape_
        
        fit_params['location_lower_ci_width']=fit_params.location-location_quantiles[0]
        fit_params['location_upper_ci_width']=location_quantiles[1]-fit_params.location
        
        fit_params['scale_lower_ci_width']=fit_params.scale-scale_quantiles[0]
        fit_params['scale_upper_ci_width']=scale_quantiles[1]-fit_params.scale
    
    fit_params['distribution_name']= eva.distribution.name   

    return fit_params