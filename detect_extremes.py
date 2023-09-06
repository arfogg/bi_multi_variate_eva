# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:54:38 2023

@author: A R Fogg
"""

import pandas as pd
import numpy as np

import pyextremes


def find_block_maxima(data,df_data_tag,df_time_tag='datetime',block_size=pd.to_timedelta("365.2425D"),
                      extremes_type='high', leap_year_check=False):
    """

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing a column with pd.Timestamp and a data column to 
        extract extremes from
    df_data_tag : string
        Tag describing the column of data which extremes should be extracted from.
    df_time_tag : string, optional
        Tag describing the column of data which contains pd.Timestamp. The default 
        is 'datetime'
    block_size : pd.Timedelta, optional
        Block size over which individual extremes are found. The default is 
        pd.to_timedelta("365.2425D").
    extremes_type : string, optional
        If extremes_type='high' block maxima are found. If extremes_type='low',
        block minima are found. The default is 'high'.
    leap_year_check : BOOL, optional
        If leap_year_check==True the code will check for multiple extremes in a
        in each year parsed. Note over *many* years of data this will be inadequate,
        since block_size=pd.to_timedelta("365.2425D") leaves some tiny remainder of 
        year for e.g. 10 years, which this hack will deal with. 

    Returns
    -------
    extremes : pd.DataFrame
        column for the datetime of each extremes and column containing
        the observed extreme value

    """
    
    # First, check the parsed columns exist in data
    if set([df_data_tag,df_time_tag]).issubset(data.columns):
        print('Extracting extremes for ',df_data_tag)
    else:
        print('ERROR: detect_extremes.find_block_maxima')
        print('Either '+df_data_tag+' or '+df_time_tag+' does not exist in dataframe')
        print('Exiting...')
        raise NameError(df_data_tag+' or '+df_time_tag+' does not exist in dataframe')
    
    # Convert to a pandas series with datetime as the index
    series=pd.Series(data=data[df_data_tag].values,index=data[df_time_tag])
    
    extremes=pyextremes.extremes.get_extremes(series, "BM", extremes_type=extremes_type, block_size=block_size)
    
    extremes=extremes.to_frame().reset_index()
    
    extremes.rename(columns={"extreme values":"extreme"}, inplace=True)
    print(extremes.columns)
    if leap_year_check==True:
        print('Running Leap Year double check in detect_extremes.find_block_maxima')
        years_parsed=np.array(pd.DatetimeIndex(extremes.datetime).year.unique())
        for i in range(years_parsed.size):
            ex_ind,=np.where(np.array(pd.DatetimeIndex(extremes.datetime).year) == years_parsed[i])
            if ex_ind.size>1:
                print('More than one extreme detected for ',years_parsed[i])
                remove_ind=extremes.extreme.iloc[ex_ind].idxmin()
                print('Removing duplicate extreme:')
                print(extremes.iloc[remove_ind])
                extremes.drop(index=remove_ind, inplace=True)
    
    return extremes