# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:54:38 2023

@author: A R Fogg
"""

import pandas as pd
import numpy as np

import pyextremes


def find_joint_block_maxima(data, df_x_tag, df_y_tag, 
                            df_time_tag='datetime',
                            md_fr = 0.5,
                            block_size=pd.to_timedelta("365.2425D"),
                            extremes_type='high'):
    """
    Find block maxima for two variables, ensuring the output has maxima for
    both datasets over the same block in each row
    
    Please note this function was inspired by pyextremes. Please see
    https://github.com/georgebv/pyextremes/tree/master
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing a column with pd.Timestamp and two data columns
        with names df_x_tag and df_y_tag.
    df_x_tag : string
        Tag describing the column of data which extremes should be extracted 
        from.
    df_y_tag : string
        Tag describing the column of data which extremes should be extracted 
        from.
    df_time_tag : string, optional
        Tag describing the column of data which contains pd.Timestamp. 
        The default is 'datetime'.
    md_fr : float
        Any block with greater than md_fr*100 percentage missing data
        is returned as an empty slice. The default is 0.5.
    block_size : pd.Timedelta, optional
        Block size over which individual extremes are found. The default is 
        pd.to_timedelta("365.2425D").
    extremes_type : string, optional
        If extremes_type='high' block maxima are found. If extremes_type='low',
        block minima are found. The default is 'high'.

    Returns
    -------
    empty_blocks : list of pd.Interval
        Blocks in which there was inadequate data coverage according to
        parsed md_fr.
    x_extreme_t : list of pd.Timestamp
        Datetime of the x_extrema within each block.
    x_extreme : list of floats
        Value of the x extrema.
    y_extreme_t : list of pd.Timestamp
        Datetime of the y_extreme within each block.
    y_extreme : list of floats
        Value of the y extrema.

    """
    
    # First, check the parsed columns exist in data
    if set([df_x_tag, df_y_tag, df_time_tag]).issubset(data.columns):
        print('Extracting extremes for ',df_x_tag, ' and ', df_y_tag)
    else:
        print('ERROR: detect_extremes.find_block_maxima')
        print('  Either '+df_x_tag+' or '+df_y_tag+' or '+df_time_tag+' does not exist in dataframe')
        print('  Exiting...')
        raise NameError(df_x_tag+' or '+df_y_tag+' or '+df_time_tag+' does not exist in dataframe')
    
    # Check the parsed extremes type is valid
    if extremes_type not in ['high', 'low']:
        print('ERROR: invalid extremes_type entered.')
        print('  Valid options are "high" or "low".')
        print('  Exiting...')
        raise ValueError(extremes_type + " is not a valid option for extremes_type!")
    
    # Select extreme selection function
    if extremes_type == "high":
        extreme_selection_func = np.nanargmax
    else:
        extreme_selection_func = np.nanargmin
        
    # Define the time blocks, rounding up so the last block may be short!
    n_time_blocks = int( np.ceil( (data[df_time_tag].max()-data[df_time_tag].min()) / block_size ) )
    time_blocks = pd.interval_range(
        start = data[df_time_tag].iloc[0],
        freq = block_size,
        periods = n_time_blocks,
        closed = "left")
    
    empty_blocks, x_extreme_t, x_extreme, y_extreme_t, y_extreme = [], [], [], [], []
    for block in time_blocks:
        # Loop through defined time blocks, and select
        #   a subset of the DataFrame
        df_slice = data.loc[ (data[df_time_tag] >= block.left) 
                            & (data[df_time_tag] < block.right) ]

        # If there's data
        if len(df_slice) > 0:
            # Calculate the fraction of rows which are == np.nan
            #   in the slice
            x_fr = df_slice[df_x_tag].isna().sum()/len(df_slice)
            y_fr = df_slice[df_y_tag].isna().sum()/len(df_slice)
            
            if (x_fr > md_fr) | (y_fr > x_fr) : 
                # Do not record an extreme for this interval,
                #   instead output as a missing interval
                empty_blocks.append(block)
            else:
                # Extract extrema
                x_idx = extreme_selection_func(df_slice[df_x_tag])
                x_extreme_t.append(df_slice[df_time_tag].iloc[x_idx])
                x_extreme.append(df_slice[df_x_tag].iloc[x_idx])
                
                y_idx = extreme_selection_func(df_slice[df_y_tag])
                y_extreme_t.append(df_slice[df_time_tag].iloc[y_idx])
                y_extreme.append(df_slice[df_y_tag].iloc[y_idx])
                
        else:
            empty_blocks.append(block)
            
    return empty_blocks, x_extreme_t, x_extreme, y_extreme_t, y_extreme


def find_block_maxima_independently_using_pyextremes(data,df_data_tag,df_time_tag='datetime',block_size=pd.to_timedelta("365.2425D"),
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
    
    # errors='ignore' means block with no data will be ignored
    extremes=pyextremes.extremes.get_extremes(series, "BM", extremes_type=extremes_type, block_size=block_size, errors='ignore')
    
    extremes=extremes.to_frame().reset_index()
    
    extremes.rename(columns={"extreme values":"extreme"}, inplace=True)
    
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