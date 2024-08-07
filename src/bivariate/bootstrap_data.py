# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:38:34 2024

@author: A R Fogg
"""

import numpy as np
import pandas as pd

from . import transform_uniform_margins


def produce_single_bootstrap_sample(data, extract_length):
    """
    Function to product a bootstrapping sample from one given
    data set.

    Based on code by Dr Dáire Healy here -
    https://github.com/arfogg/bi_multi_variate_eva/blob/c8f7911b5aa911a074a33b862224563460357ce3/r_code/bootstrapped_chi.R

    Parameters
    ----------
    data : np.array
        The input data to be bootstrapped. Must be np.array,
        pd.Series won't work.
    extract_length : int
        On average, the data samples appended to the bootstrap
        will be extract_length number of data points long.

    Returns
    -------
    bootstrap_sample : np.array
        A sample of length data.size of bootstrapped data.

    """

    # Initialise empty list
    bootstrap_sample = []

    # While our output is shorter than the input data
    #   we will continue adding data
    while len(bootstrap_sample) < len(data):

        # Choose a random start and end point from the
        #   input data to resample
        start_point = int(np.random.choice(data.size, size=1)[0])
        end_point = int(start_point + np.random.geometric(1.0 / extract_length,
                                                          size=1)[0])

        # If not beyond the end of the data, append
        #   some data to the new array
        if end_point < len(data):
            bootstrap_sample = np.append(bootstrap_sample,
                                         data[start_point:end_point])

    # Ensure output sample isn't longer that the original sample
    bootstrap_sample = np.array(bootstrap_sample[0:len(data)])

    return bootstrap_sample


def produce_dual_bootstrap_sample(data_x, data_y, extract_length):
    """
    Function to product a bootstrapping sample from a given
    two parameter data set.

    Based on code by Dr Dáire Healy here -
    https://github.com/arfogg/bi_multi_variate_eva/blob/c8f7911b5aa911a074a33b862224563460357ce3/r_code/bootstrapped_chi.R

    Parameters
    ----------
    data_x : np.array
        The input x data to be bootstrapped. Must be np.array,
        pd.Series won't work.
    data_y : np.array
        The input y data to be bootstrapped. Must be np.array,
        pd.Series won't work.
    extract_length : int
        On average, the data samples appended to the bootstrap
        will be extract_length number of data points long.

    Returns
    -------
    bootstrap_sample_x : np.array
        A sample of length data.size of bootstrapped x data.
    bootstrap_sample_y : np.array
        A sample of length data.size of bootstrapped y data.
    """

    # Initialise empty bootstrap lists
    bootstrap_sample_x = []
    bootstrap_sample_y = []

    # While our output is shorter than the input data
    #   we will continue adding data
    while len(bootstrap_sample_x) < len(data_x):

        # Choose a random start and end point from the
        #   input data to resample
        start_point = int(np.random.choice(data_x.size, size=1)[0])
        end_point = int(start_point + np.random.geometric(1.0 / extract_length,
                                                          size=1)[0])

        # If not beyond the end of the data, append
        #   some data to the new array
        if end_point < len(data_x):
            bootstrap_sample_x = np.append(bootstrap_sample_x,
                                           data_x[start_point:end_point])
            bootstrap_sample_y = np.append(bootstrap_sample_y,
                                           data_y[start_point:end_point])

    # Ensure output sample isn't longer that the original sample
    bootstrap_sample_x = np.array(bootstrap_sample_x[0:len(data_x)])
    bootstrap_sample_y = np.array(bootstrap_sample_y[0:len(data_y)])

    return bootstrap_sample_x, bootstrap_sample_y


def iterative_dual_bootstrap(data_x, data_y, extract_length, n_iterations=100):
    """
    Create a matrix of many bootstraps of the same length, from
    a pair of parameters.

    Parameters
    ----------
    data_x : np.array
        The input x data to be bootstrapped. Must be np.array,
        pd.Series won't work.
    data_y : np.array
        The input y data to be bootstrapped. Must be np.array,
        pd.Series won't work.
    extract_length : int
        On average, the data samples appended to the bootstrap
        will be extract_length number of data points long.
    n_iterations : int, optional
        Number of iterations to run. The default is 100.

    Returns
    -------
    bootstrap_sample_x : np.array
        Bootstrap sample of shape data_x.size x n_iterations.
    bootstrap_sample_y : np.array
        Bootstrap sample of shape data_y.size x n_iterations.
    """

    print('Producing a bootstrapped sample - may be slow')
    print('Data length: ', data_x.size)
    print('Number of iterations requested: ', n_iterations)

    # Initialise empty bootstrap matrix
    bootstrap_sample_x = np.full((data_x.size, n_iterations), np.nan)
    bootstrap_sample_y = np.full((data_y.size, n_iterations), np.nan)

    # Loop through number of parsed iterations
    for i in range(n_iterations):
        # Produce bootstrap
        bootstrap_sample_x[:, i], bootstrap_sample_y[:, i] \
            = produce_dual_bootstrap_sample(data_x, data_y, extract_length)

    return bootstrap_sample_x, bootstrap_sample_y


def iterative_dual_bootstrap_um(data_x, data_y, extract_length,
                                n_iterations=100):
    """
    Create a matrix of many bootstraps of the same length,
    and return data in both data scale and on uniform margins.

    Parameters
    ----------
    data_x : np.array
        The input x data to be bootstrapped. Must be np.array,
        pd.Series won't work.
    data_y : np.array
        The input y data to be bootstrapped. Must be np.array,
        pd.Series won't work.
    extract_length : int
        On average, the data samples appended to the bootstrap
        will be extract_length number of data points long.
    n_iterations : int, optional
        Number of iterations to run. The default is 100.

    Returns
    -------
    bootstrap_sample_x_ds : np.array
        Bootstrap sample of x, of shape data_x.size x
        n_iterations in data scale.
    bootstrap_sample_x_um : np.array
        Bootstrap sample of x, of shape data_x.size x
        n_iterations on uniform margins.
    bootstrap_sample_y_ds : np.array
        Bootstrap sample of y, of shape data_y.size x
        n_iterations in data scale.
    bootstrap_sample_y_um : np.array
        Bootstrap sample of y, of shape data_y.size
        n_iterations on uniform margins.

    """

    print('Producing a bootstrapped sample in data scale and on uniform')
    print('margins - may be slow')
    print('Data length: ', data_x.size)
    print('Number of iterations requested: ', n_iterations)

    # Initialise empty bootstrap matrices for x and y
    bootstrap_sample_x_ds = np.full((data_x.size, n_iterations), np.nan)
    bootstrap_sample_x_um = np.full((data_x.size, n_iterations), np.nan)
    bootstrap_sample_y_ds = np.full((data_y.size, n_iterations), np.nan)
    bootstrap_sample_y_um = np.full((data_y.size, n_iterations), np.nan)

    # Loop through the number of parsed iterations
    for i in range(n_iterations):

        # Produce dual bootstrap sample
        bootstrap_sample_x_ds[:, i], bootstrap_sample_y_ds[:, i] = \
            produce_dual_bootstrap_sample(data_x, data_y, extract_length)

        # Transform data to uniform margins
        bootstrap_sample_x_um[:, i] = transform_uniform_margins. \
            transform_from_data_scale_to_uniform_margins_empirically(
                                    bootstrap_sample_x_ds[:, i], plot=False)
        bootstrap_sample_y_um[:, i] = transform_uniform_margins. \
            transform_from_data_scale_to_uniform_margins_empirically(
                                    bootstrap_sample_y_ds[:, i], plot=False)

    return bootstrap_sample_x_ds, bootstrap_sample_x_um, \
        bootstrap_sample_y_ds, bootstrap_sample_y_um,


def produce_n_bootstrap_sample(df, cols, extract_length):
    """
    Function to product a bootstrapping sample from a given
    n parameter data set.

    Based on code by Dr Dáire Healy here -
    https://github.com/arfogg/bi_multi_variate_eva/blob/c8f7911b5aa911a074a33b862224563460357ce3/r_code/bootstrapped_chi.R

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing columns defined by cols.
    cols : list of strings
        Names of the columns to be bootstrapped.
    extract_length : int
        On average, the data samples appended to the bootstrap
        will be extract_length number of data points long.

    Returns
    -------
    bootstrap_df : pd.DataFrame
        Dataframe with columns defined by cols, with
        bootstrapped data.

    """

    # Initialise bootstrap DataFrame
    bootstrap_df = pd.DataFrame(columns=cols)

    # While our output is shorter than the input data
    #   we will continue adding data
    while len(bootstrap_df) < len(df):

        # Choose a random start and end point from the
        #   input data to resample
        start_point = int(np.random.choice(len(df), size=1)[0])
        end_point = int(start_point + np.random.geometric(1.0 / extract_length,
                                                          size=1)[0])

        # If not beyond the end of the data, append
        #   some data to the new array
        if end_point < len(df):

            loop_df = pd.DataFrame({})

            # Loop through the columns to be bootstrapped
            for col in cols:
                loop_df[col] = df[col].iloc[start_point:end_point]

            bootstrap_df = pd.concat([bootstrap_df, loop_df],
                                     ignore_index=True)

    # Ensure output sample isn't longer that the original sample
    bootstrap_df = bootstrap_df.iloc[:len(df)].copy(deep=True)

    return bootstrap_df


def produce_n_bootstrap_sample_um(df, cols, extract_length):
    """
    Function to product a bootstrapping sample from a given
    two parameter data set.

    Based on code by Dr Dáire Healy here -
    https://github.com/arfogg/bi_multi_variate_eva/blob/c8f7911b5aa911a074a33b862224563460357ce3/r_code/bootstrapped_chi.R

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing columns defined by cols.
    cols : list of strings
        Names of the columns to be bootstrapped.
    extract_length : int
        On average, the data samples appended to the bootstrap
        will be extract_length number of data points long.

    Returns
    -------
    bootstrap_df : pd.DataFrame
        Dataframe with columns defined by cols, with
        bootstrapped data in data scale.
    bootstrap_df_um : pd.DataFrame
        Dataframe with columns defined by cols, with
        bootstrapped data on uniform margins.

    """

    # Initialise bootstrap DataFrame
    bootstrap_df = pd.DataFrame(columns=cols)

    # While our output is shorter than the input data
    #   we will continue adding data
    while len(bootstrap_df) < len(df):

        # Choose a random start and end point from the
        #   input data to resample
        start_point = int(np.random.choice(len(df), size=1)[0])
        end_point = int(start_point + np.random.geometric(1.0 / extract_length,
                                                          size=1)[0])

        # If not beyond the end of the data, append
        #   some data to the new array
        if end_point < len(df):

            loop_df = pd.DataFrame({})

            # Loop through the columns to be bootstrapped
            for col in cols:
                loop_df[col] = df[col].iloc[start_point:end_point]

            bootstrap_df = pd.concat([bootstrap_df, loop_df],
                                     ignore_index=True)

    # Ensure output sample isn't longer that the original sample
    bootstrap_df = bootstrap_df.iloc[:len(df)].copy(deep=True)

    # Transform to uniform margins
    bootstrap_df_um = pd.DataFrame({})
    for col in cols:
        bootstrap_df_um[col] = transform_uniform_margins. \
            transform_from_data_scale_to_uniform_margins_empirically(
                                            bootstrap_df[col], plot=False)

    return bootstrap_df, bootstrap_df_um
