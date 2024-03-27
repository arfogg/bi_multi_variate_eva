# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:38:34 2024

@author: A R Fogg
"""

import numpy as np

import transform_uniform_margins

def produce_bootstrap_sample(data, extract_length):
    """
    Function to product a bootstrapping sample from a given 
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
    # data must be np.array, df col doesn't work
    
    bootstrap_sample = []
    
    # While our output is shorter than the input data
    #   we will continue adding data
    while len(bootstrap_sample) < len(data):
        
        # Choose a random start and end point from the
        #   input data to resample
        start_point = int( np.random.choice(data.size, size=1)[0] )
        end_point = int( start_point + np.random.geometric(1.0 / extract_length, size=1)[0] )
        
        # If not beyond the end of the data, append
        #   some data to the new array
        if end_point < len(data):
            bootstrap_sample=np.append(bootstrap_sample,data[start_point:end_point])
            
    # Check we aren't longer that the original sample 
    bootstrap_sample = np.array(bootstrap_sample[0:len(data)])
    
    return bootstrap_sample

def iterative_bootstrap(data, extract_length, n_iterations=100):
    """
    Create a matrix of many bootstraps of the same length.

    Parameters
    ----------
    data : np.array
        The input data to be bootstrapped. Must be np.array,
        pd.Series won't work.
    extract_length : int
        On average, the data samples appended to the bootstrap
        will be extract_length number of data points long.
    n_iterations : int, optional
        Number of iterations to run. The default is 100.

    Returns
    -------
    bootstrap_sample : np.array
        Bootstrap sample of shape data.size x n_iterations.

    """
    print('Producing a bootstrapped sample - may be slow')
    print('Data length: ', data.size)
    print('Number of iterations requested: ', n_iterations)
    
    bootstrap_sample=np.full((data.size,n_iterations), np.nan)
    
    for i in range(n_iterations):
        bootstrap_sample[:,i]=produce_bootstrap_sample(data, extract_length)
        
    return bootstrap_sample

def iterative_bootstrap_um(data, extract_length, n_iterations=100):
    """
    Create a matrix of many bootstraps of the same length, 
    and return data in both data scale and on uniform margins.

    Parameters
    ----------
    data : np.array
        The input data to be bootstrapped. Must be np.array,
        pd.Series won't work.
    extract_length : int
        On average, the data samples appended to the bootstrap
        will be extract_length number of data points long.
    n_iterations : int, optional
        Number of iterations to run. The default is 100.

    Returns
    -------
    bootstrap_sample_ds : np.array
        Bootstrap sample of shape data.size x n_iterations in
        data scale.
    bootstrap_sample_um : np.array
        Bootstrap sample of shape data.size x n_iterations on
        uniform margins.
    """

    print('Producing a bootstrapped sample in data scale and on uniform margins - may be slow')
    print('Data length: ', data.size)
    print('Number of iterations requested: ', n_iterations)
    
    bootstrap_sample_ds=np.full((data.size,n_iterations), np.nan)
    bootstrap_sample_um=np.full((data.size,n_iterations), np.nan)
    
    for i in range(n_iterations):
        
        bootstrap_sample_ds[:,i]=produce_bootstrap_sample(data, extract_length)
        bootstrap_sample_um[:,i]=transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(bootstrap_sample_ds[:,i], plot=False)
        
    return bootstrap_sample_ds, bootstrap_sample_um

    