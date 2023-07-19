# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 13:40:02 2023

@author: A R Fogg
"""

import pandas as pd
import numpy as np

from copulas.multivariate import GaussianMultivariate

def fit_copula_bivariate(x_extremes, y_extremes, x_name, y_name):
    """
    Function to fit a copula to x and y extremes

    Parameters
    ----------
    x_extremes : np.array
        X extremes on uniform margins.
    y_extremes : np.array
        Y extremes on uniform margins.
    x_name : string
        String descriptor for x values e.g. 'AE'.
    y_name : string
        String descriptor for y values e.g. 'AL'.

    Returns
    -------
    None.

    """
    
    # First, check same number of extremes in x and y
    if np.array(x_extremes).size == np.array(y_extremes).size:
        print('Fitting copula to parsed extremes')
    else:
        print('ERROR: fit_copula_to_extremes.fit_copula_bivariate')
        print('x_extremes and y_extremes must have the same length')
        raise NameError('x_extremes and y_extremes must have the same length')
    
    # Format the extremes to how copulas wants them
    copula_df=pd.DataFrame({x_name:x_extremes,
                            y_name:y_extremes})
    
    # Initialise the copula - testing with GaussianMultivariate
    copula = GaussianMultivariate()
    # Fit copula to the extremes
    copula.fit(copula_df)
    
    return copula

