# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:08:40 2024

@author: A R Fogg

Class definitions
"""

import numpy as np

from scipy.stats import genextreme
from scipy.stats import gumbel_r


class gevd_fitter():
    """
    A class which contains model fitting information based
    on input extrema.
    """
    
    def __init__(self, extremes):
        """
        Initialise the gevd_fitter class. Fits a GEVD or 
        Gumbel distribution.

        Parameters
        ----------
        extremes : np.array or pd.Series
            List of extrema to fit a model to.

        Returns
        -------
        None.

        """
        
        # Fit a model
        self.fit_model(extremes)
    

    def fit_model(self, extremes):
        """
        Fit a GEVD or Gumbel to the parsed
        extrema.

        Parameters
        ----------
        extremes : np.array or pd.Series
            List of extrema to fit a model to.

        Returns
        -------
        None.

        """
        
        # Select best fitting distribution
        self.select_distribution(extremes)
        
        # Fit model
        fitted_params = self.distribution.fit(extremes)
        
        # Freeze the fitted model
        self.frozen_dist = self.distribution(*fitted_params)
        
        # Assign other parameters per model
        if self.distribution_name == 'genextreme':
            self.shape_ = fitted_params[0]
            self.location = fitted_params[1]
            self.scale = fitted_params[2]
            self.formatted_dist_name = 'GEVD'
            
        elif self.distribution_name == 'gumbel_r':
            self.shape_ = 0.
            self.location = fitted_params[0]
            self.scale = fitted_params[1]
            self.formatted_dist_name = 'Gumbel'
            
        # Assign AIC
        self.aic = self.akaike_info_criterion(extremes, self.frozen_dist)
        

    def select_distribution(self, extremes):
        """
        Choose the best fitting distribution
        based on which has the lowest AIC.

        Parameters
        ----------
        extremes : np.array or pd.Series
            List of extrema to fit a model to.

        Returns
        -------
        None.

        """
        
        # Define loop lists
        distributions = [genextreme, gumbel_r]
        names = ['genextreme', 'gumbel_r']
        aic_arr = []

        for dist in distributions:
            # Fit the model    
            params = dist.fit(extremes)
            
            # Freeze distribution
            frozen = dist(*params)
            
            # Calculate AIC
            aic = self.akaike_info_criterion(extremes, frozen)
            aic_arr.append(aic)
        
        # Find model with lowest AIC
        min_index, = np.where(aic_arr == np.min(aic_arr))
        
        # Assign the selected distribution to the class
        self.distribution_name = names[min_index[0]]
        self.distribution = distributions[min_index[0]]
    
    def akaike_info_criterion(self, data, model):
        """
        Calculate AIC.

        Parameters
        ----------
        data : np.array or pd.Series
            Data the model has been fit too, e.g.
            extrema.
        model : scipy rv_frozen
            Frozen scipy rv_continuous distribution
            function.

        Returns
        -------
        aic : float
            The calculated AIC.

        """
        # Define parameters for AIC
        loglikelyhood = np.sum(model.logpdf(data))
        k = len(model.args)
        n = len(data)
        
        # AIC with modification for small sample size
        aic = (2. * k) - (2. * loglikelyhood) + ( ( (2. * k**2) + (2. * k) ) / (n - k - 1) )
         
        return aic
