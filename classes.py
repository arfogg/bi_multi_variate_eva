# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:08:40 2024

@author: A R Fogg

Class definitions
"""

import numpy as np

from scipy.stats import genextreme
from scipy.stats import gumbel_r
from scipy.stats import ttest_ind

import fit_model_to_extremes



class gevd_fitter():
    
    # Do I need an __init__ ?
    # def __init__(self):
         
    def __init__(self, extremes):
        
        self.fit_model(extremes)
    
    # Fit model to data
    def fit_model(self, extremes): #, extremes_method, extremes_type,
                           #df_data_tag, df_time_tag='datetime',
                           #fitting_type = 'Emcee', block_size = None):

        # will eventually recode fitting here

        # fit_params = fit_model_to_extremes.fit_gevd_or_gumbel(extremes_df, extremes_method, extremes_type,
        #                        df_data_tag, df_time_tag=df_time_tag,
        #                        fitting_type=fitting_type, block_size=block_size)
                               

        # self.shape_lower_ci_width = float(fit_params.shape_lower_ci_width)
        # self.shape_upper_ci_width = float(fit_params.shape_upper_ci_width)
        # self.location_lower_ci_width = float(fit_params.location_lower_ci_width)
        # self.location_upper_ci_width = float(fit_params.location_upper_ci_width)
        # self.scale_lower_ci_width = float(fit_params.scale_lower_ci_width)
        # self.scale_upper_ci_width = float(fit_params.scale_upper_ci_width)
        
        
        # Select best fitting distribution
        self.select_distribution(extremes)
        
        # Fit model
        fitted_params = self.distribution.fit(extremes)
        
        self.frozen_dist = self.distribution(*fitted_params)
        
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
            
        self.aic = self.akaike_info_criterion(extremes, self.frozen_dist)
        
        
        
        
        # if blah = gumbel then shape = 0
        
        
    def select_distribution(self, extremes):
        
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
        
        min_index, = np.where(aic_arr == np.min(aic_arr))

        self.distribution_name = names[min_index[0]]
        self.distribution = distributions[min_index[0]]
        
       
    
    
    
    def akaike_info_criterion(self, data, model):
        
        loglikelyhood = np.sum(model.logpdf(data))

        k = len(model.args)
        n = len(data)
        # AIC with modification for small sample size
        aic = (2. * k) - (2. * loglikelyhood) + ( ( (2. * k**2) + (2. * k) ) / (n - k - 1) )
         
        return aic
    
    
    
    
    
    
    
    
    
        # work out which fits better
        
        # return all the fitted params
        
    
    # Want to be able to do the fitting (for now just use the function)
    #   but then for everything to magically be stored in the class somehow
    
    
    
    # Be able to return all these things
    # ------
    # shape_
    # location
    # scale
    # shape_lower_ci_width
    # shape_upper_ci_width
    # location_lower_ci_width
    # location_upper_ci_width
    # scale_lower_ci_width
    # scale_upper_ci_width
    # distribution_name
    # formatted_dist_name