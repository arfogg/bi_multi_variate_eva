# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:09:18 2024

@author: A R Fogg

Bootstrap GEVD fit class definition
"""

from . import return_period_plot_1d
from .gevd_fitter import gevd_fitter

class bootstrap_gevd_fit():
    """
    A class which contains bootstrapped data and their fits.
    """
    def __init__(self, bootstrap_extrema, true_fit):
        """
        Initialise class

        Parameters
        ----------
        bootstrap_extrema : np.array
            Bootstrapped extrema. Of shape n_extrema x 
            n_bootstraps.

        Returns
        -------
        None.

        """
        # Store the bootstrapped data
        self.bs_data = bootstrap_extrema
        
        # Store the number of extrema and bootstraps
        self.n_extrema = bootstrap_extrema.shape[0]
        self.n_ci_iterations = bootstrap_extrema.shape[1]
        
        # Store the fit to the true data
        self.true_fit = true_fit
        
        self.fit_gevd()
        
    def fit_gevd(self):
        
        gevd_fitter_arr = []
        for i in range(self.n_ci_iterations):
            gevd_fitter_arr.append( gevd_fitter(self.bs_data[:,i], dist = self.true_fit.distribution_name) )
        
        self.gevd_fitter_arr = gevd_fitter_arr
        
    def calc_return_levels(self, distribution_name, block_size, bs_return_periods, true_fit):
        """
        Calculate the return levels for these bootstrapped extrema.

        Parameters
        ----------
        distribution_name : string
            Which distribution the true data are fit to. Either
            'genextreme' or 'gumbel_r'.
        block_size : pd.Timedelta
            Block size for extrema detection.
        bs_return_periods : np.array
            Return periods (in years) to calculate the levels at.
        true_fit : gevd_fitter class
            Gevd_fitter object for the fit to true extrema.

        Returns
        -------
        None.

        """
        
        # Store the parsed information
        self.distribution_name = distribution_name
        self.block_size = block_size
        self.periods = bs_return_periods
        
        # Calculate the return levels
        levels, shape_, location, scale = return_period_plot_1d.return_level_bootstrapped_data(self, 
                                                                                               self.n_ci_iterations, 
                                                                                               bs_return_periods)
        
        # levels, shape_, location, scale = return_period_plot_1d.return_level_bootstrapped_data(self.bs_data, 
        #                                 self.n_ci_iterations, self.distribution_name,
        #                                 self.block_size, bs_return_periods, true_fit)

        # Store the return levels and associated fit information
        self.levels = levels
        self.shape_ = shape_
        self.location = location
        self.scale = scale