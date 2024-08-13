# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:09:18 2024

@author: A R Fogg

Bootstrap GEVD fit class definition.
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
        true_fit : gevd_fitter class
            Object containing GEVD fitting information for the true data.
            Contains attributes listed below. See gevd_fitter.py for
            definition.
            extremes : np.array
                List of extrema the model is fit to.
            extremes_unif_empirical : np.array
                Parsed extrema converted to uniform margins empirically.
            extremes_unif_CDF : np.array
                Parsed extrema converted to uniform margins using the fitted
                GEVD or Gumbel distribution.
            distribution_name : string
                String name of fitted distribution. Either 'genextreme' or
                'gumbel_r'.
            distribution : scipy.rv_continuous
                Empty / generic distribution of selected distribution.
            frozen_dist : frozen scipy.rv_continuous
                Frozen version of fitted distribution.
            shape_ : float
                Fitted shape parameter.
            location : float
                Fitted location parameter.
            scale : float
                Fitted scale parameter.
            formatted_dist_name : string
                A formatted version of distribution name for plot labels.
            aic : float
                Akaike Information Criterion for fit.
            fit_guess : dictionary
                Dictionary containing guess initial parameters
                for fitting the distribution. Keys 'c' for shape,
                'scale', and 'loc' for location.
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

        # Fit distributions to each bootstrap
        self.fit_gevd()

    def fit_gevd(self):
        """
        Fit a GEVD model to each individual bootstrapped
        dataset, and store an array of gevd_fitter classes.

        Returns
        -------
        None.

        """

        # Initialise empty list
        gevd_fitter_arr = []

        # Looping through individual bootstraps
        for i in range(self.n_ci_iterations):
            # Create and append a gevd_fitter for each bootstrap
            gevd_fitter_arr.append(gevd_fitter(
                self.bs_data[:, i],
                dist=self.true_fit.distribution_name,
                fit_guess=self.true_fit.params_dict))

        # Assign the list of gevd_fitters to this class
        self.gevd_fitter_arr = gevd_fitter_arr

    def calc_return_levels(self, distribution_name, block_size,
                           bs_return_periods, true_fit):
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
            Object containing GEVD fitting information for the true data.
            Contains attributes listed below. See gevd_fitter.py for
            definition.
            extremes : np.array
                List of extrema the model is fit to.
            extremes_unif_empirical : np.array
                Parsed extrema converted to uniform margins empirically.
            extremes_unif_CDF : np.array
                Parsed extrema converted to uniform margins using the fitted
                GEVD or Gumbel distribution.
            distribution_name : string
                String name of fitted distribution. Either 'genextreme' or
                'gumbel_r'.
            distribution : scipy.rv_continuous
                Empty / generic distribution of selected distribution.
            frozen_dist : frozen scipy.rv_continuous
                Frozen version of fitted distribution.
            shape_ : float
                Fitted shape parameter.
            location : float
                Fitted location parameter.
            scale : float
                Fitted scale parameter.
            formatted_dist_name : string
                A formatted version of distribution name for plot labels.
            aic : float
                Akaike Information Criterion for fit.
            fit_guess : dictionary
                Dictionary containing guess initial parameters
                for fitting the distribution. Keys 'c' for shape,
                'scale', and 'loc' for location.
        Returns
        -------
        None.

        """

        # Store the parsed information
        self.distribution_name = distribution_name
        self.block_size = block_size
        self.periods = bs_return_periods

        # Calculate the return levels
        levels, shape_, location, scale = \
            return_period_plot_1d.\
            return_level_bootstrapped_data(self, self.n_ci_iterations,
                                           bs_return_periods)

        # Store the return levels and associated fit information
        self.levels = levels
        self.shape_ = shape_
        self.location = location
        self.scale = scale
