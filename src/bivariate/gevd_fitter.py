# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:08:40 2024

@author: A R Fogg

GEVD fit class definition.
"""

import numpy as np

from scipy.stats import genextreme
from scipy.stats import gumbel_r

from . import transform_uniform_margins


class gevd_fitter():
    """
    A class which contains model fitting information based
    on input extrema.
    """

    def __init__(self, extremes, dist=None, fit_guess={},
                 shape_threshold=0.005):
        """
        Initialise the gevd_fitter class. Fits a GEVD or
        Gumbel distribution.

        Parameters
        ----------
        extremes : np.array or pd.Series
            List of extrema to fit a model to.
        dist : string, optional
            Distribution to use for fitting. If == None,
            best fitting distribution is chosen using
            select_distribution. Valid options 'genextreme'
            or 'gumbel_r'. The default is None.
        fit_guess : dictionary, optional
            Dictionary containing guess initial parameters
            for fitting the distribution. Keys 'c' for shape,
            'scale', and 'loc' for location. The default is
            {}.
        shape_threshold : float, optional
            A genextreme distribution is fitted. If the absolute value of
            the resulting shape parameter is less than or equal to this value,
            a gumbel_r distribution is returned instead.

        Returns
        -------
        None.

        """

        # Store the extrema
        self.extremes = np.array(extremes)

        # Fit a model
        self.fit_model(dist=dist, fit_guess=fit_guess)

        # Convert extrema to uniform margins empirically
        self.extremes_unif_empirical = transform_uniform_margins.\
            transform_from_data_scale_to_uniform_margins_empirically(
                self.extremes)

        # Convert extrema to uniform margins using CDF
        self.extremes_unif_CDF = transform_uniform_margins.\
            transform_from_data_scale_to_uniform_margins_using_CDF(
                self.extremes, self)

        # Define dictionary of fit parameters
        self.params_dict = {'c': self.shape_,
                            'scale': self.scale,
                            'loc': self.location}

    def fit_model(self, dist=None, fit_guess={}, shape_threshold=0.005):
        """
        Fit a GEVD or Gumbel to the parsed
        extrema.

        Parameters
        ----------
        dist : string, optional
            Distribution to use for fitting. If == None,
            best fitting distribution is chosen using
            select_distribution. Valid options 'genextreme'
            or 'gumbel_r'. The default is None.
        fit_guess : dictionary, optional
            Dictionary containing guess initial parameters
            for fitting the distribution. Keys 'c' for shape,
            'scale', and 'loc' for location. The default is {}.
        shape_threshold : float, optional
            A genextreme distribution is fitted. If the absolute value of
            the resulting shape parameter is less than or equal to this value,
            a gumbel_r distribution is returned instead.

        Returns
        -------
        None.

        """
        if dist is None:
            # Select best fitting distribution
            self.select_distribution()
        else:
            # Use parsed distribution
            self.distribution_name = dist
            self.distribution = genextreme if dist == 'genextreme' \
                else gumbel_r

        # Fit model
        if fit_guess == {}:
            fitted_params = self.distribution.fit(self.extremes)
        else:
            # Make a copy of fit_guess so global distribution isn't changed
            fit_guess = fit_guess.copy()
            args = fit_guess.pop('c')
            # Different inputs for different distributions
            if self.distribution_name == 'genextreme':
                fitted_params = self.distribution.fit(self.extremes, args,
                                                      **fit_guess)
            elif self.distribution_name == 'gumbel_r':
                fitted_params = self.distribution.fit(self.extremes,
                                                      **fit_guess)

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
        self.aic = self.akaike_info_criterion(self.extremes, self.frozen_dist)

    def select_distribution(self, shape_threshold=0.005):
        """
        Choose the best fitting distribution
        based on which has the lowest AIC.

        Parameters
        ----------
        shape_threshold : float, optional
            A genextreme distribution is fitted. If the absolute value of
            the resulting shape parameter is less than or equal to this value,
            a gumbel_r distribution is returned instead.

        Returns
        -------
        None.

        """

        # Fit GEVD, and see what the shape value is
        shape_, location, scale = genextreme.fit(self.extremes)

        # Assess the magnitude of the shape parameter
        if abs(shape_) > shape_threshold:
            # Shape is large enough, genextreme is returned
            self.distribution_name = 'genextreme'
            self.distribution = genextreme
        else:
            # Shape is small, so a Gumbel is likely a better fit
            self.distribution_name = 'gumbel_r'
            self.distribution = gumbel_r

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
        aic = (2. * k) - (2. * loglikelyhood) + \
            (((2. * k**2) + (2. * k)) / (n - k - 1))

        return aic
