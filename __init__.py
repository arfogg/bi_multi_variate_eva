# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:36:17 2024

@author: A R Fogg

Initialisation file which imports code.
"""

# Import package code
from .src.bivariate import bootstrap_data
from .src.bivariate import calculate_return_periods_values
from .src.bivariate import detect_extremes
from .src.bivariate import determine_AD_AI
from .src.bivariate import fit_copula_to_extremes
from .src.bivariate import plotting_utils
from .src.bivariate import qq_plot
from .src.bivariate import return_period_plot_1d
from .src.bivariate import transform_uniform_margins

# Import classes
from .src.bivariate.gevd_fitter import gevd_fitter
from .src.bivariate.bootstrap_gevd_fit import bootstrap_gevd_fit