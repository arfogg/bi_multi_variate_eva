<h1 align="center">:bar_chart: bi_multi_variate_eva :chart_with_upwards_trend:</h1> 

[![Downloads](https://img.shields.io/github/downloads/arfogg/bi_multi_variate_eva/total.svg)](#)
[![GitHub release](https://img.shields.io/github/v/release/arfogg/bi_multi_variate_eva)](#)
[![GitHub release date](https://img.shields.io/github/release-date/arfogg/bi_multi_variate_eva)](#)
[![GitHub last commit](https://img.shields.io/github/last-commit/arfogg/bi_multi_variate_eva)](#)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13384969.svg)](https://doi.org/10.5281/zenodo.13384969)

[![Stars](https://img.shields.io/github/stars/arfogg/bi_multi_variate_eva?style=social&color=%23FFB31A)](#)
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](https://www.python.org/)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![Issues](https://img.shields.io/github/issues/arfogg/bi_multi_variate_eva.svg)](https://github.com/arfogg/bi_multi_variate_eva/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)

Python package to run bivariate and multivariate extreme value analysis on generic data - work in progress.
Bivariate analysis is described in detail by Fogg et al (2024) currently under review at AGU journal _Space Weather_. Preprint available:

Fogg, A. R., D. Healy, C. M. Jackman, et al (2024). _Bivariate Extreme Value Analysis for Space Weather Risk Assessment: solar wind - magnetosphere driving in the terrestrial system._ ESS Open Archive. doi: [10.22541/essoar.172612544.43585872/v1](https://doi.org/10.22541/essoar.172612544.43585872/v1)

**License:** CC0-1.0

**Support:** please [create an issue](https://github.com/arfogg/bi_multi_variate_eva/issues) or contact [arfogg](https://github.com/arfogg) directly. Any input on the code / issues found are greatly appreciated and will help to improve the software.

## Table of Contents
- [Required Packages](#required-packages)
- [Installing the code](#installing-the-code)
- [Bivariate Analysis](#bivariate-analysis)     
   * [(1) Getting your data ready](#1-getting-your-data-ready) 
   * [(2) Checking for Asymptotic Dependence](#2-checking-for-asymptotic-dependence) 
   * [(3) Extract extrema](#3-extract-extrema) 
   * [(4) Fit a model to the extrema](#4-fit-a-model-to-the-extrema) 
   * [(5) Transform extrema data to uniform margins](#5-transform-extrema-data-to-uniform-margins) 
   * [(6) Bootstrapping the extrema](#6-bootstrapping-the-extrema)
   * [(7) Fit a copula to both sets of extrema](#7-fit-a-copula-to-both-sets-of-extrema) 
   * [(8) Take a sample from the copula](#8-take-a-sample-from-the-copula) 
   * [(9) Plot diagnostic to assess copula fit](#9-plot-diagnostic-to-assess-copula-fit) 
   * [(10) Plot return period as a function of two variables](#10-plot-return-period-as-a-function-of-two-variables) 
- [Multivariate Analysis](#multivariate-analysis)
- [Acknowledgements](#acknowledgements)

## Required Packages

scipy, numpy, matplotlib, pandas, copulas

See [environment.yml](environment.yml) for details.

Install copulas using pip or see documentation [here](https://pypi.org/project/copulas/).

## Using the code

Download the code:
```python
git clone https://github.com/arfogg/bi_multi_variate_eva
```

Importing:
```python
from bi_multi_variate_eva import *
```

## Bivariate Analysis

An example walkthrough of running the Bivariate Extreme Value Analysis on two variables x and y. For theory, a recommended text is: Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values. Springer.

#### (1) Getting your data ready

Make sure there are no datagaps in your timeseries. You can either remove the rows, or interpolate. This requires an data-expert user decision so is **not** included in this package. You must do this before using the package. For some functions, the data is parsed as a pandas.DataFrame.

#### (2) Checking for Asymptotic Dependence

The function `plot_extremal_dependence_coefficient` within `determine_AD_AI` creates a diagnostic plot to examine asymptotic dependence/independence.

For example:

```python
fig, ax_data, ax_data_unif, ax_edc, chi, chi_lower_q, chi_upper_q = \
    determine_AD_AI.plot_extremal_dependence_coefficient(x, y, x_bs_um, y_bs_um, n_bootstrap,
                                                         "X", "Y", "(units)", "(units)")
```

Timeseries (`np.array` or `pd.Series`) of x and y are parsed, with their bootstraps (transformed to uniform margins), number of bootstraps and strings for plot labels.

#### (3) Extract extrema

Extract extremes for both X and Y using `detect_extremes.find_joint_block_maxima`. Analysis on points above threshold maxima yet to be implemented.

For example:
```python
empty_blocks, x_extreme_t, x_extreme, y_extreme_t, y_extreme = \
            detect_extremes.find_joint_block_maxima(data_df, 'x', 'y')

x_extremes_df = pd.DataFrame({'datetime':x_extreme_t, 'extreme':x_extreme})
y_extremes_df = pd.DataFrame({'datetime':y_extreme_t, 'extreme':y_extreme})    
```

A dataframe of evenly sampled x and y are parsed, with their respective dataframe column names. These are transformed to individual parameter DataFrames.


#### (4) Fit a model to the extrema

Fit a GEVD or Gumbel distribution to both sets of extrema (i.e. for x and y) using `gevd_fitter` class.

For example:
```python
x_gevd_fit = gevd_fitter(x_extremes_df.extreme)
y_gevd_fit = gevd_fitter(y_extremes_df.extreme)
```

By initialising the `gevd_fitter` class, a GEVD or Gumbel model is fit to the extrema. Fitting information is stored in the object.

#### (5) Transform extrema data to uniform margins

Transform x and y extrema from data scale (as it looks on the instrument) to uniform margins. This happens within the `gevd_fitter` class.

You can plot a diagnostic about the transformation of one of the variables using `transform_uniform_margins.plot_diagnostic`.

#### (6) Bootstrapping the extrema

To facilitate error calculation, we bootstrap the extrema. For each of the N bootstraps, a random selection of indices from between 0 to n_extrema-1 is chosen (where n_extrema is the number of extrema in each dataset). This set of indices is used to select points from both x and y. This ensures joint selection, so we retain the physical link between x and y.

For example:
```python
x_bootstrap = np.full((n_extrema, N), np.nan)
y_bootstrap = np.full((n_extrema, N), np.nan)
        
for i in range(N):
    # Select indices to get bootstraps from
    ind = np.random.choice(np.linspace(0, n_extrema-1, n_extrema), n_extrema)
    x_bootstrap[:, i] = x_extremes_df.extreme.iloc[ind]
    y_bootstrap[:, i] = y_extremes_df.extreme.iloc[ind]
```

By then using a `bootstrap_gevd_fit` object, GEVD or Gumbel fits are estimated for each bootstrap.

```python
x_bs_gevd_fit = bootstrap_gevd_fit(x_bootstrap, x_gevd_fit)
y_bs_gevd_fit = bootstrap_gevd_fit(y_bootstrap, y_gevd_fit)
```

#### (7) Fit a copula to both sets of extrema

Fit a copula to x and y extrema using `fit_copula_to_extremes.fit_copula_bivariate`.

For example:
```python
copula = fit_copula_to_extremes.fit_copula_bivariate(x_extremes_unif, y_extremes_unif, 'X', 'Y')
```

#### (8) Take a sample from the copula

Using your copula from (6), extract a sample, e.g.: `copula_sample = copula.sample(100)`.

Transform that sample back to data scale:
```python
x_sample = transform_uniform_margins.\
                transform_from_uniform_margins_to_data_scale(copula_sample[:, 0], x_gevd_fit)
y_sample = transform_uniform_margins.\
                transform_from_uniform_margins_to_data_scale(copula_sample[:, 0], y_gevd_fit)
```

#### (9) Plot diagnostic to assess copula fit

To plot histograms of the copula in data scale (with GEVD/Gumbel fitted to observed extrema overplotted) and on uniform margins, use `transform_uniform_margins.plot_copula_diagnostic`. 

For example:
```python
fig_copula_1d, ax_copula_1d = transform_uniform_margins.\
                                    plot_copula_diagnostic(copula_sample[:, 0], copula_sample[:, 1],
                                                           x_sample, y_sample, x_gevd_fit, y_gevd_fit,
                                                           'X', 'Y')
```

Alternatively, to compare the 2D distributions of the observed extrema and copula sample, use `fit_copula_to_extremes.qualitative_copula_fit_check_bivariate`.

For example:
```python
fig_copula_2d, ax_copula_2d = fit_copula_to_extremes.\
                    qualitative_copula_fit_check_bivariate(x_extremes_df.extreme, y_extremes_df.extreme,
                                                           x_sample, y_sample, 'X', 'Y')
```

#### (10) Plot return period as a function of two variables

To plot the return period as a function of x and y, with standard contours.

For example:
```python
fig_rp, ax_rp = calculate_return_periods_values.\
                        plot_return_period_as_function_x_y(copula,
                                                           np.nanmin(x_extremes_df.extreme),
                                                           np.nanmax(x_extremes_df.extreme),
                                                           np.nanmin(y_extremes_df.extreme),
                                                           np.nanmax(y_extremes_df.extreme),
                                                           'X', 'Y', 'X (units)', 'Y (units)',
                                                           bs_copula_arr, N)
```

Where bs_copula_arr is a list of copulae fit to each bootstrap, which is used to calculate confidence intervals.

## Multivariate Analysis

To be completed

## Acknowledgements

[ARF](https://github.com/arfogg) gratefully acknowledges the support of [Irish Research Council](https://research.ie/) Government of Ireland Postdoctoral Fellowship GOIPD/2022/782.

[CMJ](https://github.com/caitrionajackman), [MJR](https://github.com/mjrutala), [SCM](https://github.com/SeanMcEntee), and [SJW](https://github.com/08walkersj) were supported by [Science Foundation Ireland](https://www.sfi.ie/) award 18/FRL/6199.
