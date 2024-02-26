# bi_multi_variate_eva

Code to run bivariate and multivariate extreme value analysis on generic data - work in progress.

**License:** CC0-1.0

**Support:** please [create an issue](https://github.com/arfogg/bi_multi_variate_eva/issues) or contact [arfogg](https://github.com/arfogg) directly. Any input on the code / issues found are greatly appreciated and will help to improve the software.

## Table of Contents
- [Required Packages](#required-packages)
- [Installing the code](#installing-the-code)
- [Bivariate Analysis](#bivariate-analysis)     
   * [(1) Getting your data ready](#(1)-getting-your-data-ready) 
   * [(2) Checking for Asymptotic Dependence](#(2)-checking-for-asymptotic-dependence) 
   * [(3) Extract extrema](#(3)-extract-extrema) 
   * [(4) Fit a model to the extrema](#(4)-fit-a-model-to-the-extrema) 
   * [(5) Transform extrema data to uniform margins](#(5)-transform-extrema-data-to-uniform-margins) 
   * [(6) Fit a copula to both sets of extrema](#(6)-fit-a-copula-to-both-sets-of-extrema) 
   * [(7) Take a sample from the copula](#(7)-take-a-sample-from-the-copula) 
   * [(8) Plot diagnostic to assess copula fit](#(8)-plot-diagnostic-to-assess-copula-fit) 
   * [(9) Plot return period as a function of two variables](#(9)-plot-return-period-as-a-function-of-two-variables) 
- [Multivariate Analysis](#multivariate-analysis)
- [Acknowledgements](#acknowledgements)

## Required Packages

scipy, numpy, matplotlib, pandas, seaborn, pyextremes, copulas

[Install pyextremes following instructions from its github here](https://github.com/georgebv/pyextremes)

[Install copulas using pip or see documentation here](https://pypi.org/project/copulas/)


## Installing the code

First, the code must be downloaded using `git clone https://github.com/arfogg/bi_multi_variate_eva`

## Bivariate Analysis

#### (1) Getting your data ready

Make sure there are no datagaps in your timeseries. You can either remove the rows, or interpolate. This requires an data-expert user decision so is **not** included in this package. You must do this before using the package. For some functions, the data is parsed as a pandas.DataFrame.

#### (2) Checking for Asymptotic Dependence

The function `plot_extremal_dependence_coefficient` within `determine_AD_AI` creates a diagnostic plot to examine asymptotic dependence/independence.

For example:
```python
determine_AD_AI.plot_extremal_dependence_coefficient(x, y, "X", "Y", "(units)", "(units)")
```

#### (3) Extract extrema

Extract extremes for both X and Y using `detect_extremes.find_block_maxima`. Analysis on points above threshold maxima yet to be implemented.

For example:
```python
x_extremes_df=detect_extremes.find_block_maxima(df,'x',df_time_tag='datetime',block_size=block_size,extremes_type='high')
y_extremes_df=detect_extremes.find_block_maxima(df,'y',df_time_tag='datetime',block_size=block_size,extremes_type='high')
```

#### (4) Fit a model to the extrema

Fit a GEVD or Gumbel distribution to both sets of extrema (i.e. for x and y) using `fit_model_to_extremes.fit_gevd_or_gumbel`.

For example:
```python
x_gevd_fit_params=fit_model_to_extremes.fit_gevd_or_gumbel(x_extremes_df, 'BM', 'high','extreme',df_time_tag='datetime',fitting_type='Emcee', block_size=block_size)
y_gevd_fit_params=fit_model_to_extremes.fit_gevd_or_gumbel(y_extremes_df, 'BM', 'high','extreme',df_time_tag='datetime',fitting_type='Emcee', block_size=block_size)
```

#### (5) Transform extrema data to uniform margins

Transform x and y extrema from data scale (as it looks on the instrument) to uniform margins empirically using `transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically` or using the cumulative distribution function with `transform_uniform_margins.transform_from_data_scale_to_uniform_margins_using_CDF`.

For example:
```python
# Empirically
x_extremes_unif_empirical=transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(x_extremes_df.extreme)
y_extremes_unif_empirical=transform_uniform_margins.transform_from_data_scale_to_uniform_margins_empirically(y_extremes_df.extreme)
# Using cumulative distribution function
x_extremes_unif=transform_uniform_margins.transform_from_data_scale_to_uniform_margins_using_CDF(x_extremes_df.extreme, x_gevd_fit_params,distribution=x_gevd_fit_params.distribution_name[0])
y_extremes_unif=transform_uniform_margins.transform_from_data_scale_to_uniform_margins_using_CDF(y_extremes_df.extreme, y_gevd_fit_params,distribution=y_gevd_fit_params.distribution_name[0])
```

You can plot a diagnostic about the transformation of one of the variables using `transform_uniform_margins.plot_diagnostic`:
```python
fig_um_x,ax_um_x=transform_uniform_margins.plot_diagnostic(x_extremes_df.extreme, x_extremes_unif_empirical, x_extremes_unif, x_gevd_fit_params, 'X')
```

#### (6) Fit a copula to both sets of extrema

Fit a copula to x and y extrema using `fit_copula_to_extremes.fit_copula_bivariate`.

For example:
```python
copula=fit_copula_to_extremes.fit_copula_bivariate(x_extremes_unif, y_extremes_unif, 'X', 'Y')
```
 
#### (7) Take a sample from the copula

Using your copula from (6), extract a sample, e.g.: `copula_sample=copula.sample(100)`.

Transform that sample back to data scale:
```python
x_sample_in_data_scale=transform_uniform_margins.transform_from_uniform_margins_to_data_scale(copula_sample[:,0], x_gevd_fit_params)
y_sample_in_data_scale=transform_uniform_margins.transform_from_uniform_margins_to_data_scale(copula_sample[:,0], y_gevd_fit_params)
```

#### (8) Plot diagnostic to assess copula fit

To plot histograms of the copula in data scale (with GEVD/Gumbel fitted to observed extrema overplotted) and on uniform margins, use `transform_uniform_margins.plot_copula_diagnostic`. 

For example:
```python
fig_copula_1d,ax_copula_1d=transform_uniform_margins.plot_copula_diagnostic(copula_sample[:,0], copula_sample[:,1], x_sample_in_data_scale, y_sample_in_data_scale, x_gevd_fit_params, y_gevd_fit_params, 'X', 'Y')
```

Alternatively, to compare the 2D distributions of the observed extrema and copula sample, use `fit_copula_to_extremes.qualitative_copula_fit_check_bivariate`.

For example:
```python
fig_copula_2d,ax_copula_2d=fit_copula_to_extremes.qualitative_copula_fit_check_bivariate(x_extremes_df.extreme, y_extremes_df.extreme, x_sample_in_data_scale, y_sample_in_data_scale, 'X', 'Y')
```

#### (9) Plot return period as a function of two variables

To plot the return period as a function of x and y, with standard contours.

For example:
```python
fig_return_period,ax_return_period=calculate_return_periods_values.plot_return_period_as_function_x_y(copula,np.nanmin(x_extremes_df.extreme),np.nanmax(x_extremes_df.extreme),np.nanmin(y_extremes_df.extreme),np.nanmax(y_extremes_df.extreme),'X','Y', x_gevd_fit_params, y_gevd_fit_params, 'X (units)', 'Y (units)', n_samples=1000,block_size=block_size)
```

## Multivariate Analysis

To be completed

## Acknowledgements

ARF gratefully acknowledges the support of Irish Research Council Government of Ireland Postdoctoral Fellowship GOIPD/2022/782.
