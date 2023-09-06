# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:44:18 2023

@author: A R Fogg
"""

import scipy

import numpy as np
import matplotlib.pyplot as plt

def transform_from_data_scale_to_uniform_margins_empirically(data, plot=False):
    """
    Transform the data to uniform margins empirically
    by ranking the data

    Parameters
    ----------
    data : np.array
        
    plot : BOOL, optional
        If plot == True, plots the distributions of data in data 
        scale and on uniform margins. The default is False.

    Returns
    -------
    data_unif : np.array
        Data on uniform margins

    """
    
    # Transform the variables to uniform margins
    data_unif=scipy.stats.rankdata(data)/(data.size+1)

    if plot==True:
        
        fig,ax=plt.subplots(ncols=2,figsize=(8,4))
        
        ax[0].hist(data, bins=25, density=True, rwidth=0.8, color='cornflowerblue')
        ax[0].set_ylabel('Normalised Occurrence')
        ax[0].set_xlabel('Data in data scale')
        
        ax[1].hist(data_unif, bins=25, density=True, rwidth=0.8, color='darkorange')
        ax[1].set_ylabel('Normalised Occurrence')
        ax[1].set_xlabel('Data on uniform margins')
        
        plt.show()

    return data_unif

def transform_from_data_scale_to_uniform_margins_using_CDF(data, fit_params, distribution='genextreme', plot=False):
    """
    Transforms the data to uniform margins by plugging into the CDF
    (a probability integral transform)
    Distribution is G(x) = some formula
    1 - G(x) = u
    where u is on uniform margins, x is in data scale

    Parameters
    ----------
    data : np.array
        Data in data scale.
    fit_params : df
        For distribution='genextreme', must contain parameters scale, shape_, location.
    distribution : TYPE, optional
        DESCRIPTION. The default is 'genextreme'.
    plot : BOOL, optional
        If plot == True, plots the distributions of data in data
        scale and on uniform margins. The default is False.

    Returns
    -------
    data_unif : np.array
        Data transformed to uniform margins

    """
    
    if distribution=='genextreme':
        print('Transforming from data scale to uniform margins for GEVD distribution')
        data_unif=np.full(data.size,np.nan)
        for i in range(data.size):
            # 1- G(x) = u
            # u = data on uniform margins
            
            # Something happens with the math when we do it all in one line.
            # so calculating "number" to the exponent
            #number=(1.0 + (fit_params.shape_.to_numpy() * ( (data[i]-fit_params.location.to_numpy())/(fit_params.scale.to_numpy()) )))
            #data_unif[i]=1.0 - np.exp(-1.0 *np.sign(number)*(np.abs(number))**(-1.0/fit_params.shape_)  )
            data_unif[i]=1.0 - np.exp(-1.0 * (1.0 + (fit_params.shape_ * ( (data[i]-fit_params.location)/(fit_params.scale) )))**(-1.0/fit_params.shape_)  )
            #print(' ')
            #print(data_unif[i], data[i] )
            #print((-1.0/fit_params.shape_.values[0]))
            #print(number)
            #print(number**(-1.0/fit_params.shape_.values[0]))
            #print(np.sign(number)*(np.abs(number))**(-1.0/fit_params.shape_))
            
            #print(fit_params.shape_)
    
    elif distribution=='gumbel_r':
        print('Transforming from data scale to uniform margins for the Gumbel distribution')
        data_unif=np.full(data.size,np.nan)
        for i in range(data.size):
            # 1- G(x) = u
            # u = data on uniform margins
            data_unif[i]=1.0 - np.exp( -1.0*np.exp( -1.0*( (data[i]-fit_params.location)/(fit_params.scale) ) ) )
            
    else:
        print('ERROR: distribution "'+distribution+'" not implemented yet or incorrect spelling')
        raise NameError(distribution+' distribution not implemented yet or incorrect spelling')
    
    if plot==True:
        fig,ax=plt.subplots(ncols=2,figsize=(8,4))
        
        ax[0].hist(data, bins=25, density=True, rwidth=0.8, color='cornflowerblue')
        ax[0].set_ylabel('Normalised Occurrence')
        ax[0].set_xlabel('Data in data scale')
        
        ax[1].hist(data_unif, bins=25, density=True, rwidth=0.8, color='darkorange')
        ax[1].set_ylabel('Normalised Occurrence')
        ax[1].set_xlabel('Data on uniform margins')
        
        plt.show()
        
    return data_unif
    

def transform_from_uniform_margins_to_data_scale(data_unif,fit_params, distribution='genextreme', plot=False):
    """
    Transform the data from uniform margins back to data scale
    using the CDF. 
    Distribution is G(x) = some formula
        1 - G(x) = u
    where u is on uniform margins, x is in data scale
    So we solve for x.

    Parameters
    ----------
    data_unif : np.array
        Data on uniform margins.
    fit_params : df
        For distribution='genextreme', must contain parameters scale, shape_, location.
    distribution : TYPE, optional
        DESCRIPTION. The default is 'genextreme'.
    plot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    """
    
    
    if distribution=='genextreme':
        # For the GEVD distribution
        print('Transforming data from uniform margins to data scale for GEVD distribution')    
        data=np.full(data_unif.size,np.nan)
        for i in range(data.size):
            data[i]=( (fit_params.scale) / ( fit_params.shape_ * (-np.log(1-data_unif[i])) ** fit_params.shape_ ) )-(fit_params.scale/fit_params.shape_)+(fit_params.location)
    
    
    if plot==True:
        fig,ax=plt.subplots(ncols=2,figsize=(8,4))
        
        ax[0].hist(data, bins=25, density=True, rwidth=0.8, color='cornflowerblue')
        ax[0].set_ylabel('Normalised Occurrence')
        ax[0].set_xlabel('Data in data scale')
        
        ax[1].hist(data_unif, bins=25, density=True, rwidth=0.8, color='darkorange')
        ax[1].set_ylabel('Normalised Occurrence')
        ax[1].set_xlabel('Data on uniform margins')
        
        plt.show()
    
    return data
    
    
    
    
    