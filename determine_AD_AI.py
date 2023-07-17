# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:40:20 2023

@author: A R Fogg

based on R code sent by Daire Healy (Maynooth)
"""

import scipy

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

# R code by Daire
# set.seed(12)

# # simulating data from a multivariate gaussian 
# mu_x = 0
# mu_y = 0
# sd_x = 1
# sd_y = 1
# corr = 0.95

# simulations = MASS::mvrnorm(n = 10000, c(mu_x, mu_y), matrix(c(sd_x*sd_x, corr*sd_x*sd_y, corr*sd_x*sd_y, sd_y*sd_y), nrow = 2))
# x = simulations[,1]
# y = simulations[,2]

# plot(x,y)

# # We want to transform these variables to uniform.
# # We can use probability integral transform if we know the distribution of the data
# # or if we can estimate them.
# # For a quick check we can do this empirically:
# x_unif = rank(x)/(length(x)+1)
# y_unif = rank(y)/(length(y)+1)
# plot(x_unif,y_unif)

# # calculate the "extremal dependence coefficient" of the two variables, called χ
# # for a range of quantiles u \in (0,1)
# # if χ -> 0 as u -> 1 then the variables are asymptotically independent (AI),
# # otherwise they are asymptotically dependent (AD)
# u = seq(0.01, 0.99, by = 0.01)
# chi = c()
# for(i in u){
#   chi = c(chi, sum(x_unif>i & y_unif>i)/sum(x_unif>i)) # estimate of prob(x>u | y>u)
# }

# # if this line goes to 0 as u -> 1 then the variables are AI
# plot(u, chi, type = 'l')


def test_check_AD_AI():
    
    # Making the same function as in Daire's code
    mean = [0, 0]
    cov = [[1, 0.95], [0.95, 1]]  # diagonal covariance
    
    x, y = np.random.multivariate_normal(mean, cov, 10000).T
    
    fig,ax=plt.subplots()
    
    ax.plot(x,y,linewidth=0.0,marker='o',fillstyle='none')
    
    # Transform the variables to uniform (??? need to understand with Daire)
    # I think this is just normalising between 0 and 1
    x_unif=scipy.stats.rankdata(x)/(x.size+1)
    y_unif=scipy.stats.rankdata(y)/(y.size+1)
    
    fig,ax=plt.subplots()
    ax.plot(x_unif, y_unif, linewidth=0.0, marker='x')
    
    # Calculate the "extremal dependence coefficient", called chi
    #    for a range of quantiles u (from 0 to 1)
    # if chi -> 0 as u -> 1, then x and y are asymptotically independent
    #   otherwise they are asymtotically dependent
    
    u=np.linspace(0,0.99,100)  # bins of width 0.01
    chi=[]
    for i in range(u.size):
        top,=np.where((x_unif>u[i]) & (y_unif>u[i]))
        bottom,=np.where(x_unif>u[i])
        chi.append( (top.size)/(bottom.size) )
        
    fig, ax=plt.subplots()
    ax.plot(u,chi)
    
    
    
    
def plot_extremal_dependence_coefficient(x_data,y_data, x_name, y_name, x_units, y_units, csize=17):
    
    # # TEST DATA
    # # Making the same function as in Daire's code
    # mean = [0, 0]
    # cov = [[1, 0.95], [0.95, 1]]
    # x_data, y_data = np.random.multivariate_normal(mean, cov, 10000).T
    
    
    
    # Define plotting window for generic test plotting
    fig=plt.figure(layout="constrained",figsize=(15,11))
    gs=GridSpec(2,2,figure=fig)

    # Define three axes to plot the data on
    ax_data=fig.add_subplot(gs[0,0])
    ax_data_unif=fig.add_subplot(gs[0,1])
    ax_edc=fig.add_subplot(gs[1,:])
    
    # Plot the original input data
    ax_data.plot(x_data,y_data,linewidth=0.0,marker="o", fillstyle='none', color='grey')
    #h_data=ax_data.hist2d(x_data, y_data, bins=25, density=True)
    #cb_data=fig.colorbar(h_data[3],ax=ax_data)
    # Formatting
    ax_data.set_xlabel(x_name+' '+x_units, fontsize=csize)
    ax_data.set_ylabel(y_name+' '+y_units, fontsize=csize)
    for label in (ax_data.get_xticklabels() + ax_data.get_yticklabels()):
        label.set_fontsize(csize)
    #cb_data.ax.tick_params(labelsize=csize)
    #cb_data.set_label("Normalised occurrence", fontsize=csize)
    # Transform the variables into "uniform" - ask Daire
    x_unif=scipy.stats.rankdata(x_data)/(x_data.size+1)
    y_unif=scipy.stats.rankdata(y_data)/(y_data.size+1)
    # PLOT X_UNIF AS FUNCTION OF X_DATA TO GET VISUALISATION OF EMPIRICAL CDF
    
    
    # Plot out these uniform data
    
    # Notes from meeting
    #    put it onto uniform margins
    #   put everything onto same marginal distributions - e.g. if one variable
    #   is gaussian and one not - comparing them will get biased information and results are over/underestimating dependence
    #   so put the data into it's CDF and then you get out a uniform distribution
    #   ^ so do this seperately to y and x -> so you get "data transformed through prob integral transform onto uniform margins"
    
    ax_data_unif.plot(x_unif, y_unif, linewidth=0.0, marker='x', color='orange')
    #h_unif=ax_data_unif.hist2d(x_unif, y_unif, bins=25, density=True)
    #cb_unif=fig.colorbar(h_unif[3],ax=ax_data_unif)
    # Formatting
    ax_data_unif.set_xlabel(x_name+" on uniform margins", fontsize=csize)
    ax_data_unif.set_ylabel(y_name+" on uniform margins", fontsize=csize)
    for label in (ax_data_unif.get_xticklabels() + ax_data_unif.get_yticklabels()):
        label.set_fontsize(csize)
    #cb_unif.ax.tick_params(labelsize=csize)
    #cb_unif.set_label("Normalised occurrence", fontsize=csize)
    
    # Calculate the "extremal dependence coefficient", called chi
    #    for a range of quantiles u (from 0 to 1)
    # if chi -> 0 as u -> 1, then x and y are asymptotically independent
    #   otherwise they are asymtotically dependent
    u=np.linspace(0,0.99,100)  # bins of width 0.01
    chi=[]
    for i in range(u.size):
        top,=np.where((x_unif>u[i]) & (y_unif>u[i]))
        bottom,=np.where(x_unif>u[i])
        chi.append( (top.size)/(bottom.size) )

    ax_edc.plot(u,chi, color='orange')
    # Formatting
    ax_edc.set_xlabel("Quantiles", fontsize=csize)
    ax_edc.set_ylabel("Extremal Dependence Coefficient, $\chi$", fontsize=csize)
    for label in (ax_edc.get_xticklabels() + ax_edc.get_yticklabels()):
        label.set_fontsize(csize)
    
    
    
def test_plot_edc():
    
    mean = [0, 0]
    cov = [[1, 0.95], [0.95, 1]]
    x_data, y_data = np.random.multivariate_normal(mean, cov, 10000).T
    
    x_name='X'
    y_name='Y'
    x_units='(units)'
    y_units='(units)'
    
    plot_extremal_dependence_coefficient(x_data, y_data, x_name, y_name, x_units, y_units)
    
    
