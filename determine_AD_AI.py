# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 15:40:20 2023

@author: A R Fogg

based on R code sent by Daire Healy (Maynooth)
"""

import scipy

import numpy as np
import matplotlib.pyplot as plt


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