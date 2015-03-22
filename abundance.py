#Duncan Campbell
#March 21, 2015
#Yale University


"""
functions used to calculate abundance functions, e.g. halo mass function, given a sample.
"""

import numpy as np
from scipy import optimize
import sys
import matplotlib.pyplot as plt

def fit_abundance(x, weights, bins, xlog=True, fit_type='schechter', show_fit=True):
    """
    given objects with property 'x' and weights, fit a functional form to the raw 
    abundances.
    
    Parameters
    ==========
    x: array_like
    
    weights: array_like
    
    bins: array_like
    
    xlog: boolean, optional
        If True, use log values of x. 
    
    fit_type: string, optional
        type of function to use for fit
    
    Returns
    =======
    dndx: function
    
    """
    
    #get empirical measurements of the abundance function
    dn, x = raw_abundance(x, weights, bins, xlog=xlog, monotonic_correction=False)
    
    #get function forms of the fitting function
    f_fit, log_f_fit = _get_fitting_function(fit_type)
   
    #get initial guesses for the schechter parameters
    n0 = np.median(dn) 
    x0 = np.median(x)
    s0 = np.median((np.log10(dn)[1:]-np.log10(dn)[:-1])/np.diff(x))
    
    #fit function
    params, cov = optimize.curve_fit(log_f_fit, x, np.log10(dn), p0=[n0,x0,s0])

    #apply fitted parameters to function
    dndx = lambda x: f_fit(x,*params)
    
    #plot mass function
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.9)
    plt.plot(x,dn,'.')
    plt.plot(x,dndx(x),'-')
    plt.yscale('log')
    plt.ylim([10**-7,10**1])
    plt.show(block=show_fit)
    
    return dndx


def raw_abundance(x, weights, bins, xlog=True, monotonic_correction=True):
    """
    given objects with property 'x' and weights, return an abundances.
    
    Parameters
    ==========
    x: array_like
    
    weights: array_like
    
    bins: array_like
    
    xlog: boolean, optional
        If True, use log values of x. 
    
    monotonic_correction: boolean, optional
        If True, attempt to correct for minor non-monotonicity
    
    Returns
    =======
    dndx: dn, x
    """
    
    if xlog==True:
        x = np.log10(x)
    
    if np.shape(weights)==():
        weights = np.array([weights]*len(x))
    
    n = np.histogram(x,bins,weights=weights)[0]
    bin_centers = (bins[:-1]+bins[1:])/2.0
    dx = bins[1:]-bins[:-1]
    dn = n/dx
    
    #remove bins with zero counts
    keep = (dn>0.0)
    dn = dn[keep]
    bin_centers = bin_centers[keep]
    
    if not _is_monotonic(dn,bin_centers):
        print("warning, function is not monotonic.")
    
    if monotonic_correction==False:
        return dn, bin_centers
        
    reverse = _is_reversed(dn,bin_centers)
    
    sorted_inds = np.argsort(dn)
    dn = dn[sorted_inds]
    if reverse==True:
        bin_centers = bin_centers[::-1]
    
    #if two values are exactly the same, add 0.5 an average count to one of them
    i=0
    avg_weight = np.mean(weights)
    print(dn)
    for val1,val2 in zip(dn[:-1],dn[1:]):
        if val1==val2:
            dn[i+1] = dn[i]+avg_weight/dx[i+1]*0.5
            dn[i] = dn[i]-avg_weight/dx[i+1]*0.5
        i+=1
    
    #check
    if not _is_monotonic(dn,bin_centers):
        raise ValueError("abundance function could not be calculated.")
    
    return dn, bin_centers


def _get_fitting_function(name):
    """
    return a fitting function
    """

    def schechter_function(x, a, b, c):
        x = 10**x
        b = 10**b
        val = a * (x/b)**c * np.exp(-x/b)
        return val
    
    def log_schechter_function(x, a, b, c):
        x = 10**x
        b = 10**b
        val = a * (x/b)**c * np.exp(-x/b)
        return np.log10(val)

    def double_schechter_function(x, a1, b1, c1, a2, c2):
        x = 10**x
        b1 = 10**b1
        val = a1 * (x/b1)**c1 * np.exp(-x/b1) + a2 * (x/b1)**c2 * np.exp(-x/b1)
        return val
    
    def log_double_schechter_function(x, a1, b1, c1, a2, c2):
        x = 10**x
        b1 = 10**b1
        val = a1 * (x/b1)**c1 * np.exp(-x/b1) + a2 * (x/b1)**c2 * np.exp(-x/b1)
        return np.log10(val)
         
        
    if name=='schechter':
        return schechter_function, log_schechter_function
    elif name=='double_schechter':
        return double_schechter_function, log_double_schechter_function
    else:
        raise ValueError("fitting function not avalable.")


def _is_monotonic(x,y):
    """
    Is the tabulated function y(x) monotonically increasing or decreasing?
    """
    sorted_inds = np.argsort(x)
    x = x[sorted_inds]
    y = y[sorted_inds]
    
    N_greater = 0
    N_less = 0
    for i in range(1,len(x)):
       if y[i]>y[i-1]: N_greater = N_greater+1
       else: N_less = N_less+1
    
    if (N_greater==len(x)-1) | (N_less==len(x)-1):
        return True
    else: return False

def _is_reversed(x,y):
    """
    Does the tabulated function y(x) decrease for increasing x?
    """
    sorted_inds = np.argsort(x)
    x = x[sorted_inds]
    y = y[sorted_inds]
    
    N_greater = 0
    N_less = 0
    
    for i in range(1,len(x)):
       if y[i]>y[i-1]: N_greater = N_greater+1
       else: N_less = N_less+1
    
    if (N_greater > N_less):
        return False
    else: return True
