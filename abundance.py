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

def fit_abundance(x, weights, bins, xlog=True, fit_type='schechter', p=None, show_fit=True):
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
    
    p: dict, optional
        initial guesses of parameters in fit
    
    show_fit: boolean, optional
        plot fit
    
    Returns
    =======
    dndx: function
    
    """
    from astropy.modeling import fitting
    
    #get empirical measurements of the abundance function
    dn, x, err = raw_abundance(x, weights, bins, xlog=xlog, show=False)
    
    #get function forms of the fitting function
    f_model = _get_fitting_function(fit_type)
    
    #fit function to tabulate abundance function
    f_init = f_model(**p)
    fit_f = fitting.LevMarLSQFitter()
    f = fit_f(f_init, x, dn, weights=1.0/dn)
    
    if show_fit:
        fig = plt.figure(figsize=(3.3,3.3))
        fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.9)
        plt.errorbar(x, dn, yerr=err, fmt='.', color='k')
        plt.plot(x,f(x),'-')
        plt.yscale('log')
        plt.show()
    
    return f


def raw_abundance(x, weights, bins, xlog=True, monotonic_correction=False, show=False):
    """
    given objects with property 'x' and weights, return tabulated abundances.
    
    Parameters
    ==========
    x: array_like
    
    weights: array_like
    
    bins: array_like
    
    xlog: boolean, optional
        If True, use log values of x. 
    
    monotonic_correction: boolean, optional
        If True, attempt to correct for minor non-monotonicity
    
    show: boolean, optional
        plot abundance function 
    
    Returns
    =======
    dndx: dn, x, err
    """
    
    if xlog==True:
        x = np.log10(x)
    
    if np.shape(weights)==():
        weights = np.array([weights]*len(x))
    
    n = np.histogram(x, bins, weights=weights)[0]
    bin_centers = (bins[:-1]+bins[1:])/2.0
    dx = bins[1:]-bins[:-1]
    dn = n/dx
    
    raw_counts = np.histogram(x, bins=bins)[0]
    err = (1.0/np.sqrt(raw_counts))*dn
    
    #remove zeros
    keep = (raw_counts>0)
    dn = dn[keep]
    bin_centers = bin_centers[keep]
    err = err[keep]
    
    if show==True:
        fig = plt.figure(figsize=(3.3,3.3))
        fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.9)
        plt.errorbar(bin_centers, dn, yerr=err,fmt='.')
        plt.yscale('log')
        plt.show()
    
    return dn, bin_centers, err


####Utility Functions#####################################################################
def _get_fitting_function(name):
    """
    return a model fitting function, and a function class
    """

    from astropy.modeling.models import custom_model
    import custom_utilities as cu

    @custom_model
    def schechter_model(x, phi1=1, x1=1, alpha1=-1):
        norm = np.log(10.0)*phi1
        val = norm*(10.0**((x-x1)*(1.0+alpha1)))*np.exp(-10.0**(x-x1))
        return val

    @custom_model
    def double_schechter_model(x, phi1=1, phi2=1, x1=1, x2=1, alpha1=-1, alpha2=-1):
        norm = np.log(10.0)
        val = norm *\
              (np.exp(-10.0**(x - x1)) * 10.0**(x - x1) *\
                   phi1 * (10.0**((x - x1) * alpha1)) +\
               np.exp(-10.0**(x - x2)) * 10.0**(x - x2) *\
                   phi2 * (10.0**((x - x2) * alpha2)))
        return val

    if name=='schechter':
        return schechter_model
    elif name=='double_schechter':
        return double_schechter_model
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
