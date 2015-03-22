#Duncan Campbell
#March 21, 2015
#Yale University


"""
functions used to calculate abundance functions, e.g. halo mass function, given a sample.
"""

import numpy as np
from scipy import optimize


def fit_abundance(x, weights, bins, xlog=True, fit_type='schechter'):
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
    
    dn, x = raw_abundance(x, weights, bins, xlog=xlog, monotonic_correction=False)
    
    params, cov = _get_fitting_function(fit_type)
    
    optimize.curve_fit(f_fit, x, dn)
    
    dndx = Lambda x: f_fit(x,*params)
    
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
    
    if use_log==True:
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
         return a * (x/b)**c * np.exp(-x)

    def double_schechter_function(x, a1, b1, c1, a2, b2, c2):
         return a1 * (x/b1)**c1 * np.exp(-x) + a1 * (x/b1)**c1 * np.exp(-x)
        
    if name=='schechter':
        return schechter_function
    elif name=='double_schechter':
        return double_schechter_function
    else:
        raise ValueError("fitting function not avalable.")


