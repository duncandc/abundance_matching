#Duncan Campbell
#Yale University
#March 2015

"""
Build a SHAM mock.
"""

from __future__ import print_function, division
import numpy as np

from scipy import interpolate
from scipy import optimize
from scipy import integrate
from scipy import stats
import matplotlib.pyplot as plt

import sys
import time


__all__=['AM','abundance_function','make_SHAM_mock']


def AM(dn_dx, x, dn_dy, y, P_x=None, y_min=None, y_max=None, ny=100):
    """
    Abundance match galaxy property 'x' to halo property 'y'.
    
    Functional forms of the abundance of galaxies and haloes are required.
    
    Determines P(x_gal | y_halo), given the centered distribution, i.e. all other moments.
    
    In detail P should be of the form: P_x(y, mu_xy(y)), where y is a property of the halo
    and mu_xy(y) is the mean of the distribution as a function of y.  This function simply
    solves for the form of mu_xy, returning the full conditional probability function.
    
    Parameters
    ==========
    dn_dx: array_like
        galaxy abundance as a function of property 'x'
    
    x: array_like
    
    dn_dy: array_like
        halo abundance as a function of property 'y'
    
    y: array_like
    
    P_x: function
        The centered probability distribution of P(x_gal | y_halo).  If this is None,
        traditional abundance matching is done with no scatter.
    
    y_min: float
        minimum value to determine the form of P(x|y) for
    
    y_max: float
        maximum value to determine the form of P(x|y) for
    
    ny: int
        number of points used to sample the first moment in the range [y_min,y_max]]
    
    tol: float, optional
        tolerance of error on of the 1st moment of P(x| y).
    
    Returns
    =======
    mu : function 
        first moment of P(x| y)
    """
    
    #process input parameters
    x = np.array(x)
    y = np.array(y)
    dn_dx = np.array(dn_dx)
    dn_dy = np.array(dn_dy)
    
    #check halo abundance range inputs
    if y_min==None:
        y_min = np.amin(y)
    if y_max==None:
        y_min = np.amax(y)
    if y_min > y_max:
        raise ValueError("y_min must be less than y_max!")
    
    #define y samples of x(y). This will be the number of parameters to minimize when 
    #solving for x(y).  More parameters will slow the code down!
    ys = np.linspace(y_min,y_max,ny)
    
    #define y samples used when integrating dn_dy*P(x|y)dy. This needs to be finely spaced 
    #enough to get accurate integrals of P(x|y)dy. This can get very narrow for small
    #scatter and/or for steep x(y) relations.
    yy = np.linspace(y_min,y_max,1000)
    
    #keep halo abundance function for values in user specified range.
    keep = ((y>=y_min) & (y<=y_max))
    y = y[keep]
    dn_dy = dn_dy[keep]
    
    ##########################################################
    #step 0: preliminary work
    
    #put in order so the independent variable is monotonically increasing
    #galaxies
    inds = np.argsort(x)
    x = x[inds]
    dn_dx = dn_dx[inds]
    #haloes
    inds = np.argsort(y)
    y = y[inds]
    dn_dy = dn_dy[inds]
    
    #enforce monotonic abundance functions
    if not _is_monotonic(x,dn_dx):
        raise ValueError("galaxy abundance function must be monotonic")
    if not _is_monotonic(y,dn_dy):
        raise ValueError("halo abundance function must be monotonic")
    
    #check direction of increasing number density.  Usually this is reversed.
    #This affects the sign of integrations, and how interpolations are implemented.
    if _is_reversed(x,dn_dx):
        reverse_x = True
    if _is_reversed(y,dn_dy):
        reverse_y = True
    
    print("x decreases as dn_dx increases: ", reverse_x)
    print("y decreases as dn_dy increases: ", reverse_y)
    
    #check numerical values of dn_dx and dn_dy
    #trim down ranges if they return numbers they are too small
    keep = (dn_dx>10.0**(-20.0))
    if not np.all(keep):
        print("Triming x-range to keep number densities above 1e-20")
        x = x[keep]
        dn_dx = dn_dx[keep]
    keep = (dn_dy>10.0**(-20.0))
    if not np.all(keep):
        print("Triming y-range to keep number densities above 1e-20")
        y = y[keep]
        dn_dy = dn_dy[keep]
    
    #convert tabulated abundance functions into function objects using spline 
    #interpolation.
    #galaxy abundance function
    dn_dx = interpolate.InterpolatedUnivariateSpline(x, dn_dx)
    #halo abundance function
    dn_dy = interpolate.InterpolatedUnivariateSpline(y, dn_dy)
    
    #plot the input abundances of galaxies and haloes
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.9)
    plt.plot(x,np.log10(dn_dx(x)),'-', color="blue")
    plt.plot(y,np.log10(dn_dy(y)),'-', color="black")
    plt.xlabel(r'$x,y$')
    plt.ylabel(r'$dn/dx,y$')
    plt.show(block=True)

    #calculate the cumulative abundance functions
    N_cum_halo = _cumulative_abundance(dn_dy, y, reverse=reverse_y)
    N_cum_gal  = _cumulative_abundance(dn_dx, x, reverse=reverse_x)
    
    #galaxy cumulative abundance function number density range must span that of the halo 
    #abundance function for abundance matching to be possible.
    N_min = min(N_cum_halo(y_min),N_cum_halo(y_max))
    N_max = max(N_cum_halo(y_min),N_cum_halo(y_max))
    print("maximum halo cumulative abundnace: {0}".format(N_max))
    print("mimimum halo cumulative abundnace: {0}".format(N_min))
    print("maximum galaxy cumulative abundnace: {0}".format(np.amax(N_cum_gal(x))))
    print("mimimum galaxy cumulative abundnace: {0}".format(np.amin(N_cum_gal(x))))
    if not (np.amax(N_cum_gal(x)) >= N_max) & (np.amin(N_cum_gal(x)) <= N_min):
        raise ValueError("Galaxy cumulative abundances must span range of halo abundances!")
    
    #calculate the inverse cumulative abundances
    if reverse_x==True:
        N_cum_gal_inv  = interpolate.InterpolatedUnivariateSpline(N_cum_gal(x)[::-1], x[::-1], k=1)
    else:
        N_cum_gal_inv  = interpolate.InterpolatedUnivariateSpline(N_cum_gal(x), x, k=1)
    if reverse_y==True:    
        N_cum_halo_inv = interpolate.InterpolatedUnivariateSpline(N_cum_halo(y)[::-1], y[::-1], k=1)
    else:
        N_cum_halo_inv = interpolate.InterpolatedUnivariateSpline(N_cum_halo(y), y, k=1)

    #calculate pivot point. below this point, the galaxy abundance function will be 
    #incomplete given the halo value range under consideration.
    x_pivot = N_cum_gal_inv(N_cum_halo(y_min))
    print("minimum x value in range: {0}".format(x_pivot))

    #discard the galaxy abundance function below this point.
    keep = (x>x_pivot)
    x = x[keep]

    #plot the cumulative abundances of galaxies and haloes
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.9)
    plt.plot(x,np.log10(N_cum_gal(x)), '-', color="black")
    plt.plot(y,np.log10(N_cum_halo(y)), '-', color="black")
    plt.xlabel(r'$x,y$')
    plt.ylabel(r'$N(>x,y)$')
    plt.show(block=True)
    
    ##########################################################
    #step 1: solve for first moment in the absence of scatter.  This is abundance matching
    #with no scatter.
    x_y1 = N_cum_gal_inv(N_cum_halo(ys))
    x_y1 = interpolate.InterpolatedUnivariateSpline(ys, x_y1, k=1)

    #plot x(y)
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.9)
    plt.plot(y,x_y1(y),'.')
    plt.xlabel(r'$M_{\rm vir}$')
    plt.ylabel(r'$M_{*}$')
    plt.show(block=True)
    
    #if we are assuming no scatter in the relation, we are done.
    if P_x==None:
        return x_y1
    
    #apply this first estimate of the 1st moment to P(x|y)
    P1 = lambda y: P_x(y, mu_xy=x_y1)
    
    #calculate the minimum x property for which we will be complete.
    x_min = P1(y_min).interval(0.999)[1]
    print("Results only valid for galaxy function down to: {0} ".format(x_min))
    
    #show the first estimate of the first moment.  Also show limits.
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.9)
    plt.plot(ys, x_y1(ys), '-', color="green")
    plt.plot([y_min,y_max],[x_min,x_min], color='black')
    plt.plot([y_min,y_min],[x_min,np.amax(x)], color='black')
    plt.xlabel(r'$M_{\rm vir}$')
    plt.ylabel(r'$M_{*}$')
    plt.show()
    
    ##########################################################
    #step 2: get second estimate of the first moment by integrating to get the galaxy
    #abundance function of haloes and solve for N_halo(x) = N_halo(y) to get a second
    #estimate of x(y)
    
    #define integrand
    def integrand(y,x):
        return P1(y).pdf(x)*dn_dy(y)
    
    """
    #take a look at the integrals that need to be calculated.
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.9)
    plt.show(block=False)
    for xx in x:
        print("x: ", xx)
        f = lambda yy :P1(yy).pdf(xx)
        plt.plot(yy,f(yy),color='grey')
        plt.draw()
        time.sleep(0.3)
    plt.show()
    """
    
    #do integral
    dn_halo_dx = np.zeros(len(x)) #define array to store numeric integration result
    for i, xx in enumerate(x):
        f = lambda y: integrand(y,xx) #simplify the integrand
        dn_halo_dx[i] = integrate.simps(f(yy),yy)
    
    #get galaxy abundance function function (number of haloes as a function of x)
    dn_halo_dx = interpolate.InterpolatedUnivariateSpline(x, dn_halo_dx, k=1)
    
    #get new cumulative function
    N_cum_halo_x = _cumulative_abundance(dn_halo_dx, x, reverse = reverse_x)
    
    #get inverse cumulative mass function
    if reverse_x==True:
        N_cum_halo_x_inv = interpolate.InterpolatedUnivariateSpline(N_cum_halo_x(x)[::-1], x[::-1], k=1)
    else:
        N_cum_halo_x_inv = interpolate.InterpolatedUnivariateSpline(N_cum_halo_x(x), x, k=1)
    
    #get x2(x)
    x2_x1 = N_cum_gal_inv(N_cum_halo_x(x))
    x2_x1 = interpolate.InterpolatedUnivariateSpline(x, x2_x1, k=1)
    
    #get x2(y).  This is our second estimate.
    x_y2 = x2_x1(x_y1(ys))
    x_y2 = interpolate.InterpolatedUnivariateSpline(ys, x_y2, k=1)
    
    #apply new estimate of the first moment to P(x|y)
    P2 = lambda y: P_x(y, mu_xy=x_y2)
    
    #plot first two estimates of the first moment of x(y)
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.9)
    plt.plot(ys,x_y1(ys),'-',color='red')
    plt.plot(ys,x_y2(ys),'.',color='blue',ms=2)
    plt.xlabel(r'$M_{\rm vir}$')
    plt.ylabel(r'$M_{*}$')
    plt.show(block=True)
    
    #plot what the galaxy abundance function looks like at this step
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.9)
    plt.plot(x,np.log10(dn_dx(x)),'-',color='red')
    plt.plot(x,np.log10(dn_halo_dx(x)),'.',color='blue',ms=2)
    plt.plot([x_min,x_min],[-8,1],color='grey')
    plt.xlabel(r'$M_{*}$')
    plt.ylabel(r'n')
    plt.ylim([-8,1])
    plt.show(block=True)
    
    ##########################################################
    #step 2.5, repeat step 2 a few times to refine the estimate.
    
    Pi = P2 #previous iteration of the P(x|y)
    x_yi = x_y2 #previous iteration of the x(y)
    for i in range(0,100): 
        
        #define integrand
        def integrand(y,x):
             return Pi(y).pdf(x)*dn_dy(y)
    
        #do integral
        dn_halo_dx = np.zeros(len(x)) #define array to store numeric integration result
        for i, xx in enumerate(x):
            f = lambda y: integrand(y,xx) #simplify the integrand
            dn_halo_dx[i] = integrate.simps(f(yy),yy)
    
        #get galaxy abundance function function (number of haloes as a function of x)
        dn_halo_dx = interpolate.InterpolatedUnivariateSpline(x, dn_halo_dx, k=1)
    
        #get new cumulative function
        N_cum_halo_x = _cumulative_abundance(dn_halo_dx, x, reverse = reverse_x)
    
        #get inverse cumulative mass function
        if reverse_x==True:
            N_cum_halo_x_inv = interpolate.InterpolatedUnivariateSpline(N_cum_halo_x(x)[::-1], x[::-1], k=1)
        else:
            N_cum_halo_x_inv = interpolate.InterpolatedUnivariateSpline(N_cum_halo_x(x), x, k=1)
    
        #get x2(x)
        x2_x1 = N_cum_gal_inv(N_cum_halo_x(x))
        x2_x1 = interpolate.InterpolatedUnivariateSpline(x, x2_x1, k=1)
    
        #get x2(y).  This is our second estimate.
        x_yi = x2_x1(x_yi(ys))
        x_yi = interpolate.InterpolatedUnivariateSpline(ys, x_yi, k=1)
    
        #apply new estimate of the first moment to P(x|y)
        Pi = lambda y: P_x(y, mu_xy=x_yi)
    
    P2 = Pi #save those final estimates for the next step
    x_y2 = x_yi
    
    ##########################################################
    #step 3: use the latest estimate of x(y) to solve for a new x(y) which minimizes the
    #difference between the input galaxy abundance function, and the inferred abundance 
    #function given the halo abundance function and P(x|y).
    
    def F2(p):
        """
        Takes derivatives of x(y) as parameters
        """
        #print(p)
        #integrate the derivatives to get original function
        x_y = np.zeros(len(ys))
        norm = x_y2(ys[0])
        x_y[0] = x_y2(ys[0])
        x_y[1:] = norm+np.cumsum(p)*np.diff(ys)
        
        x_y = interpolate.InterpolatedUnivariateSpline(ys, x_y, k=1)
        
        #apply first estimate of the 1st moment
        P = lambda y: P_x(y, mu_xy=x_y)
        
        #define integrand
        def integrand(y,x):
            return P(y).pdf(x)*dn_dy(y)
        
        #do integral
        dn_halo_dx = np.zeros(len(x)) #define array to store numeric integration result
        for i, xx in enumerate(x):
            f = lambda y: integrand(y,xx) #simplify the integrand
            dn_halo_dx[i] = integrate.simps(f(yy),yy)
        
        #only compare above x_min completeness limit
        keep  = (x>x_min)
        
        #define quantity to minimize
        estimate = np.log10(dn_halo_dx[keep])
        truth = np.log10(dn_dx(x)[keep])
        chi = stats.chisquare(estimate,truth)[0]
        diff = np.sum(np.fabs(estimate-truth))
        
        val = np.sum(((estimate-truth)/truth)**2.0)
        
        #plot the iteration
        ax1.plot(x,np.log10(dn_halo_dx),'-',color='grey')
        ax2.plot(ys,x_y(ys),'-',color='grey')
        plt.draw()
        
        return val


    #set up plot to show progress of minimization routine
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.6, 3.3))
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.9)
    ax1 = axes[0]
    ax2 = axes[1]
    #on the left, show the galaxy abundance function
    ax1.plot(x,np.log10(dn_dx(x)),'-',color='red') #truth in red
    #on the right, show the x(y) estimations
    ax2.plot(ys,x_y1(ys),'-',color='red') #first estimate in red
    ax2.plot(ys,x_y2(ys),'-',color='black') #second estimate in black
    ax2.set_xlabel(r'$M_{\rm vir}$')
    ax2.set_ylabel(r'$M_{*}$')
    plt.show(block=False)
    
    #calculate the bounds of the parameters of solution, using the starting point is an 
    #upper-limit.
    
    #bounds for the parameters. enforce that they are all >0, i.e. that x(y) increases.
    #calculate the derivative of x(y)
    dx_y = x_y2.derivative(n=1)
    dx_y = dx_y(ys[:-1])
    
    """
    #plot the derivatives
    plt.figure()
    plt.plot(y[:-1],dx_y)
    plt.show()
    """
    
    #set lower limits
    min_vals = np.array([0.0]*len(dx_y)) #cannot be negative (must monotonically increase)
    max_vals = np.array([None]*len(dx_y)) #no upper limit
    #keep the parameters fixed that fall below the completeness limit
    fix = (x_y2(ys)[:-1]<x_min)
    min_vals[fix] = dx_y[fix]
    max_vals[fix] = dx_y[fix]
    norm = x_y2(ys[0]) #the normalization is taken to be the first point
    deriv_bounds = [(min_val,max_val) for min_val, max_val in zip(min_vals,max_vals)]
    
    #do minimization of the derivatives w/ normalization fixed, F2
    result = optimize.minimize(F2, dx_y, bounds=deriv_bounds, method="L-BFGS-B", options={'maxiter': 10, 'disp':True})

    #integrate derivatives to get x(y)
    x_y = np.zeros(len(ys))
    x_y[0] = norm
    x_y[1:] = norm+np.cumsum(result.x)*np.diff(ys)
    x_y = interpolate.InterpolatedUnivariateSpline(ys, x_y, k=1)
    
    return x_y


def make_SHAM_mock(mock, P_xy, mock_prop='mvir', gal_prop='mstar', use_log_mock_prop=True):
    """
    make a SHAM mock given a halo catalogue, mock, the probability of galaxy property 'x' 
    given a halo with property 'y', where mock[mock_prop] returns halo property 'y'.
    
    Parameters
    ==========
    mock: array_like
        structured array containing halo catalogue
    
    P_xy: function
        Function that returns x_gal given y_halo
    
    mock_prop: string
        key into mock which returns the halo property to build the SHAM mock
    
    Returns
    =======
    mock: structured array
        mock with new column containing galaxy property gal_prop
    """
    from numpy.lib.recfunctions import append_fields
    
    mock = mock.view(np.recarray)
    
    if use_log_mock_prop==True:
        y = np.log10(mock[mock_prop])
    else:
        y = mock[mock_prop]
    
    x = P_xy(y).rvs(len(mock))
    
    if gal_prop in mock.dtype.names:
        mock[gal_prop] = x
    else:
        mock = append_fields(mock,gal_prop,x)
    
    return mock


def abundance_function(x, weights, bins, use_log=True):
    """
    given observations of objects with property 'x' and weights, return an abundance 
    function
    
    Parameters
    ==========
    x: array_like
    
    weights: array_like
    
    bins: array_like
    
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
    
    print(dn)
    
    #check
    if not _is_monotonic(dn,bin_centers):
        raise ValueError("abundance function could not be calculated.")
    
    return dn, bin_centers


def _abundance_function_from_tabulated(x, dndx):
    """
    given a tabulated abundance function for property 'x', build a functional form.
    
    Parameters
    ==========
    x: array_like
        values of property 'x'
    
    dndx: array_like
        number density
    
    Returns
    =======
    dndx: function
    """
    
    #n_x = interpolate.interp1d(x, dndx, kind='linear')
    inds = np.argsort(x)
    n_x = interpolate.InterpolatedUnivariateSpline(x[inds], dndx[inds], k=1)
    """
    min_x = np.min(x) #minimum of the x range
    max_x = np.max(x) #maximum of the x range
    min_n = np.min(dndx) #minimum of the number density
    max_n = np.max(dndx) #maximum of the number density
    
    def f(x):
        return n_x(x)
    
    #store x and y ranges of the function
    setattr(f,'min_x',min_x)
    setattr(f,'max_x',max_x)
    setattr(f,'min_dn',min_n)
    setattr(f,'max_dn',max_n)
    
    #store tabulated form of the function
    setattr(f,'x',x)
    setattr(f,'dn',dndx)
    """
    
    return n_x


def _cumulative_abundance(f,bins,reverse=True):
    """
    calculate the cumulative abundance function by integrating the differential abundance 
    function
    """
    
    N_sample = len(bins) #number of points to sample along function
    
    if reverse==True:
        N0 = float(f(bins[-1]))*(bins[-2]-bins[-1])
    else:
        N0 = float(f(bins[0]))*(bins[1]-bins[0])
    
    #if the function is monotonically decreasing, integrate in the opposite direction
    if reverse==True:
        N_cum = integrate.cumtrapz(f(bins[::-1]), bins[::-1], initial=N0)
        N_cum = N_cum[::-1]*(-1.0)
    else:
        N_cum = integrate.cumtrapz(f(bins), bins, initial=N0)
    
    #interpolate result to get a functional form
    N_cum_func = interpolate.InterpolatedUnivariateSpline(bins, N_cum, k=1)
    
    return N_cum_func


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


def _integrate_to_get_gal_abundnace_function(P,x,y,dn_dy,y_min,y_max):
    
     #define integrand
    def integrand(y,x):
        return P(y).pdf(x)*dn_dy(y)
    
    #do integral
    dn_halo_dx = np.zeros(len(x)) #define array to store numeric integration result
    for i, xx in enumerate(x):
        print(i,xx)
        f = lambda y: integrand(y,xx) #simplify the integrand
        dn_halo_dx[i] = integrate.simps(f(y),y)
        #dn_halo_dx[i] = integrate.romberg(f,y_min,y_max)
        #dn_halo_dx[i] = integrate.quad(f,y_min,y_max)[0]
        #dn_halo_dx[i] = GaussLegendreQuadrature(f, 7, y_min,y_max)[0]
    
    #get new mass function
    dn_halo_dx = interpolate.InterpolatedUnivariateSpline(x, dn_halo_dx, k=1)
    
    return dn_halo_dx
    
