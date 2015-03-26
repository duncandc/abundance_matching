#!/usr/bin/env python

#Duncan Campbell
#March 22, 2015
#Yale University
#show example of abundance matching code

#load packages
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import sys
from AM import AM
from abundance import fit_abundance, raw_abundance
from make_mocks import make_SHAM_mock

#packages to read in halo catalogue
import custom_utilities as cu
import h5py


def main():

    #open halo catalogue
    filepath = cu.get_output_path() + 'processed_data/Multidark/Bolshoi/halo_catalogues/'
    halo_catalogue = 'hlist_1.00030.list'
    f =  h5py.File(filepath+halo_catalogue+'.hdf5', 'r')
    HC = f.get(halo_catalogue)
    HC = np.array(HC)
    #make completeness cut to halo catalogue
    mp_bolshoi = 1.35e8
    mpeak_cut = mp_bolshoi*100.0
    keep = (HC['Mpeak']>mpeak_cut)
    HC = HC[keep]
    Lbox = 250.0
    
    #calculate (sub-)halo mass function from halo catalogue
    halo_prop = 'Mpeak'
    x = HC[halo_prop]
    bins = np.arange(10.1,16,0.1)
    p = dict(phi1=0.001,x1=14,alpha1=-1.0) #initial guess for parameters to fit function
    dndm_halo = fit_abundance(x, weights=1.0/Lbox**3.0, bins=bins, xlog=True,\
                              fit_type='schechter', p = p, show_fit=True)
    
    #define stellar mass function, Baldry 2008.
    M0 = 10.648 + np.log10(0.7**2)
    phi1 = 4.26 * 10**(-3) / 0.7**3
    phi2 = 0.58 * 10**(-3) / 0.7**3
    alpha1 = -0.46
    alpha2 = -1.58
    dndm_gal = cu.schechter_function.Log_Double_Schechter(M0,M0,phi1,phi2,alpha1,alpha2)
    
    #plot the stellar mass function
    dm_star = np.arange(7,12,0.1)
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.9)
    plt.plot(dm_star,dndm_gal(dm_star))
    plt.yscale('log')
    plt.ylim([10**(-8),1])
    plt.show()
    
    #define the form of P(gal | halo)
    from scipy.stats import norm
    def sigma(y):
        return 0.2 #fixed scatter model
    def P_x(y, mu_xy, sigma=sigma):
        mu_x = mu_xy(y)
        sigma = sigma(y)
        p = norm(loc=mu_x, scale=sigma)
        return p
    
    #calculate mean of the relation
    #define samples of the galaxy abundance function
    dm_star = np.arange(0,12.5,0.01)
    dn_star = dndm_gal(dm_star)
    #define samples of the halo abundance function
    dm_halo = np.arange(10.0,15,0.01)
    dn_halo = dndm_halo(dm_halo)
    mu_xy = AM(dn_star, dm_star, dn_halo, dm_halo,\
               P_x, y_min = np.amin(dm_halo), y_max = np.amax(dm_halo), ny=30)

    #apply the mean
    P = lambda y: P_x(y, mu_xy=mu_xy)
    
    #make mock
    mock = make_SHAM_mock(HC, P, mock_prop=halo_prop, gal_prop='mstar', use_log_mock_prop=True)
    
    #plot stellar mass halo mass relation w/mean over-plotted
    fig = plt.figure(figsize=(3.3,3.3))
    fig.subplots_adjust(left=0.2, right=0.85, bottom=0.2, top=0.9)
    plt.scatter(np.log10(mock['Mpeak']),mock['mstar'], s=3, marker='.', lw=0, alpha=0.1)
    plt.plot(dm_halo,mu_xy(dm_halo),'red')
    plt.xlabel(r'$M_{\rm vir}$')
    plt.ylabel(r'$M_{*}$')
    plt.ylim([9,12])
    plt.xlim([9,15])
    plt.show(block=True)


if __name__ == '__main__':
    main() 