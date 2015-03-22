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
    #make cut to halo catalogue
    mp_bolshoi = 1.35e8
    mpeak_cut = mp_bolshoi*100.0
    keep = (HC['Mpeak']>mpeak_cut)
    HC = HC[keep]
    Lbox = 250.0
    
    #calculate (sub-)halo mass function from halo catalogue
    halo_prop = 'Mpeak'
    x = HC[halo_prop]
    bins = np.arange(10.2,16,0.1)
    dndm_halo = fit_abundance(x, weights=1.0/Lbox**3.0, bins=bins, xlog=True,\
                              fit_type='schechter', show_fit=True)
    
    #define stellar mass function, Li and White 2009
    dndm_gal = get_sdss_smf()
    
    #define the form of P(gal | halo)
    from scipy.stats import norm
    def sigma(y):
        return 0.15 #fixed scatter model
    def P_x(y, mu_xy, sigma=sigma):
        mu_x = mu_xy(y)
        sigma = sigma(y)
        p = norm(loc=mu_x, scale=sigma)
        return p
    
    #calculate mean of the relation
    #define samples of the galaxy abundance function
    dm_star = np.arange(5,12,0.1)
    dn_star = dndm_gal(dm_star)
    #define samples of the halo abundance function
    dm_halo = np.arange(10,15,0.1)
    dn_halo = dndm_halo(dm_halo)
    mu_xy = AM(dn_star, dm_star, dn_halo, dm_halo, P_x, y_min = np.amin(dm_halo), y_max = np.amax(dm_halo), ny=30)

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


def get_sdss_smf():
    from scipy import interpolate
    #define stellar mass function, Li and White, 2009
    def dn_dmstar(m,dm):
        alpha = -1.155
        m_star = 10**10.525
        phi_star = 0.0083
        
        dndm = np.exp(-m/m_star)*phi_star*(m/m_star)**alpha* dm/m_star
        return dndm
    logm = np.arange(5.0,12.5,0.1)
    m = 10.0**logm
    dlogm = logm[1:]-logm[:-1]
    dm = m[1:]-m[:-1]
    mstar = logm[:-1]
    dnstar = dn_dmstar(m[:-1],dm)/dlogm
    dndlogMstar = interpolate.interp1d(mstar,dnstar)
    
    return dndlogMstar


if __name__ == '__main__':
    main() 