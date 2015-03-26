
"""
parameterized models of the stellar mass - halo mass relation (SMHM)
"""

__all__=['Behroozi_Model','Yang_Model']

import numpy as np

class Behroozi_Model():
    """
    Parameterized SMHM relation from Behroozi et al 2013.
    
    Default parameters are from section 5 for z=0.0 (a=1), for Mpeak halo property.  They 
    have been altered to be in h=1 units.
    
    Parameters
    ==========
    epslion: float, optional
        log characteristic stellar mass to halo mass ratio
    
    M1: float, optional
        log characteristic halo mass
    
    alpha: float, optional
        faint-end slope of SMHM relation
    
    gamma: float , optional
        index of subpower law at massive end of SMHM relation
    
    delta: float, optional
        strength of subpower law at massive end of SMHM relation
        
    """
    def __init__(self, epsilon=-1.932, M1=11.359, alpha=-1.412, gamma=0.316, delta=3.508):
        self.epsilon = epsilon
        self.M1 = M1
        self.alpha=alpha
        self.gamma=gamma
        self.delta = delta
    
    def __call__(self,Mh):
        """
        Parameters
        ==========
        Mh: array_like
            log halo mass
        
        Returns
        =======
        Mstar: array_like
            log stellar mass
        """
        def f(x):
            val = -1.0 * np.log10(10.0**(self.alpha*x)+1.0) +\
                self.delta*(np.log10(1+np.exp(x)))**self.gamma/(1.0+np.exp(10.0**(-x)))
            return val
    
        Mh = 10.0**Mh
        M1 = 10.0**self.M1
        epsilon = 10.0**self.epsilon
        Mstar = np.log10(epsilon*M1) + f(np.log10(Mh/M1))-f(0.0)
        return Mstar


class Yang_Model():
    """
    Parameterized SMHM relation from Yang et al. 2003.
    
    Default parameters are from Moster et al. 2014 table 1.  They have been altered to be 
    in h=1 units.
    
    Parameters
    ==========
    norm: float, optional
        characteristic stellar-to-halo mass ratio
    
    M1: float, optional
        log characteristic halo mass
    
    beta: float, optional
        low mass slope
    
    gamma: float, optional
        high mass slope
        
    """
    def __init__(self, norm=0.0203, M1=11.741, beta=1.057, gamma=0.556):
        self.norm = norm
        self.M1 = M1
        self.beta = beta
        self.gamma= gamma
    
    def __call__(self, Mh):
       """
        Parameters
        ==========
        Mh: array_like
            log halo mass
        
        Returns
        =======
        Mstar: array_like
            log stellar mass
            
        """
        Mh = 10.0**Mh
        M1 = self.M1
        norm = self.norm
        beta = self.beta
        gamma = self.gamma
        M1 = 10.0**M1
        mstar = 2.0*norm*((Mh/M1)**(-1.0*beta)+(Mh/M1)**gamma)**(-1.0)*Mh
        return np.log10(mstar)
    