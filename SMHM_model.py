
"""
parameterized models of the stellar mass - halo mass relation
"""

__all__=['Behroozi_Model','Yang_Model']

import numpy as np

#Behroozi et al 2013
class Behroozi_Model():
    """
    stellar mass halo mass relation.  default parameters are from Behroozi 2013, where 
    h=0.7 units
    """
    def __init__(self, epsilon=-1.777, M1=11.514, alpha=-1.412, gamma=0.316, delta=3.508):
        self.epsilon = epsilon
        self.M1 = M1
        self.alpha=alpha
        self.gamma=gamma
        self.delta = delta
    
    def __call__(self,Mh):
    
        def f(x):
            val = -1.0 * np.log10(10.0**(self.alpha*x)+1.0) +\
                self.delta*(np.log10(1+np.exp(x)))**self.gamma/(1.0+np.exp(10.0**(-x)))
            return val
    
        Mh = 10.0**Mh
        M1 = 10.0**self.M1
        epsilon = 10.0**self.epsilon
        Mstar = np.log10(epsilon*M1) + f(np.log10(Mh/M1))-f(0.0)
        return Mstar

#Yang et al 2003, Moster 2014
class Yang_Model():
    """
    stellar mass halo mass relation.  default parameters are from Moster 2014, where 
    h=0.72
    """
    def __init__(self, norm=0.02820, M1=11.884, beta=1.057, gamma=0.556):
        self.norm = norm
        self.M1 = M1
        self.beta = beta
        self.gamma= gamma
    
    def __call__(self, Mh):
        Mh = 10.0**Mh
        M1 = self.M1
        norm = self.norm
        beta = self.beta
        gamma = self.gamma
        M1 = 10.0**M1
        mstar = 2.0*norm*((Mh/M1)**(-1.0*beta)+(Mh/M1)**gamma)**(-1.0)*Mh
        return np.log10(mstar)
    