import numpy as np
from astropy.cosmology import LambdaCDM
import astropy.units as u

def model(z, params):
    '''
    calculate the apparent magnitude from a model

    Paramters:
    --------------------
    z: float or 1D array
        The redshift(s) of the observations
    params: 1D array
        cosmological parameters and nuiansence parameters
        params = [H0, Om_matter, Om_lambda, M]
        The order of parameters is [Hubble constant, matter density, dark energy density, absolute magnitude]
    
    Return:
    --------------------
    mb_th: float or 1D array
        the theoretical apparent magnitude of the SNe targets given the redshifts under a certain parameters
    '''

    # pass parameters to the variables
    h0, omm, oml, M = params

    # initiate the cosmology
    cosmo = LambdaCDM(H0=h0*u.km/u.s/u.Mpc, Om0 = omm, Ode0=oml)

    dmoduli = np.array(cosmo.distmod(z))

    return dmoduli + M

