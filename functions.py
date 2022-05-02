#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 14:53:35 2022

@author: dradmin
"""
import pandas as pd
import numpy as np
from astropy.cosmology import LambdaCDM
import astropy.units as u

class dataObject:
    '''
    Inputs:
    --------------------   
    covType: string
        Type of covariance matrix to be computed. Can be 'sys' to only include
        systamatics or 'tot' to include systamatic + statistical matricies
        
    path = ('./data/'): string
        Relitive path to directory which has two data files. 
        Defaults to ./data/
        Can be set using the changePath method
        
    Attributes:
    --------------------
        Pandas DF: dataDf
        data frame with columns for zcmb, mb, and dmb. 40 rows
        
        zcmb: np 1D array
        redshift data.
        
        mb: np 1D array
        luminosity measurement.
        
        dmb np 1D array
        error on luminosity measurement.
        
        covSys: np 2D array
        40x40 systamatic covariance matrix
        
        covStat: np 2D array
        40x40 diagonal statistical covariance matrix
        
        cov: np 2D array
        40x40 total covariance matrix. Either total (Sum of covStat + covSys)
        or covStat depending on covType
        
        covInv: np 2D array
        40x40 inverse of covariance matrix
        '''
        
    def __init__(self, covType, path = ('./data/')):        
        '''
        Constructor method for dataObject class
        
        Imports data from two text files given in class (no pun intended).
        Data (including statistical error) is given in lcparam_DS17f.txt 
        and is stored as a Pandas DataFrame. Unused columns are dropped. 
        Systamatic covariance matrix is given in sys_DS17f.txt as a 1600 
        element list and is reshaped into a 40x40 numpy array
        '''
        

        if covType != 'sys' and covType != 'tot':
            raise TypeError('covType must = "sys" or "tot"')
        
        self.covType = covType
        self.path = path
        
        #import raw data and drop all unused params
        self.dataDf = pd.read_csv(self.path+'lcparam_DS17f.txt',
                              delimiter= ' ')
        
        #set useful params to attributes as np arrays (type float32)
        self.zcmb = np.asarray(self.dataDf['zcmb'],
                               dtype = np.float32)
        self.mb = np.asarray(self.dataDf['mb'],
                             dtype = np.float32)
        self.dmb = np.asarray(self.dataDf['dmb'],
                              dtype = np.float32)        

        #import raw systamatic cov list. Drop first element 
        #(first element is 40, refering to size of matrix)
        sysRaw = np.loadtxt(self.path+'sys_DS17f.txt',
                            skiprows = 1,
                            dtype = np.float32 )
        #reshape 1600x1 list into 40x40 array
        self.covSys = sysRaw.reshape(40,40)
        
        #pull statastical error from dataDf and pack into 
        #diagonal elements of 40x40 matrix
        self.covStat = np.diag(self.dmb)**2
        
        #total cov is sum of statistical and systamatic 
        self.cov = self.covStat
        if (self.covType == 'sys'):
            self.cov = self.covStat + self.covSys
            
        self.covInv = np.linalg.inv(self.cov)
    
    

    def changePath(self, path):
        '''
        changePath method used to change path (duh)
        Inputs:
            STR: path
            relative path you want the dataObject to import from
        '''
        self.path = path
        print('path is changed to', path)
        
    def changeCov(self, covType):
        if covType != 'sys' and covType != 'tot':
            raise TypeError('covType must = "sys" or "tot"')
        else:
            self.covType = covType
    
def loglike(params, data):
    '''
    Computes log likliehood from parameter array
    
    Paramters:
    --------------------
    params: 1D array
        cosmological parameters and nuiansence parameters
        params = [H0, Om_matter, Om_lambda, M]
        The order of parameters is [Hubble constant, matter density, dark 
        energy density, absolute magnitude]
        
    data: dataObject
        obbject containing data as attributes 
        
    Return:
    --------------------
    chi2: float32 
    sum over trials of -((data - model)_i * cov_ij * (data - model)_j)/2. 
    Computed as matrix multplication 
        
    
    '''
    m = model(data.zcmb, params)
    deltaM =  data.mb - m
    chi2 = deltaM @ data.covInv @ deltaM
    return(-chi2/2)



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
