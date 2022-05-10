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
import matplotlib.pyplot as plt
from getdist import MCSamples, plots

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

class make_plot:
    '''
    Inputs:
    -----------
    chains: a list of 2D array
        A list of the chains from MCMC runner
        For a single chain, each column is for one variable, each row is a record for a single step

    legend: a list of strings
        A list of the legend for each MCMC chains

    label: a list of strings
        A list of the label for each variable in the MCMC chains
        We assume that each chain has the same variables
    
    name: list of strings
        name of each variable in a chain
        We assume that each chain has the same variables
    
    burn_in: float
        fraction of the MCMC chains to remove due to burn_in
    
    Attributes:
    -----------
    samples: list of MCSample object of GetDist
        list of MCSample object of GetDist from the input chains
    
    name: list of strings
        name of each variable in the chains
    
    label: list of LaTex strings
        LaTex of each variable
    '''

    def __init__(self, chains, legend=None, label=None, name=None, burn_in=0.2):
        '''
        Constructor method for make_plot class
        '''

        if legend is None:
            self.legend = ['sample%i'%(i+1) for i in range(len(chains))]
        else:
            self.legend = legend

        if label is None:
            self.label = ['x_%i'%(i+1) for i in range(np.shape(chains[0])[1])]
        else:
            self.label = label
        
        if name is None:
            self.name = ['x%i'%(i+1) for i in range(np.shape(chains[0])[1])]
        else:
            self.name = name
        
        self.samples = []
        for i in range(len(chains)):
            self.samples.append(MCSamples(samples = chains[i], names=self.name, labels=self.label, ignore_rows = burn_in))
        
    def plot_2d(self, var_index=[0,1],accel_decel_line=False):
        '''
        Input:
        ---------------
        var_index: 1D list of int
            the index of the variables to plot
        '''
        i, j = var_index
        g = plots.get_single_plotter(width_inch = 10)
        g.settings.rc_sizes(axes_fontsize=16,lab_fontsize=28,legend_fontsize=16)
        g.plot_2d(self.samples, [self.name[i], self.name[j]])
        
        if accel_decel_line == True:
            g.plot_2d(self.samples, [self.name[i], self.name[j]],lims=[0,1,0,1])
            g.add_line(xdata=[0,2],ydata=[0,1],color='cyan',ls='--')
        else:
           g.plot_2d(self.samples, [self.name[i], self.name[j]])

        g.finish_plot(legend_labels=self.legend)
    
    def plot_1d(self, var_index=0):
        '''
        Input:
        ---------------
        var_index: int
            the index of the variable to plot
        '''
        g = plots.get_single_plotter(width_inch = 10)
        g.settings.rc_sizes(axes_fontsize=16,lab_fontsize=28,legend_fontsize=16)
        g.plot_1d(self.samples, self.name[var_index])

        g.finish_plot(legend_labels=self.legend)
