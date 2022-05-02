#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 14:53:35 2022

@author: dradmin
"""
import pandas as pd
import numpy as np

class dataObject:
    '''
    Inputs:
        STR: path
        Relitive path to directory which has two data files. 
        Defaults to ./data/
        Can be set using the changePath method
        
    Attributes:
        Pandas DF: dataDf
        data frame with columns for zcmb, mb, and dmb. 40 rows
        
        covSys: np array
        40x40 systamatic covariance matrix
        
        covStat: np array
        40x40 diagonal statistical covariance matrix
        
        covTotal: np array
        40x40 total covariance matrix. Sum of covStat + covSys
        '''
        
    def __init__(self, path = ('./data/')):        
        '''
        Constructor method for dataObject class
        
        Imports data from two text files given in class (no pun intended).
        Data (including statistical error) is given in lcparam_DS17f.txt 
        and is stored as a Pandas DataFrame. Unused columns are dropped. 
        Systamatic covariance matrix is given in sys_DS17f.txt as a 1600 
        element list and is reshaped into a 40x40 numpy array
        '''
        self.path = path
        
        #import raw data and drop all unused params
        lcParam = pd.read_csv(self.path+'lcparam_DS17f.txt', delimiter= ' ')
        self.dataDf = lcParam.drop(['#name',
                               'zhel',
                               'dz',
                               'x1',
                               'dx1',
                               'color',
                               'dcolor',
                               '3rdvar',
                               'd3rdvar',
                               'cov_m_s',
                               'cov_m_c',
                               'cov_s_c',
                               'set',
                               'ra',
                               'dec',
                               'biascor',
                               'Unnamed: 19'], 
                              axis = 1)
        
        #import raw systamatic cov list. Drop first element 
        #(first element is 40, refering to size of matrix)
        sysRaw = np.loadtxt(self.path+'sys_DS17f.txt', skiprows = 1)
        #reshape 1600x1 list into 40x40 array
        self.covSys = sysRaw.reshape(40,40)
        
        #pull statastical error from dataDf and pack into 
        #diagonal elements of 40x40 matrix
        self.covStat = np.diag(self.dataDf['dmb'])
        
        #total cov is sum of statistical and systamatic 
        self.covTotal = self.covStat + self.covSys
    
    def changePath(self, path):
        '''
        changePath method used to change path (duh)
        Inputs:
            STR: path
            relative path you want the dataObject to import from
        '''
        self.path = path
        print('path is changed to', path)
    
