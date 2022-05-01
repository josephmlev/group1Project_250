#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 14:53:35 2022

@author: dradmin
"""


import pandas as pd
import numpy as np
def importData (path = ('./data/')):
    '''
    '''
    #import raw data 
    lcParam = pd.read_csv(path+'lcparam_DS17f.txt', delimiter= ' ')
    #drop all unused params
    dataDf = lcParam.drop(['#name',
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
    
    #import raw systamatic cov list. Drop first element (40, refering to size of matrix)
    sysRaw = np.loadtxt(path+'sys_DS17f.txt', skiprows = 1)
    #reshape 1600x1 list into 40x40 array
    covSys = sysRaw.reshape(40,40)
    
    #pull statastical error from dataDf and pack into diagonal elements of 40x40 matrix
    covStat = np.diag(dataDf['dmb'])
    
    covTotal = covStat + covSys
    return(dataDF, covSys, covStat, covTotal)

