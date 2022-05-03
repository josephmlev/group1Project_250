import numpy as np
import numpy.random as rand
import functions as funs

def mcmc_sampler(data, t_max):
    '''
    Create a Markov chain for cosmological parameters
    
    Parameters:
    --------------------
    data: dataObject
        Cosmological data, including redshift (z) and covariance matrix

    t_max: int
        Length of Markov chain
    
    Return:
    --------------------
    params_chain: 2D array
        List of parameter values calculated by MCMC. Dimensions are 4 x t_max.
        params = [H0, Om_matter, Om_lambda, M]
        The order of parameters is [Hubble constant, matter density, dark energy density, absolute magnitude]
    '''

    # Initial parameter distribution is uniform over:
    # H_0 = [50,100]
    # Omega_m = [0,1]
    # Omega_lambda = [0,1]
    # M = [-25,-15] 


    #params_init = np.array([rand.uniform(50,100),rand.uniform(0,1),rand.uniform(0,1),rand.uniform(-25,-15)])
    
    params_init = [50,0.01,0.01,-25]

    # Log likelihood is associated with parameter sample to avoid calculating multiple times
    params_init = np.append(params_init,funs.loglike(params_init,data))
    
    # These variances can be tuned for better convergence
    params_sigma = [0.1, 0.001, 0.001, 0.1]

    # Building the chain of parameter sets for t_max timesteps
    params_chain = np.array([params_init])

    for _ in np.arange(0,t_max):

        params_new = metropolis_hastings(params_chain[-1],params_sigma,data)
        params_chain = np.insert(params_chain,-1,params_new,axis=0)
        
    return params_chain[:,0:4]

def metropolis_hastings(params_prev,sigmas,data):
    '''
    Create new parameter sample and decide whether to accept or reject
    
    Parameters:
    --------------------
    params_prev: 1D array
        Previous set of parameters [0:4] and their log likelihood [4] 

    sigmas: 1D array
        Variance for each cosmological parameter [0:4]

    data: dataObject
        Cosmological data, including redshift (z) and covariance matrix
    
    Return:
    --------------------
    new_params: 1D array
        List of new set of parameters [0:4] and their log likelihood [4]
    '''

    params_new = np.random.multivariate_normal(params_prev[0:4], np.diag(sigmas))

    print(params_new)

    while params_new[1] < 0 or params_new[2] < 0:
        params_new = np.random.multivariate_normal(params_prev[0:4], np.diag(sigmas))

    P_t = params_prev[-1]    
    P_new = funs.loglike(params_new,data)
    params_new = np.append(params_new,P_new)

    g_t_new = generating_function(params_prev[0:4],params_new[0:4],sigmas,data)
    g_new_t = generating_function(params_new[0:4],params_prev[0:4],sigmas,data)

    accept_prob = np.min([1, P_new / P_t * g_t_new / g_new_t])
    
    if np.random.uniform() > accept_prob:
        print('reject')
        return params_prev
    else:
        print('accept')
        return params_new

def generating_function(params_prev,params_new,sigmas,data):
    '''
    Generating function for cosmological data
    
    Parameters:
    --------------------
    params_prev: 1D array
        Previous set of parameters [0:3] and their log likelihood [4] 

    params_new: 1D array
        New set of parameters under consideration [0:3] and their log likelihood [4] 

    sigmas: 1D array
        Variance for each cosmological parameter [0:3]

    data: dataObject
        Cosmological data, including redshift (z), which is needed for model
    
    Return:
    --------------------
    new_params: 1D array
        List of new set of parameters [0:3] and their log likelihood [4]
    '''

    return(1.0)
    #params_new = np.random.Generator.multivariate_normal(params_prev[0:4], data.cov)