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

    params_init = np.array([rand.uniform(50,100),rand.uniform(0,1),rand.uniform(0,1),rand.uniform(-25,-15)])
    
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

    while params_new[1] < 0 or params_new[2] < 0:
        params_new = np.random.multivariate_normal(params_prev[0:4], np.diag(sigmas))

    g_t_new = params_prev[-1]    
    g_new_t = funs.loglike(params_new,data)
    params_new = np.append(params_new,g_new_t)

    accept_prob = np.min([1, g_t_new / g_new_t])
    
    if np.random.uniform() > accept_prob:
        return params_prev
    else:
        return params_new