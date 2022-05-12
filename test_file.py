import functions as funs
import sampler as samp

def test_prior_dist_prob_uniform():
    # Test inside boundaries of uniform region
    assert samp.prior_dist_prob([50,0,0,-25],'uniform')==1
    assert samp.prior_dist_prob([100,1,1,-15],'uniform')==1
    
    # Test outside boundaries of uniform region
    assert samp.prior_dist_prob([49,0.5,0.5,-20],'uniform')==0
    assert samp.prior_dist_prob([75,1.01,0.5,-20],'uniform')==0
    assert samp.prior_dist_prob([75,0.5,1.01,-20],'uniform')==0
    assert samp.prior_dist_prob([75,0.5,0.5,-14],'uniform')==0

def test_prior_dist_prob_gaussian():
    # Test peak and 1-sigma from peak
    assert samp.prior_dist_prob([75,0.5,0.5,-19.23],'gaussian')==1
    assert samp.prior_dist_prob([75,0.5,0.5,-19.23+0.042],'gaussian') == samp.prior_dist_prob([75,0.5,0.5,-19.23-0.042],'gaussian')
    
    # Test outside boundaries of 
    assert samp.prior_dist_prob([49,0.5,0.5,-19.23],'gaussian')==0
    assert samp.prior_dist_prob([75,1.01,0.5,-19.23],'gaussian')==0
    assert samp.prior_dist_prob([75,0.5,1.01,-19.23],'gaussian')==0

def test_len_params():
    data = funs.dataObject('tot')
    params_new = samp.metropolis_hastings(params_prev=[75,0.5,0.5,-20],sigmas=[0.7,0.05,0.05,0.2],data=data,prior='uniform')
    assert len(params_new)==4

def test_len_chain():
    data = funs.dataObject('tot')
    sigmas = [0.1,0.01,0.01,0.01]
    chain = samp.mcmc_sampler(data,10,sigmas,'uniform')
    assert len(chain)==10