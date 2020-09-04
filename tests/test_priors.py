
# TEST priors classes here
# Test priors setters and getters in test_fit_param, test_model_wrapper, test_base
# and test_integration

# HAVE PRIORS GET PASSED IN AT .fit(priors=None,XXX)

import likelihood
import numpy as np

def test_uninformative():

    p = likelihood.priors.UninformativePrior()
    #assert p.ln_prior(1) == 0
    #assert p.ln_prior(np.inf) == 0

def test_exponential():

    p = likelihood.priors.ExponentialPrior()
    print(p.ln_prior(0))
    assert np.isclose(p.ln_prior(0),0)

    #assert np.isclose(p.ln_prior(1),-1)

    #p = likelihood.priors.ExponentialPrior(loc=1,scale=2)
    #assert np.isclose(p.ln_prior(0),0)
    #assert np.isclose(p.ln_prior(1),-1)

def test_beta():

    p = likelihood.priors.BetaPrior(a=0.5,b=0.5)
    #assert np.isinf(p.ln_prior(0))
    #assert np.isinf(p.ln_prior(1))
    #assert np.isclose(p.ln_prior(0.5),-0.45158270528945466)
