"""
Fit a two-parameter linear model to plausible data. 
"""

import pytest

import dataprob

import numpy as np
import matplotlib

def _core_test(method,**method_kwargs):
    
    # ------------------------------------------------------------------------
    # Define model and generate data

    def linear_model(m,b,x): 
        return m*x + b

    gen_params = {"m":-3,
                  "b":20}

    err = 0.25
    num_points = 20

    x = np.linspace(-5,5,num_points)
    y_obs = linear_model(x=x,**gen_params) + np.random.normal(0,err,num_points)
    y_std = 2*err
    
    test_fcn = linear_model
    non_fit_kwargs = {"x":x}

    # ------------------------------------------------------------------------
    # Run analysis

    f = dataprob.setup(some_function=test_fcn,
                       method=method,
                       non_fit_kwargs=non_fit_kwargs)

    f.fit(y_obs=y_obs,
          y_std=y_std,
          **method_kwargs)

    # make estimate lands between confidence intervals
    expected = np.array([gen_params[p] for p in f.fit_df.index])
    assert np.sum(expected < np.array(f.fit_df["low_95"])) == 0
    assert np.sum(expected > np.array(f.fit_df["high_95"])) == 0
    
    fig = dataprob.plot_summary(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)

    fig = dataprob.plot_corner(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)

def test_ml():
    
    _core_test(method="ml")

@pytest.mark.slow
def test_bayesian():

    _core_test(method="mcmc")

@pytest.mark.slow
def test_bootstrap():

    _core_test(method="bootstrap")
