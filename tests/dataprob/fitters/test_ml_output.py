
import pytest
import numpy as np
import pandas as pd
from dataprob.fitters.ml import MLFitter

def linear_model(m, x):
    return m * x

def test_ml_output_reduced_chi_sq(capsys):
    # Setup data
    x = np.linspace(0, 10, 11)
    y_true = 2.0 * x
    # Add small noise so cost isn't exactly zero, avoiding division by zero weirdness if any
    y_obs = y_true + 0.1
    y_std = np.ones_like(y_obs) * 0.1
    
    # Initialize fitter
    fitter = MLFitter(linear_model, fit_parameters=["m"], non_fit_kwargs={"x": x})
    fitter.param_df = pd.DataFrame({"name": ["m"], "guess": [1.0], "lower_bound": [-10], "upper_bound": [10], "fixed": [False], "prior_mean": [0], "prior_std": [1.0]})
    
    # Run fit with verbose=2 to trigger our custom printing
    fitter.fit(y_obs=y_obs, y_std=y_std, verbose=2)
    
    captured = capsys.readouterr()
    
    # Check for our output
    assert "Reduced Chi-Sq:" in captured.out
    
    # Check for final output
    assert "Final Reduced Chi-Sq:" in captured.out
    
    # Ensure scipy output is suppressed (Cost is usually printed by scipy as 'Cost:')
    # Scipy verbose 2 prints 'Iteration', 'Cost', 'Cost reduction', 'Step norm', 'Optimality'
    # Our wrapper sets kwargs['verbose'] = 0 if verbose > 1 passed to fit()
    # But wait, we set kwargs['verbose'] = 0 *passed to scipy*.
    # So scipy shouldn't print anything.
    
    assert "Iteration" not in captured.out
    assert "Optimality" not in captured.out

