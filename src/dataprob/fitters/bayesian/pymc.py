"""
Fitter subclass for performing bayesian (MCMC) parameter estimation using PyMC.
"""
from ..base import Fitter
from ...util.check import check_int
from ...util.stats import get_kde_max
import numpy as np
import warnings
import traceback

try:
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph.op import Op
    from pytensor.gradient import grad_not_implemented
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

class PyMCFitter(Fitter):
    """
    Use Bayesian MCMC via the PyMC library to sample parameter space.
    """

    def __init__(self, *args, **kwargs):
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC is not installed. Please install it via 'pip install pymc'")
        super().__init__(*args, **kwargs)

    def _fit(self, draws=1000, tune=1000, chains=4, target_accept=0.9, **pymc_kwargs):
        """Perform MCMC sampling using PyMC."""
        self._draws = check_int(draws, "draws", 1)
        self._tune = check_int(tune, "tune", 1)
        self._chains = check_int(chains, "chains", 1)

        class NumpyJacobianOp(Op):
            itypes = [pt.dvector]
            otypes = [pt.dmatrix]

            def __init__(self, jacobian_function):
                self.jacobian_function = jacobian_function

            def perform(self, node, inputs, output_storage):
                params_numpy = inputs[0]
                result = self.jacobian_function(params_numpy)
                output_storage[0][0] = np.asarray(result, dtype='float64')

        class NumpyModelOp(Op):
            itypes = [pt.dvector]
            otypes = [pt.dvector]

            def __init__(self, model_function, non_fit_kwargs, jacobian_function=None):
                self.model_function = model_function
                self.non_fit_kwargs = non_fit_kwargs
                self.jacobian_function = jacobian_function
                if self.jacobian_function:
                    self.jacobian_op = NumpyJacobianOp(self.jacobian_function)

            def perform(self, node, inputs, output_storage):
                params_numpy = inputs[0]
                result = self.model_function(params_numpy, **self.non_fit_kwargs)
                output_storage[0][0] = np.asarray(result, dtype='float64')

            def grad(self, inputs, output_grads):
                if not self.jacobian_function:
                    return [grad_not_implemented(self, 0, inputs[0])]
                params_vec = inputs[0]
                g_y_hat = output_grads[0]
                J = self.jacobian_op(params_vec)
                g_params = pt.dot(g_y_hat, J)
                return [g_params]
        
        has_jacobian = False
        jacobian_function = None
        
        fit_func = self._model._model_to_fit
        if hasattr(fit_func, "__self__"):
            original_object = fit_func.__self__
            if hasattr(original_object, "jacobian_normalized") and callable(original_object.jacobian_normalized):
                has_jacobian = True
                jacobian_function = original_object.jacobian_normalized

        numpy_model_op = NumpyModelOp(self._model._model_to_fit,
                                      self.non_fit_kwargs,
                                      jacobian_function=jacobian_function)

        with pm.Model() as model:
            params = {}
            unfixed_param_names = []
            for p_name in self.param_df.index:
                p_info = self.param_df.loc[p_name]
                if p_info["fixed"]:
                    params[p_name] = pt.as_tensor_variable(p_info["guess"])
                    continue
                
                unfixed_param_names.append(p_name)
                prior_mean, prior_std = p_info["prior_mean"], p_info["prior_std"]
                lower, upper = p_info["lower_bound"], p_info["upper_bound"]

                if not np.isnan(prior_mean) and not np.isnan(prior_std):
                    params[p_name] = pm.TruncatedNormal(p_name, mu=prior_mean, sigma=prior_std, lower=lower, upper=upper)
                else:
                    if np.isinf(lower) or np.isinf(upper):
                        raise ValueError(f"PyMC requires finite bounds for Uniform priors. Check parameter '{p_name}'.")
                    params[p_name] = pm.Uniform(p_name, lower=lower, upper=upper)

            all_params_in_order = [params[p] for p in self.param_df.index]
            full_symbolic_vector = pt.stack(all_params_in_order)
            y_hat = numpy_model_op(full_symbolic_vector)
            pm.Normal("obs", mu=y_hat, sigma=self._y_std, observed=self._y_obs)

            initvals = {
                p_name: self.param_df.loc[p_name, "guess"]
                for p_name in unfixed_param_names
            }
        
            try:
                sampler_kwargs = pymc_kwargs.copy()
                if not has_jacobian and "step" not in sampler_kwargs:
                    warnings.warn("WARNING: No analytical Jacobian available. Using the gradient-free Slice sampler. This may be slow.", UserWarning)
                    sampler_kwargs["step"] = pm.Slice()
                
                if has_jacobian:
                    warnings.warn("INFO: Analytical Jacobian found. Using NUTS sampler.", UserWarning)

                self._fit_result = pm.sample(draws=self._draws,
                                             tune=self._tune,
                                             chains=self._chains,
                                             initvals=initvals,
                                             log_likelihood=True,
                                             target_accept=target_accept,
                                             **sampler_kwargs)
                self._success = True

            except Exception as e:
                self._success = False
                warnings.warn(f"PyMC sampling failed: {e}")
                return

        if self._success:
            unfixed_params = self.param_df.index[self._model.unfixed_mask]
            posterior = self._fit_result.posterior.stack(sample=("chain", "draw"))
            sample_list = [posterior[p].values for p in unfixed_params]
            self._samples = np.stack(sample_list, axis=1)

            log_lik_data = self._fit_result.log_likelihood.stack(sample=("chain", "draw"))
            self._lnprob = log_lik_data["obs"].values

            self._update_fit_df()

    def _update_fit_df(self):
        """Update fit_df with results from the PyMC samples."""
        if self.samples is None or len(self.samples) == 0:
            return

        estimate = get_kde_max(self._samples)
        std = np.std(self._samples, axis=0)
        low_95, high_95 = np.quantile(self._samples, [0.025, 0.975], axis=0)

        for col in ["guess", "fixed", "lower_bound", "upper_bound", "prior_mean", "prior_std"]:
            self._fit_df[col] = self.param_df[col]

        unfixed = ~np.array(self._fit_df["fixed"], dtype=bool)
        self._fit_df.loc[unfixed, "estimate"] = estimate
        self._fit_df.loc[~unfixed, "estimate"] = self._fit_df.loc[~unfixed, "guess"]
        self._fit_df.loc[unfixed, "std"] = std
        self._fit_df.loc[unfixed, "low_95"] = low_95
        self._fit_df.loc[unfixed, "high_95"] = high_95

    @property
    def fit_info(self):
        """Information about the Bayesian run."""
        output = {"Backend": "PyMC"}
        if hasattr(self, "_draws"):
            output.update({
                "Draws": self._draws,
                "Tune steps": self._tune,
                "Num chains": self._chains
            })
        output["Final sample number"] = self.samples.shape[0] if self.samples is not None else None
        if hasattr(self, "_fit_result"):
            output["Steps taken"] = self._draws * self._chains
        return output