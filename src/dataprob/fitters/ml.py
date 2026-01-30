"""
Fitter subclass for performing maximum likelihood fits.
"""

from dataprob.fitters.base import Fitter
from dataprob.util.check import check_int
import traceback
import pandas as pd

import numpy as np
import scipy.stats
import scipy.optimize as optimize

import warnings

class MLFitter(Fitter):
    """
    Fit the model to the data using nonlinear least squares.

    Standard deviation and ninety-five percent confidence intervals on parameter
    estimates are determined using the covariance matrix (Jacobian * residual
    variance) 
    """
    
    def __init__(self, some_function, **kwargs):
        """
        Initialize the MLFitter.
        """
        # Store a direct reference to the original model object/function passed by the user.
        self._original_model_object = some_function
        super().__init__(some_function, **kwargs)


    def fit(self,
            y_obs=None,
            y_std=None,
            num_samples=100000,
            **least_squares_kwargs):
        """
        Fit the model parameters to the data by maximum likelihood.

        Parameters
        ----------
        y_obs : numpy.ndarray
            observations in a numpy array of floats that matches the shape
            of the output of some_function set when initializing the fitter. 
            nan values are not allowed. y_obs must either be specified here 
            or in the data_df dataframe. 
        y_std : numpy.ndarray
            standard deviation of each observation. nan values are not allowed.
            y_std must either be specified here or in the data_df dataframe. 
        num_samples : int
            number of samples for generating corner plot
        **least_squares_kwargs : 
            any remaining keyword arguments are passed as **kwargs to
            scipy.optimize.least_squares
        """
        
        self._num_samples = check_int(value=num_samples,
                                      variable_name="num_samples",
                                      minimum_allowed=0)

        super().fit(y_obs=y_obs,
                    y_std=y_std,
                    **least_squares_kwargs)                         

    def _fit(self,**kwargs):
        """
        Fit the parameters to the model.

        Parameters
        ----------
        kwargs : dict
            any keyword arguments are passed as **kwargs to
            scipy.optimize.least_squares
        """

        to_fit = self._model.unfixed_mask
        guesses = np.array(self._model.param_df.loc[to_fit,"guess"]).copy()
        bounds = np.array([self._model.param_df.loc[to_fit,"lower_bound"],
                           self._model.param_df.loc[to_fit,"upper_bound"]]).copy()
        
        # Do the actual fit
        verbose = kwargs.get("verbose", 0)
        
        # Intercept verbose to suppress scipy output and use our own
        if verbose > 1:
            kwargs["verbose"] = 0 # Turn off scipy printing
        
        def fn(*args): 
            res = self._weighted_residuals(*args)
            
            # Print progress if verbose
            if verbose > 1:
                chi2 = np.sum(res**2)
                N = len(self.y_obs)
                to_fit = self._model.unfixed_mask
                P = np.sum(to_fit)
                dof = N - P
                if dof > 0:
                    val = chi2 / dof
                    print(f"Reduced Chi-Sq: {val:.4e}", end="\r")
            
            return res
        
        fit_kwargs = kwargs.copy()

        model_to_check = self._original_model_object
        if hasattr(model_to_check, "__self__"):
            model_to_check = model_to_check.__self__

        if hasattr(model_to_check, "jacobian_normalized") and callable(model_to_check.jacobian_normalized):
            print("INFO: Analytical Jacobian found in the model. Using for optimization.")
            
            def jac_wrapper(unfixed_params):
                full_params = np.array(self.param_df["guess"], dtype=float)
                full_params[to_fit] = unfixed_params
                
                J_unweighted = model_to_check.jacobian_normalized(full_params)
                
                y_std_norm = model_to_check.y_std_normalized
                J_weighted = J_unweighted / y_std_norm[:, np.newaxis]

                return J_weighted[:, to_fit]

            fit_kwargs["jac"] = jac_wrapper
        
        try:
            self._fit_result = optimize.least_squares(fn,
                                                      x0=guesses,
                                                      bounds=bounds,
                                                      **fit_kwargs)
            
            # Print final cost
            if verbose > 0:
                cost = self._fit_result.cost # 0.5 * sum(residuals**2)
                N = len(self.y_obs)
                P = len(guesses)
                dof = N - P
                if dof > 0:
                    chi2 = 2 * cost
                    red_chi2 = chi2 / dof
                    print(f"Final Reduced Chi-Sq: {red_chi2:.4e}")
                else:
                    print(f"Final Cost: {cost:.4e} (dof <= 0)")
            self._success = self._fit_result.success

        except KeyboardInterrupt:
            print("Fit interrupted by user. Capturing last state.")
            if hasattr(self, "_fit_result"):
                self._success = False
            else:
                raise

        if hasattr(self,"_samples"):
            del self._samples
    
        self._update_fit_df()

    def _update_fit_df(self):
        """
        Recalculate the parameter estimates from any new samples.
        """
        
        if not hasattr(self, "_fit_result"):
            return

        estimate = self._fit_result.x

        N = len(self.y_obs)
        P = len(self._fit_result.x)

        try:
            J = self._fit_result.jac
            
            # The residuals are the final weighted residuals from the fit result
            residuals = self._fit_result.fun
            
            # Degrees of freedom
            dof = N - P
            if dof <= 0:
                raise ValueError("Degrees of freedom must be positive to calculate uncertainty.")

            # Reduced Chi-Squared (variance of the weighted residuals)
            reduced_chi_squared = np.sum(residuals**2) / dof
            
            # Correct covariance calculation for weighted least squares
            cov = np.linalg.inv(np.dot(J.T, J)) * reduced_chi_squared

            variances = np.diagonal(cov)
            if np.any(variances < 0):
                warnings.warn("\n\nCovariance matrix has negative diagonal elements, indicating non-identifiable parameters. Uncertainties cannot be calculated.\n\n")
                std = np.full(P, np.nan)
                low_95 = np.full(P, np.nan)
                high_95 = np.full(P, np.nan)
            else:
                std = np.sqrt(variances)
                # 95% confidence intervals from t-distribution
                z = scipy.stats.t(dof).ppf(0.975)
                low_95 = (estimate - z*std).tolist()
                high_95 = (estimate + z*std).tolist()

        except (np.linalg.LinAlgError, AttributeError, ValueError) as e:
            w = f"\n\nCould not calculate parameter uncertainty. Reason: {e}\n\n"
            warnings.warn(w)

            std = np.nan*np.ones(P,dtype=float)
            low_95 = np.nan*np.ones(P,dtype=float)
            high_95 = np.nan*np.ones(P,dtype=float)

        # Get finalized parameters from param_df in case they were updated 
        # after the model was set and the fit_df created. 
        for col in ["guess","fixed","lower_bound","upper_bound","prior_mean",
                    "prior_std"]:
            self._fit_df[col] = self.param_df[col]

        fixed = np.array(self._fit_df["fixed"],dtype=bool).copy()
        unfixed = np.logical_not(fixed)

        self._fit_df.loc[unfixed,"estimate"] = estimate
        self._fit_df.loc[fixed,"estimate"] = self._fit_df.loc[fixed,"guess"]
        self._fit_df.loc[unfixed,"std"] = std
        self._fit_df.loc[unfixed,"low_95"] = low_95
        self._fit_df.loc[unfixed,"high_95"] = high_95

        # Check for derived parameters (e.g. Physical params from GlobalModel)
        model_to_check = self._original_model_object
        if hasattr(model_to_check, "__self__"):
            model_to_check = model_to_check.__self__

        if hasattr(model_to_check, "calculate_derived_params") and hasattr(self, "_fit_result"):
            try:
                # Recalculate covariance for derived params
                J = self._fit_result.jac
                residuals = self._fit_result.fun
                dof = N - P
                if dof > 0:
                    reduced_chi_squared = np.sum(residuals**2) / dof
                    try:
                        cov = np.linalg.inv(np.dot(J.T, J)) * reduced_chi_squared
                    except np.linalg.LinAlgError:
                        cov = np.linalg.pinv(np.dot(J.T, J)) * reduced_chi_squared
                    # Expand cov to full parameters (with zeros for fixed)
                    # IMPORTANT: Only use original parameters (from param_df), ignoring 
                    # any derived params that might have been appended to fit_df already.
                    N_orig = len(self.param_df)
                    
                    # Get masks for ORIGINAL parameters
                    fixed_orig = np.array(self._fit_df["fixed"].iloc[:N_orig], dtype=bool)
                    unfixed_orig = np.logical_not(fixed_orig)
                    
                    full_cov = np.zeros((N_orig, N_orig))
                    # cov from scaler corresponds to unfixed_orig parameters
                    if cov.shape == (np.sum(unfixed_orig), np.sum(unfixed_orig)):
                        full_cov[np.ix_(unfixed_orig, unfixed_orig)] = cov
                    else:
                        # Fallback if shape mismatch (should not happen in standard flow)
                        warnings.warn(f"Covariance shape mismatch. Expected {np.sum(unfixed_orig)}x{np.sum(unfixed_orig)}, got {cov.shape}")
                    
                    # Get estimates for ORIGINAL parameters
                    full_estimate = self._fit_df["estimate"].iloc[:N_orig].values.astype(float)

                    derived_df = model_to_check.calculate_derived_params(estimate=full_estimate, cov=full_cov, dof=dof)
                    if derived_df is not None:
                        # Append to fit_df
                        # We use concat
                        # RESET fit_df to original parameters to avoid accumulation
                        self._fit_df = self._fit_df.iloc[:N_orig].copy()
                        self._fit_df = pd.concat([self._fit_df, derived_df])
            except Exception as e:
                traceback.print_exc()
                warnings.warn(f"Could not calculate derived parameters: {e}")


    @property
    def samples(self):
        """
        Use the Jacobian spit out by least_squares to generate a whole bunch of
        fake samples.

        Approximate the covariance matrix as $(2*J^{T} \\dot J)^{-1}$, then perform
        cholesky factorization on the covariance matrix.  This can then be
        multiplied by random normal samples to create distributions that come
        from this covariance matrix.

        See:
        https://stackoverflow.com/questions/40187517/getting-covariance-matrix-of-fitted-parameters-from-scipy-optimize-least-squares
        https://stats.stackexchange.com/questions/120179/generating-data-with-a-given-sample-covariance-matrix
        """

        # If we already have samples, return them
        if hasattr(self,"_samples"):
            return self._samples

        # Return None if no fit has been run.        
        if not self._fit_has_been_run:
            return None
                
        try:
            J = self._fit_result.jac
            
            # Use the same statistically correct covariance matrix as in _update_fit_df
            residuals = self._fit_result.fun
            dof = len(self.y_obs) - len(self._fit_result.x)
            if dof <= 0:
                raise ValueError("Degrees of freedom must be positive.")
            reduced_chi_squared = np.sum(residuals**2) / dof
            cov = np.linalg.inv(np.dot(J.T, J)) * reduced_chi_squared
            
            # Check for negative variance before cholesky decomposition
            if np.any(np.diagonal(cov) < 0):
                raise np.linalg.LinAlgError("Covariance matrix has negative diagonal elements.")

            chol_cov = np.linalg.cholesky(cov).T

        except (np.linalg.LinAlgError, AttributeError, ValueError):
            w = "\n\nJacobian matrix was singular or covariance matrix was invalid. Could not generate parameter samples.\n\n"
            warnings.warn(w)

            # Return empty array
            return None

        unfixed = np.logical_not(np.array(self.fit_df["fixed"],dtype=bool))
        estimate = np.array(self.fit_df.loc[unfixed,"estimate"]).copy()
        self._samples = np.dot(np.random.normal(size=(self._num_samples,
                                                      chol_cov.shape[0])),
                                                chol_cov)
    
        self._samples = self._samples + estimate
        
        num_param = self._samples.shape[1]

        # above_mask is True for a given sample if all of the parameter values
        # are >= the lower bound for that sample
        lower_bound = np.array(self.fit_df.loc[unfixed,"lower_bound"],dtype=float)
        above_mask = np.sum(self._samples >= lower_bound,axis=1) == num_param

        # below_mak is True for a given sample if all of the parameter values
        # are <= the upper bound for that sample
        upper_bound = np.array(self.fit_df.loc[unfixed,"upper_bound"],dtype=float)
        below_mask = np.sum(self._samples <= upper_bound,axis=1) == num_param

        # Keep mask is True only if above_mask and below_mask are true for a 
        # given sample
        keep_mask = np.logical_and(above_mask,below_mask)

        # Get only samples that fit the condition 
        self._samples = self._samples[keep_mask,:]

        return self._samples


    def __repr__(self):
        """
        Output to show when object is printed or displayed in a jupyter 
        notebook.
        """

        out = ["MLFitter\n--------\n"]

        out.append(f"fit has been run: {self._fit_has_been_run}\n")
        if self._fit_has_been_run:
            out.append(f"fit results:\n")
            # Check for success attribute, but also handle interrupted fits
            if hasattr(self, "_success") and self._success:
                status = "converged"
            elif hasattr(self, "_success") and not self._success:
                status = "failed or interrupted"
            else:
                status = "unknown"
            out.append(f"  fit status: {status}\n")

            # Always try to show the dataframe if it exists
            if hasattr(self, "_fit_df"):
                for dataframe_line in repr(self.fit_df).split("\n"):
                    out.append(f"  {dataframe_line}")
                out.append("\n")
            else:
                out.append("  fit dataframe not available\n")

        return "\n".join(out)