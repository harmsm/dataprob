
from .fit_param import FitParameter

import numpy as np

import inspect

class ModelWrapper:
    """
    Wrap a model for use in likelihood calculations.

    The first N arguments with no default argument or arguments whose
    default can be coerced as a float are converted to fit parameters. The
    remaining arguments are treated as non-fittable parameters.  A specific
    set of arguments to convert to fit parameters can be specified by
    specifying 'fittable_params'.
    """

    # Attributes to hold the fit parameters and other arguments to pass
    # to the model. These have to be defined across class because we are going
    # to hijack __getattr__ and __setattr__ and need to look inside this as soon
    # as we start setting attributes.
    _mw_fit_parameters = {}
    _mw_other_arguments = {}

    def __init__(self,model_to_fit,fittable_params=None):
        """
        model_to_fit: a function or method to fit.
        fittable_params: list of arguments to fit.
        """

        # Define these here so __setattr__ and __getattr__ are looking at
        # instance-level attributes rather than class-level attributes.
        self._mw_fit_parameters = {}
        self._mw_other_arguments = {}

        self._model_to_fit = model_to_fit
        self._mw_load_model(fittable_params)

    def _mw_load_model(self,fittable_params):
        """
        Load a model into the wrapper, making the arguments into attributes.
        Fittable arguments are made into FitParameter instances.  Non-fittable
        arguments are set as generic attributes.
        """

        # model arguments
        self._mw_signature = inspect.signature(self._model_to_fit)

        # Figure out which arguments to make fittable.
        argument_gettable = []
        for p in self._mw_signature.parameters:

            # Make sure that this parameter name isn't already being used by
            # the wrapper.
            if p in dir(self.__class__):
                err = f"Parameter name '{p}' reserved by this class.\n"
                err += "Please change the argument name in your function.\n"
                raise ValueError(err)

            # See if this parameter can concievably be a fit parameter: either
            # no default or default can be coerced into a float
            try:
                np.float(self._mw_signature.parameters[p].default)
                argument_gettable.append(p)
            except (TypeError,ValueError):
                if self._mw_signature.parameters[p].default == inspect._empty:
                    argument_gettable.append(p)
                else:
                    argument_gettable.append(None)

        # If the fittable params were not specified by the user, take all params
        # up to the first one that was not gettable.
        if fittable_params is None:

            fittable_params = []
            for p in argument_gettable:
                if p is not None:
                    fittable_params.append(p)
                else:
                    break

        # Load arguments as either fit parameters or generic attributes
        for i, p in enumerate(self._mw_signature.parameters):

            if p in fittable_params:

                # Remove p from _fittable_params
                fittable_params.remove(p)

                # Make sure this argument was identified as one we could use
                # as a parameter.
                if not argument_gettable[i]:
                    err = f"default for function argument '{p}' cannot be\n"
                    err += "coerced into a float\n"
                    raise ValueError(err)

                # Grab guess from default argument
                if self._mw_signature.parameters[p].default == inspect._empty:
                    guess = None
                else:
                    guess = np.float(self._mw_signature.parameters[p].default)

                # Convert to a fit parameter
                self._mw_fit_parameters[p] = FitParameter(name=p,guess=guess)

            else:
                # Record as a generic argument, not a fit parameter.
                self._mw_other_arguments[p] = self._mw_signature.parameters[p].default

        # Check to make sure we saw all of the specified fittable_params
        if len(fittable_params) > 0:
            err = f"Specified parameter(s) are not arguments to the model:\n\n:"
            err += "   ({})".format(",".join(fittable_params))
            raise ValueError(err)


    def __setattr__(self, key, value):
        """
        Hijack __setattr__ so setting the value for fit parameters
        updates the fit guess.
        """

        # We're setting the guess of the fit parameter
        if key in self._mw_fit_parameters.keys():
            self._mw_fit_parameters[key].guess = value

        # We're setting another argument
        elif key in self._mw_other_arguments.keys():
            self._mw_other_arguments[key] = value

        # Otherwise, just set it like normal
        else:
            super(ModelWrapper, self).__setattr__(key, value)

    def __getattr__(self,key):

        # We're getting a fit parameter
        if key in self._mw_fit_parameters.keys():
            return self._mw_fit_parameters[key]

        # We're getting another argument
        if key in self._mw_other_arguments.keys():
            return self._mw_other_arguments[key]

        # Otherwise, get like normal
        else:
            super(ModelWrapper,self).__getattribute__(key)


    def _update_parameter_map(self):
        """
        Update the map between the parameter vector that will be passed in to
        the fitter and the parameters in this wrapper. This
        """

        self._param_to_p_map = []
        self._mw_kwargs = {}
        for p in self._mw_fit_parameters.keys():
            if self._mw_fit_parameters[p].fixed:
                self._mw_kwargs[p] = self.fit_parameters[p].value
            else:
                self._mw_kwargs[p] = None
                self._param_to_p_map.append(p)

        self._mw_kwargs.update(self.other_arguments)


    def _mw_observable(self,params=None):
        """
        Actual function called by the fitter.
        """

        # If parameters are not passed, stick in the current parameter
        # values
        if params is None:
            for i in range(len(self._param_to_p_map)):
                p = self._param_to_p_map[i]
                self._mw_kwargs[p] = self.fit_parameters[p].value
        else:
            if len(params) != len(self._param_to_p_map):
                err = f"Number of fit parameters ({len(params)}) does not match\n"
                err += f"number of unfixed parameters ({len(self._param_to_p_map)})\n"
                raise ValueError(err)

            for i in range(len(params)):
                self._mw_kwargs[self._param_to_p_map[i]] = params[i]

        return self._model_to_fit(**self._mw_kwargs)


    @property
    def model(self):
        """
        Return the observable.
        """

        # Update the parameter map in case the user changed a fit parameter
        # since this was last returned.
        self._update_parameter_map()

        # This model, once returned, does not have to re-run update_parameter_map
        # and should thus be faster when run again and again in regression
        return self._mw_observable

    @property
    def guesses(self):
        """
        Return an array of the guesses for the model (only including the unfixed
        parameters).
        """

        # Update the parameter map in case the user changed a fit parameter
        # since this was last returned.
        self._update_parameter_map()

        guesses = []
        for p in self._param_to_p_map:
            guesses.append(self.fit_parameters[p].guess)

        return np.array(guesses)

    @property
    def bounds(self):
        """
        Return an array of the bounds for the model (only including the unfixed
        parameters).
        """

        # Update the parameter map in case the user changed a fit parameter
        # since this was last returned.
        self._update_parameter_map()

        bounds = [[],[]]
        for p in self._param_to_p_map:
            bounds[0].append(self.fit_parameters[p].bounds[0])
            bounds[1].append(self.fit_parameters[p].bounds[1])

        return np.array(bounds)

    @property
    def names(self):
        """
        Return an array of the names of the parameters (only including the unfixed
        parameters).
        """

        # Update the parameter map in case the user changed a fit parameter
        # since this was last returned.
        self._update_parameter_map()

        names = []
        for p in self._param_to_p_map:
            names.append(self.fit_parameters[p].name)

        return names[:]

    @property
    def fit_parameters(self):
        """
        A dictionary of FitParameter instances.
        """

        return self._mw_fit_parameters

    @property
    def other_arguments(self):
        """
        A dictionary with every model argument that is not a fit parameter.
        """

        return self._mw_other_arguments
