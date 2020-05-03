
import numpy as np
from scipy import stats

import warnings

class ContinuousPrior:
    """
    Prior with a continuous function.
    """

    def __init__(self,distrib,*args,**kwargs):
        """
        distrib: continuous distribution from stats.norm.  Examples include
                 stats.norm, stats.gamma, stats.chi2, etc. See:

                 https://docs.scipy.org/doc/scipy/reference/stats.html
                 https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html

        *args: arguments used to generate a frozen RV object from the
               distribution.  Examples include scale, loc, df, etc.  The
               details depend on the distribution.
        **kwargs: keyword arguments used to generate a frozen RV object from
                  the distribution.  Examples include scale, loc, df, etc.
                  The details depend on the distribution.
        """

        # Make sure this is a continuous variable
        if not isinstance(distrib,stats.rv_continuous):
            err = "distrib must be a continuous distribution from scipy.stats\n"
            err += "(Formally: an instance of the stats.rv_continuous class).\n\n"
            err += "https://docs.scipy.org/doc/scipy/reference/stats.html"
            raise ValueError(err)

        # Create a frozen continuous RV object
        try:
            self._frozen = distrib(*args,**kwargs)
        except Exception as e:
            err = "\n\nThe continuous distribution threw an error during\n"
            err += "initialization (see trace).\n\n"
            raise type(e)(err) from e

        # Get interval with 100% of the pdf
        interval = self._frozen.interval(1)

        # If one or both of the interval values is infinty, grab a chunk
        # with *most* of the density.
        if np.isinf(interval[0]) or np.isinf(interval[1]):
            interval = self._frozen.interval(1-(1e-8))

        # If we have a nan in our interval, we can't normalize the prior.
        # Throw a warning.
        if np.isnan(interval[0]) or np.isnan(interval[1]):
            self._normalization = 1.0
            self._log_normalization = 0
            warning = "could not normalize prior pdf\n"
            warnings.warn(warning)

        # Integrate under the prior to create a log normalization
        else:
            steps = np.linspace(*interval,int(1e6))
            self._normalization = np.sum(a.pdf(steps)*(steps[1]-steps[0]))
            self._log_normalization = np.log(self._normalization)


    def ln_prior(self,p):
        """
        Return ln_prior of parameter p.
        """

        return self._frozen.logpdf(p) - self._log_normalization


class GaussianPrior(ContinuousPrior):
    """
    Prior following a gaussian distribution.
    """

    def __init__(self,loc=0,scale=1,*args,**kwargs):
        """
        loc: mean of the distribution
        scale: standard deviation of the distribution

        For args and kwargs, see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
        """

        super(GaussianPrior,self).__init__(distrib=stats.norm,
                                           loc=loc,scale=scale,
                                           *args,**kwargs)

class Chi2Prior(ContinuousPrior):
    """
    Prior following a chi2 distribution.
    """

    def __init__(self,df,loc=0,scale=1,*args,**kwargs):
        """
        df: degrees of freedom.
        loc: location of distribution
        scale: scale of distribution

        For args and kwargs, see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
        """

        super(Chi2Prior,self).__init__(distrib=stats.chi2,
                                       df=df,loc=loc,scale=scale,
                                       *args,**kwargs)

class BetaPrior(ContinuousPrior):
    """
    Prior following a beta distribution.
    """

    def __init__(self,a,b,loc=0,scale=1,*args,**kwargs):
        """
        a: shape parameter
        b: shape parameter
        loc: location of distribution
        scale: scale of distribution

        For args and kwargs, see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
        """

        super(BetaPrior,self).__init__(distrib=stats.beta,
                                       a=a,b=b,loc=loc,scale=scale,
                                       *args,**kwargs)

class ExponentialPrior(ContinuousPrior):
    """
    Prior following a exponential distribution.
    """

    def __init__(self,loc=0,scale=1,*args,**kwargs):
        """
        loc: location of distribution
        scale: scale of distribution

        For args and kwargs, see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html
        """

        super(ExponentialPrior,self).__init__(distrib=stats.expon,
                                              loc=loc,scale=scale,
                                              *args,**kwargs)

class UniformPrior:
    """
    Uniform prior.
    """

    def ln_prior(self,p):

        return 0.0
