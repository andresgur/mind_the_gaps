import copy
import pprint as pp
from typing import Callable, Dict, List, Union

import celerite2.jax
import celerite2.jax.terms as j_terms
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from celerite2.jax.distribution import CeleriteNormal

from mind_the_gaps.gp.gaussian_process import BaseGP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve
from mind_the_gaps.models.celerite2.mean_terms import (
    ConstantMean,
    FixedMean,
    GaussianMean,
    LinearMean,
)
from mind_the_gaps.models.kernel_spec import KernelSpec


class Celerite2GP(BaseGP):
    """Wrapper round the Celerite2 Gaussian Process to handle compute, building the mean,
    returning parameters names etc.

    Inherits from:
    ---------------
    BaseGP : Base class for Gaussian Process models
    """

    def __init__(
        self,
        kernel_spec: KernelSpec,
        lightcurve: GappyLightcurve,
        meanmodel: str = None,
        mean_params: jax.Array = None,
    ):
        """Initialize the Celerite2 Gaussian Process with a kernel specification,
        lightcurve, and random number generator key. It sets up the mean model based on
        the provided meanmodel and mean_params. If the meanmodel is None, it defaults to
        a fixed mean model. The kernel parameters are initialized from the kernel_spec.

        Parameters
        ----------
        kernel_spec : KernelSpec
            Specification of the kernel to use, containing the terms and their parameters.
        lightcurve : GappyLightcurve
            The lightcurve data to fit the Gaussian Process to.
        rng_key : jnp.array
            Random number generator key for sampling.

        meanmodel : str, optional
            The type of mean model to use, can be "Constant", "Linear", "Gaussian" or "Fixed", defaults to "Fixed".
        mean_params : jax.Array, optional
            Parameters for the mean model, if applicable. If None, defaults to a fixed mean based on the lightcurve.
        """

        self.kernel_spec = kernel_spec
        self._lightcurve = lightcurve
        self.mean_params = mean_params
        self.meanmodel = self._build_mean_model(meanmodel, mean_params)
        if self.meanmodel.sampled_mean:
            init_params = jnp.concatenate(
                [mean_params, self.kernel_spec.get_param_array()]
            )
        else:
            init_params = self.kernel_spec.get_param_array()
        self.gp = self.compute_fit(self._lightcurve.times, params=init_params)

    def _build_mean_model(
        self, meanmodel: str, mean_params: jax.Array
    ) -> Union[FixedMean, ConstantMean, GaussianMean, LinearMean, float]:
        """Return the mean model based on meanmodel. Can either be a

        Parameters
        ----------
        meanmodel : str
            Mean model to use, : "Constant", "Linear", "Gaussian", "Fixeddefaults to a fixed parameter of the mean if None
        Returns
        -------
        Union[ConstantMean,LinearMean, GaussianMean, FixedMean]
            Mean numpyro model for the kernel, can be a ConstantMean, LinearMean, GaussianMean or a float
        Raises
        ------
        ValueError
            If the meanmodel is not one of the accepted options.
        """

        if meanmodel is None or meanmodel.lower() == "fixed":
            return FixedMean(lightcurve=self._lightcurve)

        elif meanmodel.lower() == "constant":
            return ConstantMean(lightcurve=self._lightcurve)

        elif meanmodel.lower() == "linear":
            return LinearMean(lightcurve=self._lightcurve)

        elif meanmodel.lower() == "gaussian":
            return GaussianMean(lightcurve=self._lightcurve)
        else:
            raise ValueError(
                f"Invalid mean model specified: '{meanmodel}'. Must be None, 'fixed', 'constant', 'linear', or 'gaussian'."
            )

    def numpyro_dist(self) -> CeleriteNormal:
        """Get the numpyro distribution for the Gaussian Process.

        Returns
        -------
        CeleriteNormal
            Numpyro distribution for the Gaussian Process.
        """
        return self.gp.numpyro_dist()

    def compute_fit(
        self,
        t: jax.Array,
        params: jax.Array = None,
    ) -> None:
        """Set up the Celerite2 kernel, GaussianProcess and call compute on it with
        the appropriate params.

        Parameters
        ----------
        params : jnp.array
            Parameter values for the kernel
        t : jnp.array
            Times for the lightcurve observations

        fit : bool
            Whether the GP is being fitted (i.e. during parameter optiization) or sampled.
        """

        mean_params = params[: self.meanmodel.sampled_parameters]
        kernel_params = params[self.meanmodel.sampled_parameters :]
        self.kernel_spec.update_params_from_array(kernel_params)
        mean = self.meanmodel.compute_mean(params=mean_params)
        kernel = self.kernel_spec._get_celerite2_kernel_fit()
        self.gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
        self.gp.compute(
            t,
            yerr=self._lightcurve.dy,
            check_sorted=False,
        )

    def compute_sample(
        self,
        t: jax.Array,
    ) -> None:
        """Sample the parameters, set up the Celerite2 kernel, GaussianProcess and call compute on it with
        the appropriate params.

        Parameters
        ----------
        params : jnp.array
            Parameter values for the kernel
        t : jnp.array
            Times for the lightcurve observations

        fit : bool
            Whether the GP is being fitted (i.e. during parameter optiization) or sampled.
        """
        mean = self.meanmodel.sample_mean()
        kernel = self.kernel_spec._get_celerite2_kernel_sample()
        self.gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
        self.gp.compute(
            t,
            yerr=self._lightcurve.dy,
            check_sorted=False,
        )

    def _get_kernel(self, fit=True) -> j_terms.Term:
        """Get the kernel for the Gaussian Process based on the kernel specification.

        Parameters
        ----------
        fit : bool, optional
            Whether the parameters are being fit or sampled, by default True

        Returns
        -------
        j_terms.Term
            The Celerite2 kernel term constructed from the kernel specification.

        Raises
        ------
        ValueError
            If a parameter in the kernel specification is not fixed and does not have a prior distribution.
        """

        terms = []

        for i, term in enumerate(self.kernel_spec.terms):
            kwargs = {}

            for name, param_spec in term.parameters.items():
                full_name = f"terms[{i}]:log_{name}"
                if not param_spec.fixed and param_spec.prior is None:
                    raise ValueError(
                        f"Missing prior distribution for parameter '{full_name}'"
                    )
                if fit or param_spec.fixed:
                    val = param_spec.value

                else:
                    dist_cls = param_spec.prior
                    val = numpyro.sample(
                        full_name,
                        dist_cls(*param_spec.bounds),
                    )

                kwargs[name] = jnp.exp(val)
            terms.append(term.term_class(**kwargs))

        kernel = terms[0]
        for t in terms[1:]:
            kernel += t
        return kernel

    def get_psd(self) -> np.ndarray:
        """Get the power spectral density for the kernel at the current parameters

        Returns
        -------
        np.ndarray
            Power spectral density
        """
        kernel = self.kernel_spec.get_kernel(fit=True)

        return kernel.get_psd

    def negative_log_likelihood(self, params: jax.Array, fit=True) -> float:
        """Calculate the negtaive log likelihood.

        Parameters
        ----------
        params : jnp.array
            Parameters for the Gaussian Process
        fit : bool, optional
            Whether the parameters are being fit or sampled, by default True

        Returns
        -------
        float
            Negative Log Likelihood
        """
        self.compute_fit(self._lightcurve.times, params=params)
        nll_value = -self.gp.log_likelihood(self._lightcurve.y)
        return nll_value

    def get_parameter_vector(self) -> jax.Array:
        """Get the parameters for the GP

        Returns
        -------
        jax.Array
            Gaussian process parameters
        """
        return self.kernel_spec.get_param_array()

    def set_parameter_vector(self, params: jax.Array):
        """Sets the parameter vector, by setting up the kernel and calling compute

        Parameters
        ----------
        params : jnp.array
            Parameters to compute the GP for.
        """
        self.compute_fit(params=params, t=self._lightcurve.times, fit=True)

    def log_likelihood(self, observations: jax.Array) -> float:
        """Get the log likelihood of the GP model

        Parameters
        ----------
        observations : jnp.array
            The observation for which to compute the log likelihood

        Returns
        -------
        float
            Log likelihood of the model
        """
        return self.gp.log_likelihood(y=observations)

    def get_parameter_bounds(self) -> Dict:
        """Returns the bounds on the kernel parameters.

        Returns
        -------
        Dict
            A dict containing the bounds on the parameters.
        """

        return self.kernel_spec.get_bounds_array()

    def log_prior(self):
        raise NotImplementedError

    def get_parameter_names(self) -> List[str]:
        """Get the names on the kernel parameters.

        Returns
        -------
        [str]
            A list containing the kernel parameter names.
        """
        if self.meanmodel.sampled_mean:
            param_names = (
                self.meanmodel.param_names + self.kernel_spec.get_param_names()
            )
        else:
            param_names = self.kernel_spec.get_param_names()
        return param_names

    def standarized_residuals(self, include_noise=True):
        """Returns the standarized residuals (see e.g. Kelly et al. 2011) Eq. 49.
        You should set the gp parameters to your best or mean (median) parameter values prior to calling this method

        Parameters
        ----------
        include_noise: bool,
            True to include any jitter term into the standard deviation calculation. False ignores this contribution.
        """
        pred_mean, pred_var = self.gp.predict(
            self._lightcurve.y, return_var=True, return_cov=False
        )
        if include_noise:
            pred_var += self.gp.kernel.jitter
        std_res = (self._lightcurve.y - pred_mean) / jnp.sqrt(pred_var)
        return std_res

    def predict(self, y: jax.Array, **kwargs) -> Union[tuple, jax.Array]:
        """Compute the conditional predictive distribution of the model by calling celerite2's predict method.

        Parameters
        ----------
        y : np.ndarray
            Observations at the coordinates of the lightcurve times.
        **kwargs : dict
            Additional keyword arguments to pass to the celerite predict method.
        Returns
        ------

        tuple
            mu, (mu, cov), or (mu, var) depending on the values of return_cov and
            return_var. See https://github.com/exoplanet-dev/celerite2/blob/main/python/celerite2/core.py

        """
        return self.gp.predict(y, **kwargs)

    def standarized_residuals(self, include_noise=True):
        """Returns the standarized residuals (see e.g. Kelly et al. 2011) Eq. 49.
        You should set the gp parameters to your best or mean (median) parameter values prior to calling this method

        Parameters
        ----------
        include_noise: bool,
            True to include any jitter term into the standard deviation calculation. False ignores this contribution.
        """
        pred_mean, pred_var = self.gp.predict(
            self._lightcurve.y, return_var=True, return_cov=False
        )

        # if include_noise:
        #    ....

        std_res = (self._lightcurve.y - pred_mean) / jnp.sqrt(pred_var)
        return std_res
