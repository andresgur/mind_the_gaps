import pprint as pp
from typing import Callable, Dict, List, Union

import celerite2
import celerite2.jax
import celerite2.jax.terms as j_terms
import jax
import jax.numpy as jnp
import numpy as np
import numpyro

from mind_the_gaps.gp.gaussian_process import BaseGP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve
from mind_the_gaps.models.celerite2.mean_terms import (
    ConstantMean,
    FixedMean,
    GaussianMean,
    LinearMean,
)

# from mind_the_gaps.models.celerite2.kernel_terms import ConstantModel, Model
from mind_the_gaps.models.celerite.mean_models import (
    GaussianModel,
    LinearModel,
    SineModel,
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
        rng_key: jnp.array,
        bounds: dict = None,
        meanmodel: str = None,
        mean_params: jax.Array = None,
    ):
        """Initialise the celerite2 Gaussian Process

        Parameters
        ----------
        kernel_fn : Callable
            Numpyro Kernel funcion that returns a Celerite2 Term
        lightcurve : GappyLightcurve
            The lightcurve for the Gaussian process
        params : jnp.array
            _description_
        rng_key : jnp.array
            _description_
        bounds : dict, optional
            _description_, by default None
        mean : str
            Mean model for the gaussian process either "Constant", "Linear" or "Gaussian", by default None
        """

        self.kernel_spec = kernel_spec
        self._lightcurve = lightcurve
        self.rng_key = rng_key
        # self.bounds = bounds
        self.mean_params = mean_params
        self.meanmodel, self.fit_mean = self._build_mean_model(meanmodel, mean_params)

        init_params = jnp.concatenate([mean_params, self.kernel_spec.get_param_array()])
        self.compute(self._lightcurve.times, fit=True, params=init_params)

    def _build_mean_model(
        self, meanmodel: str, mean_params: jax.Array
    ) -> Union[FixedMean, ConstantMean, GaussianMean, LinearMean, float]:
        """Return the mean model based on meanmodel. Can either be a

        Parameters
        ----------
        meanmodel : str
            Mean model to use, : "Constant", "Linear", "Gaussian" defaults to a fixed parameter of the mean if None
        Returns
        -------
        Union[ConstantMean,LinearMean, GaussianMean, float]
            Mean numpyro model for the kernel, can be a ConstantMean, LinearMean, GaussianMean or a float
        """

        if meanmodel is None or meanmodel.lower() == "fixed":
            return (
                FixedMean(lightcurve=self._lightcurve),
                False,
            )
        elif meanmodel.lower() == "constant":
            return (
                ConstantMean(lightcurve=self._lightcurve),
                True,
            )
        elif meanmodel.lower() == "linear":
            return (
                LinearMean(lightcurve=self._lightcurve),
                True,
            )
        elif meanmodel.lower() == "gaussian":
            return GaussianMean(lightcurve=self._lightcurve), True
        else:
            raise ValueError(
                f"Invalid mean model specified: '{meanmodel}'. Must be None, 'fixed', 'constant', 'linear', or 'gaussian'."
            )

    def numpyro_dist(self):
        return self.gp.numpyro_dist()

    def compute(self, t: jax.Array, fit: bool, params: jax.Array = None) -> None:
        """Set up the Celerite2 kernel, GaussianProcess and call compute on it with
        the appropriate params.

        Parameters
        ----------
        params : jnp.array
            Parameter values for the kernel
        t : jnp.array
            Times for the lightcurve observations

        fit : bool
            Whether the GP is being fitted
        """
        # self.params = params
        # kernel, mean = self.kernel_fn(
        #    params=params,
        #    fit=fit,
        #    rng_key=self.rng_key,
        #    bounds=self.bounds,
        #    mean_model=self.meanmodel,
        # )
        #
        #
        #
        if fit:
            mean_params = params[: self.meanmodel.no_parameters]
            kernel_params = params[self.meanmodel.no_parameters :]
            self.kernel_spec.update_params_from_array(kernel_params)
            mean = self.meanmodel.compute_mean(fit=fit, params=mean_params)
        else:
            mean = self.meanmodel.compute_mean(fit=fit, rng_key=self.rng_key)

        kernel = self._get_kernel(fit=fit)

        self.gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
        self.gp.compute(
            t,
            yerr=self._lightcurve.dy,
            check_sorted=False,
        )

    def _get_kernel(self, fit=True):

        rng_key = self.rng_key
        terms = []

        for i, term in enumerate(self.kernel_spec.terms):
            kwargs = {}

            for name, param_spec in term.parameters.items():
                full_name = f"terms[{i}]:log_{name}"

                if fit or param_spec.fixed:
                    val = param_spec.value
                else:
                    dist_cls = param_spec.prior
                    val = numpyro.sample(
                        full_name, dist_cls(*param_spec.bounds), rng_key=rng_key
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
        kernel = self._get_kernel(fit=True)

        return kernel.get_psd

    def negative_log_likelihood(self, params: jax.Array, fit=True) -> float:
        """Calcultae the negtaive log likelihood

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
        self.compute(self._lightcurve.times, params=params, fit=fit)
        nll_value = -self.gp.log_likelihood(self._lightcurve.y)

        return nll_value

    def get_parameter_vector(self) -> jax.Array:
        """Get the parameters for the GP

        Returns
        -------
        jax.Array
            Gaussian process parameters
        """
        return self.params

    def set_parameter_vector(self, params: jax.Array):
        """Sets the parameter vector, by setting up the kernel and calling compute

        Parameters
        ----------
        params : jnp.array
            Parameters to compute the GP for.
        """
        self.compute(params=params, t=self._lightcurve.times, fit=True)

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

        return self.kernel_spec.get_param_names()


def get_kernel(rng_key, kernel_spec, fit=False):

    terms = []

    for i, term in enumerate(kernel_spec.terms):
        kwargs = {}

        for name, param_spec in term.parameters.items():
            full_name = f"term{i}_{name}"

            if fit or param_spec.fixed:
                kwargs[name] = param_spec.value
            else:
                dist_cls = param_spec.prior

                val = numpyro.sample(
                    full_name, dist_cls(*np.log(param_spec.bounds)), rng_key=rng_key
                )
                kwargs[name] = np.exp(val)

        terms.append(term.term_class(**kwargs))

    kernel = terms[0]
    for t in terms[1:]:
        kernel += t
    return kernel
