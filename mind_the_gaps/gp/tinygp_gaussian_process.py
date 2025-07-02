from typing import Callable, Dict, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import tinygp
from jax import jit, vmap
from tinygp.kernels.quasisep import Celerite

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


class TinyGP(BaseGP):
    """Wrapper round the TinyGP Gaussian Process to handle compute, building the mean,
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
        """Initialise the celerite2 Gaussian Process

        Parameters
        ----------
        kernel_spec : KernelSpec
            KernelSpec object that defines the kernel terms and stores param values & bounds.
        lightcurve : GappyLightcurve
            The lightcurve for the Gaussian process.
        rng_key : jnp.array
            _description_
        meanmodel : str
            Mean model for the gaussian process either "Constant", "Linear" or "Gaussian", by default None
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

        self.compute_fit(self._lightcurve.times, params=init_params)

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

    def numpyro_dist(self):
        return self.gp.numpyro_dist()

    def compute_fit(
        self, t: jax.Array, params: jax.Array = None, log_like: bool = False
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
        kernel = self.kernel_spec._get_jax_kernel_fit()
        self.gp = tinygp.GaussianProcess(
            kernel=kernel,
            X=t,
            diag=self._lightcurve.dy**2 + 1e-12,
            mean=mean,
        )
        if log_like:
            return self.gp.log_probability(self._lightcurve.y)

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

        kernel = self.kernel_spec._get_jax_kernel_sample()
        self.gp = tinygp.GaussianProcess(
            kernel=kernel, X=t, diag=self._lightcurve.dy**2, mean=mean
        )

    def compute(
        self, t: jax.Array, fit: bool, params: jax.Array = None, jitter: float = 1e-6
    ) -> None:
        """Set up the TinyGP kernel, GaussianProcess and call compute on it with
        the appropriate params.

        Parameters
        ----------
        params : jnp.array
            Parameter values for the kernel
        t : jnp.array
            Times for the lightcurve observations

        fit : bool
            Whether the GP is being fitted
        params : jnp.array, optional
            Parameters to use for the Gaussian Process, by default None
        jitter : float, optional
            Jitter term to add to the diagonal of the covariance matrix, by default 1e-6
        """
        if fit:
            mean_params = params[: self.meanmodel.sampled_parameters]
            kernel_params = params[self.meanmodel.sampled_parameters :]
            self.kernel_spec.update_params_from_array(kernel_params)
            mean = self.meanmodel.compute_mean(params=mean_params)
        else:
            mean = self.meanmodel.compute_mean()

        kernel = self._get_kernel(fit=fit)
        self.gp = tinygp.GaussianProcess(
            kernel=kernel,
            X=t,
            diag=self._lightcurve.dy**2 + jitter,
            mean=mean,
        )

    def _get_kernel(self, fit=True):

        terms = []
        for i, term in enumerate(self.kernel_spec.terms):
            kwargs = {
                name: (
                    -1.0e-10
                    if param_spec.zeroed
                    else jnp.exp(
                        param_spec.value
                        if (fit or param_spec.fixed)
                        else numpyro.sample(
                            f"terms[{i}]:log_{name}",
                            param_spec.prior(*param_spec.bounds),
                        )
                    )
                )
                for name, param_spec in term.parameters.items()
            }
            terms.append(term.term_class(**kwargs))

        kernel = terms[0]
        for t in terms[1:]:
            kernel += t
        return kernel

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
        self.compute_fit(self._lightcurve.times, params=params)
        nll_value = -self.gp.log_probability(y=self._lightcurve.y)

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
        self.compute_fit(params=params, t=self._lightcurve.times)

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
        return self.gp.log_probability(y=observations)

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

    def standardized_residuals(self, include_noise=True):
        """Returns the standardized residuals (see e.g. Kelly et al. 2011) Eq. 49.
        You should set the gp parameters to your best or mean (median) parameter values prior to calling this method.

        Parameters
        ----------
        include_noise: bool
            True to include any jitter term into the standard deviation calculation. False ignores this contribution.
        """

        _, cond = self.gp.condition(self._lightcurve.y, self._lightcurve.times)
        pred_mean = cond.mean
        pred_var = cond.variance

        if include_noise:

            jitter_var = getattr(self.gp.kernel, "jitter", 0.0)
            pred_var = pred_var + jitter_var

        std_res = (self._lightcurve.y - pred_mean) / jnp.sqrt(pred_var)
        return std_res

    def predict(self, y, **kwargs) -> tuple:
        """Compute the conditional predictive distribution of the model by calling celerite's predict method.

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
            return_var. See https://celerite.readthedocs.io/en/stable/python/gp/#celerite.GP.predict.

        """
        return self.gp.predict(y, **kwargs)
