import pprint as pp
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


@jit
def get_psd_value_jax(
    alpha_real,
    beta_real,
    alpha_complex_real,
    alpha_complex_imag,
    beta_complex_real,
    beta_complex_imag,
    omega,
):
    w2 = omega**2
    p = 0.0

    # Real terms
    p += jnp.sum(alpha_real * beta_real / (beta_real**2 + w2))

    # Complex terms
    w02 = beta_complex_real**2 + beta_complex_imag**2
    num = (
        alpha_complex_real * beta_complex_real + alpha_complex_imag * beta_complex_imag
    ) * w02 + (
        alpha_complex_real * beta_complex_real - alpha_complex_imag * beta_complex_imag
    ) * w2
    denom = w2**2 + 2.0 * (beta_complex_real**2 - beta_complex_imag**2) * w2 + w02**2
    p += jnp.sum(num / denom)

    return jnp.sqrt(2.0 / jnp.pi) * p


# Vectorized for an array of omega
get_psd_jax = vmap(get_psd_value_jax, in_axes=(None, None, None, None, None, None, 0))


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

        self.gp = self.compute(self._lightcurve.times, fit=True, params=init_params)

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

    def compute(self, t: jax.Array, fit: bool, params: jax.Array = None) -> None:
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
        """
        if fit:
            mean_params = params[: self.meanmodel.sampled_parameters]
            kernel_params = params[self.meanmodel.sampled_parameters :]
            self.kernel_spec.update_params_from_array(kernel_params)
            mean = self.meanmodel.compute_mean(fit=fit, params=mean_params)
        else:
            mean = self.meanmodel.compute_mean(fit=fit)

        kernel = self._get_kernel(fit=fit)
        self.gp = tinygp.GaussianProcess(
            kernel=kernel,
            X=t,
            diag=self._lightcurve.dy**2,
            mean=mean,
        )
        # return gp

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

    def get_psd(self, freq_in_hz=True) -> np.ndarray:
        """Get the power spectral density for the kernel at the current parameters

        Returns
        -------
        np.ndarray
            Power spectral density
        """
        kernel = self.kernel_spec.get_kernel(fit=True)

        return self.get_psd_from_kernel(kernel, freq_in_hz=freq_in_hz)

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

    def get_psd_from_kernel(self, kernel, freq_in_hz=True, real_eps=1e-5):
        """
        Build a callable PSD function from a tinygp Celerite kernel.

        Args:
            kernel: A tinygp kernel (sum of Celerite terms)
            freq_in_hz: If True, expects frequencies in Hz and converts to omega
            real_eps: Threshold for deciding whether b and d are "zero" (real term)

        Returns:
            A function psd(omega) or psd(f) → PSD values
        """
        if hasattr(kernel, "terms"):
            terms = kernel.terms
        else:
            terms = [kernel]

        alpha_real = []
        beta_real = []
        alpha_complex_real = []
        alpha_complex_imag = []
        beta_complex_real = []
        beta_complex_imag = []

        for term in terms:
            if not isinstance(term, tinygp.kernels.quasisep.Celerite):
                raise NotImplementedError(f"Unsupported kernel type: {type(term)}")

            a = term.a
            b = term.b
            c = term.c
            d = term.d

            if jnp.abs(b) < real_eps and jnp.abs(d) < real_eps:
                alpha_real.append(a)
                beta_real.append(c)
            else:
                alpha_complex_real.append(a)
                alpha_complex_imag.append(b)
                beta_complex_real.append(c)
                beta_complex_imag.append(d)

        alpha_real = jnp.array(alpha_real)
        beta_real = jnp.array(beta_real)
        alpha_complex_real = jnp.array(alpha_complex_real)
        alpha_complex_imag = jnp.array(alpha_complex_imag)
        beta_complex_real = jnp.array(beta_complex_real)
        beta_complex_imag = jnp.array(beta_complex_imag)

        def psd_fn(freqs):
            omega = 2 * jnp.pi * freqs if freq_in_hz else freqs
            return get_psd_jax(
                alpha_real,
                beta_real,
                alpha_complex_real,
                alpha_complex_imag,
                beta_complex_real,
                beta_complex_imag,
                omega,
            )

        return psd_fn
