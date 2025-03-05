from typing import Callable, Union

import celerite2
import celerite2.jax
import jax
import jax.numpy as jnp
import numpyro

from mind_the_gaps.gp.gaussian_process import BaseGP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve
from mind_the_gaps.models.celerite2.mean_terms import (
    ConstantMean,
    GaussianMean,
    LinearMean,
)

# from mind_the_gaps.models.celerite2.kernel_terms import ConstantModel, Model
from mind_the_gaps.models.celerite.mean_models import (
    GaussianModel,
    LinearModel,
    SineModel,
)


class Celerite2GP(BaseGP):

    def __init__(
        self,
        kernel_fn: Callable,
        lightcurve: GappyLightcurve,
        mean: Union[float, Callable],
        params: jnp.array,
        rng_key: jnp.array,
        bounds: dict = None,
    ):

        self.kernel_fn = kernel_fn
        self._lightcurve = lightcurve
        self.rng_key = rng_key
        self.bounds = bounds
        self.mean_model, self.fit_mean = self._build_mean_model(mean)
        self.params = params
        self.compute(self.params, self._lightcurve.times, fit=True)

    def _build_mean_model(self, meanmodel: str):
        # if meanmodel is None:
        #    return self._lightcurve.mean, False
        if meanmodel is None:
            return ConstantMean(lightcurve=self._lightcurve), False

    def numpyro_dist(self):
        self.gp.numpyro_dist()

    def compute(self, params: jnp.array, t: jnp.array, fit: bool) -> None:
        self.params = params
        kernel, mean = self.kernel_fn(
            params=params,
            fit=fit,
            rng_key=self.rng_key,
            bounds=self.bounds,
            mean_model=self.mean_model,
        )
        self.gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
        self.gp.compute(t, yerr=self._lightcurve.dy, check_sorted=False)

    def get_psd(self):
        kernel = self.kernel_fn(
            params=self.params, fit=True, rng_key=self.rng_key, bounds=self.bounds
        )
        return kernel.get_psd

    def negative_log_likelihood(self, params: jnp.array, fit=True):
        self.compute(params, self._lightcurve.times, fit=fit)
        # jax.debug.print("params {params}", params=params)
        # Should this be y with value passed through?
        nll_value = -self.gp.log_likelihood(self._lightcurve.y)
        # jax.debug.print("nll_value: {nll_value}", nll_value=nll_value)
        return nll_value

    def get_parameter_vector(self):
        return self.params

    def set_parameter_vector(self, params: jnp.array):
        self.compute(params=params, t=self._lightcurve.times, fit=True)

    def log_likelihood(self, observations: jnp.array):
        return self.gp.log_likelihood(y=observations)

    def get_parameter_bounds(self):
        return self.bounds

    def log_prior(self):
        raise NotImplementedError

    def get_parameter_names(self):
        raise NotImplementedError
