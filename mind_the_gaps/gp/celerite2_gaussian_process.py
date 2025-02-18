from typing import Callable, Union

import celerite2
import celerite2.jax
import jax
import jax.numpy as jnp
import numpyro

from mind_the_gaps.gp.gaussian_process import BaseGP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve

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
        seed: int = 0,
        bounds: dict = None,
    ):

        self.kernel_fn = kernel_fn
        self._lightcurve = lightcurve
        self.rng_key = jax.random.PRNGKey(seed)
        self.bounds = bounds
        self.mean = mean
        self.init_params = params
        self.setup_gp(self.init_params, self._lightcurve.times, fit=False)

    def setup_gp(self, params: jnp.array, t: jnp.array, fit: bool):
        kernel = self.kernel_fn(
            params=params, fit=fit, rng_key=self.rng_key, bounds=self.bounds
        )
        self.gp = celerite2.jax.GaussianProcess(kernel, mean=self.mean)
        self.gp.compute(t, yerr=self._lightcurve.dy, check_sorted=False)

    def negative_log_likelihood(self, params: jnp.array, fit=True):

        self.setup_gp(params=params, t=self._lightcurve.times, fit=fit)
        jax.debug.print("params:{}", params)
        nll_value = -self.gp.log_likelihood(self._lightcurve.y)
        jax.debug.print("nll: {}", nll_value)
        return nll_value

    def numpyro_model(self, t, y=None, params=None, fit=False):
        self.setup_gp(params, t, fit=fit)
        numpyro.sample("obs", self.gp.numpyro_dist(), obs=self._lightcurve.y)


#    def compute(self) -> None:  # , times: np.array, errors: np.array):

# self.gp.compute(self._lightcurve.times, yerr=self._lightcurve.dy + 1e-12)
