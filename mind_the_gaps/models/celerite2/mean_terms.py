from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


class MeanFunction(ABC):
    """Base class for all mean functions."""

    def __init__(self, lightcurve: jnp.array):
        self._lightcurve = lightcurve

    @abstractmethod
    def compute_mean(self, params: jnp.array, fit: bool, rng_key: int):
        """Compute the mean based on the function form and parameters."""
        pass


class FixedMean(MeanFunction):
    no_parameters = 1

    def __init__(self, lightcurve):
        super().__init__(lightcurve=lightcurve)
        self.mean_value = lightcurve.mean
        self.bounds = jnp.array([])

    def compute_mean(self, params: jax.Array = None, rng_key: int = None):
        return self.mean_value


class ConstantMean(MeanFunction):
    """Constant mean function: m(t) = c"""

    no_parameters = 1

    def __init__(self, lightcurve):
        super().__init__(lightcurve=lightcurve)
        self.bounds = jnp.array(
            [[jnp.min(self._lightcurve.y)], [jnp.max(self._lightcurve.y)]]
        )

    def compute_mean(self, params: jnp.array, rng_key: int):
        if params:
            return params[: self.no_parameters]
        else:
            return numpyro.sample(
                "mean",
                dist.Uniform(jnp.min(self._lightcurve.y), jnp.max(self._lightcurve.y)),
                rng_key=rng_key,
            )


class LinearMean(MeanFunction):
    """Linear mean function: m(t) = mt + b"""

    @classmethod
    def count_parameters(cls):
        return 2

    def __init__(self, lightcurve):
        super().__init__(lightcurve=lightcurve)
        self.bounds = jnp.array(
            [[jnp.min(self._lightcurve.y)], [jnp.max(self._lightcurve.y)]]
        )

    def compute_mean(self, rng_key: int, params: jnp.array = None, bounds: dict = None):
        if params:
            m, b = params[: self.count_parameters]

        else:
            m = numpyro.sample(
                "m",
                dist.Uniform(bounds["mean_params"][0][0], bounds["mean_params"][0][1]),
                rng_key=rng_key,
            )
            b = numpyro.sample(
                "b",
                dist.Uniform(bounds["mean_params"][1][0], bounds["mean_params"][1][1]),
                rng_key=rng_key,
            )
        return m * self.t + b


class GaussianMean(MeanFunction):
    """Gaussian mean function: m(t) = A * exp(- (t - mu)^2 / (2 * sigma^2))"""

    @classmethod
    def count_parameters(cls):
        return 3

    def compute_mean(
        self, t: jnp.array, rng_key: int, params: jnp.array = None, bounds: dict = None
    ):
        if params:
            A, mu, sigma = params[: self.count_parameters]

        else:
            A = numpyro.sample(
                "amplitude",
                dist.Uniform(bounds["mean_params"][0][0], bounds["mean_params"][0][1]),
                rng_key=rng_key,
            )
            mu = numpyro.sample(
                "mu",
                dist.Uniform(bounds["mean_params"][1][0], bounds["mean_params"][1][1]),
                rng_key=rng_key,
            )
            sigma = numpyro.sample(
                "sigma",
                dist.Uniform(bounds["mean_params"][2][0], bounds["mean_params"][2][1]),
                rng_key=rng_key,
            )
        return A * jnp.exp(-((t - mu) ** 2) / (2 * sigma**2))
