from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve


class MeanFunction(ABC):
    """Base class for mean functions in Celerite2 Gaussian Processes."""

    def __init__(self, lightcurve: jnp.array):
        self._lightcurve = lightcurve

    @abstractmethod
    def compute_mean(self):
        """Compute the mean function value based on the provided parameters.
        Should be implemented by subclasses.
        """
        pass


class FixedMean(MeanFunction):
    """Fixed mean function: m(t) = mean(y)
    This class does not sample parameters and uses the mean of the lightcurve directly.
    """

    no_parameters = 1
    sampled_parameters = 0

    def __init__(self, lightcurve):
        """Initialize the FixedMean with the lightcurve data.

        Parameters
        ----------
        lightcurve : GappyLightcurve
            The lightcurve data to compute the mean from.
        """
        super().__init__(lightcurve=lightcurve)
        self.mean_value = lightcurve.mean
        self.bounds = jnp.array([])
        self.sampled_mean = False

    def compute_mean(self, fit=None, params: jnp.array = None, rng_key: int = None):
        """Return the fixed mean value of the lightcurve.

        Parameters
        ----------
        fit : bool, optional
            Whether the mean parameters are being fitted during optimisation, by default None
        params : jax.Array, optional
            Parameters to use for the mean function if being fitted for optimisation, by default None
        rng_key : int, optional
            Random number generator key for sampling, by default None

        Returns
        -------
        jnp.array
            The fixed mean value of the lightcurve.
        """
        return self.mean_value


class ConstantMean(MeanFunction):
    """Constant mean function: m(t) = c where mean can be sampled."""

    no_parameters = 1
    sampled_parameters = 1
    param_names = ["mean"]

    def __init__(self, lightcurve: GappyLightcurve):
        """Initialize the ConstantMean with the lightcurve data.

        Parameters
        ----------
        lightcurve : GappyLightcurve
            The lightcurve data to compute the mean from.
        """
        super().__init__(lightcurve=lightcurve)
        self.bounds = jnp.array(
            [[jnp.min(self._lightcurve.y)], [jnp.max(self._lightcurve.y)]]
        )
        self.sampled_mean = True

    def compute_mean(
        self,
        params: jnp.array = None,
        fit: bool = True,  # rng_key: int = None
    ) -> jnp.array:
        """Return the constant mean value of the lightcurve.

        Parameters
        ----------
        params : jnp.array, optional
            parameters to use for the mean function, by default None
        fit : bool, optional
            Whether the mean parameters are being fitted during optimisation, by default True
        rng_key : int, optional
            Random number generator key for sampling, by default None

        Returns
        -------
        jnp.array

        """
        if fit and jnp.size(params) == self.no_parameters:

            return params
        else:
            return numpyro.sample(
                "mean",
                dist.Uniform(jnp.min(self._lightcurve.y), jnp.max(self._lightcurve.y)),
                # rng_key=rng_key,
            )


class LinearMean(MeanFunction):
    """Linear mean function: m(t) = mt + b"""

    param_names = ["m", "b"]
    no_parameters = 2

    def __init__(self, lightcurve: GappyLightcurve):
        """Initialize the LinearMean with the lightcurve data.

        Parameters
        ----------
        lightcurve : GappyLightcurve
            The lightcurve data to compute the mean from.
        """
        super().__init__(lightcurve=lightcurve)
        self.bounds = jnp.array(
            [[jnp.min(self._lightcurve.y)], [jnp.max(self._lightcurve.y)]]
        )
        self.sampled_mean = True

    def compute_mean(
        self,
        # rng_key: int,
        params: jnp.array = None,
        fit: bool = True,
        bounds: dict = None,
    ):
        """Return the linear mean value of the lightcurve

        Parameters
        ----------
        rng_key : int
            Random number generator key for sampling.
        params : jnp.array, optional
            Parameters to use for the mean function if being fitted for optimisation, by default None
        fit : bool, optional
            Whether the mean parameters are being fitted during optimisation, or sampled, by default True
        bounds : dict, optional
            Parameter bounds for the mean function, by default None

        Returns
        -------
        jnp.array
            The computed mean value at the times of the lightcurve.
        """

        if fit and jnp.size(params) == self.no_parameters:
            m, b = params  # [: self.count_parameters]

        else:
            m = numpyro.sample(
                "m",
                dist.Uniform(bounds["mean_params"][0][0], bounds["mean_params"][0][1]),
                # rng_key=subkey,
            )
            rng_key, subkey = jax.random.split(rng_key)
            b = numpyro.sample(
                "b",
                dist.Uniform(bounds["mean_params"][1][0], bounds["mean_params"][1][1]),
                # rng_key=subkey,
            )
        return m * self._lightcurve.t + b


class GaussianMean(MeanFunction):
    """Gaussian mean function: m(t) = A * exp(- (t - mu)^2 / (2 * sigma^2))"""

    param_names = ["A", "mu", "sigma"]
    no_parameters = 3

    def __init__(self, lightcurve: GappyLightcurve):
        """Initialize the GaussianMean with the lightcurve data.

        Parameters
        ----------
        lightcurve : GappyLightcurve
            The lightcurve data to compute the mean from.
        """
        super().__init__(lightcurve=lightcurve)
        self.bounds = jnp.array(
            [[jnp.min(self._lightcurve.y)], [jnp.max(self._lightcurve.y)]]
        )
        self.sampled_mean = True

    def compute_mean(
        self,
        t: jnp.array,
        # rng_key: int,
        fit: bool = True,
        params: jnp.array = None,
        bounds: dict = None,
    ):
        """Return the Gaussian mean value of the lightcurve.

        Parameters
        ----------
        t : jnp.array
            The times at which to compute the mean function.
        rng_key : int
            Random number generator key for sampling.
        fit : bool, optional
            Whether the mean parameters are being fitted during optimisation, or sampled, by default True
        params : jnp.array, optional
            Parameters to use for the mean function if being fitted for optimisation, by default None
        bounds : dict, optional
            Parameter bounds for the mean function, by default None

        Returns
        -------
        jnp.array
            The computed mean value at the specified times.
        """
        if fit and jnp.size(params) == self.no_parameters:
            A, mu, sigma = params

        else:
            A = numpyro.sample(
                "amplitude",
                dist.Uniform(bounds["mean_params"][0][0], bounds["mean_params"][0][1]),
                # rng_key=rng_key,
            )
            mu = numpyro.sample(
                "mu",
                dist.Uniform(bounds["mean_params"][1][0], bounds["mean_params"][1][1]),
                # rng_key=rng_key,
            )
            sigma = numpyro.sample(
                "sigma",
                dist.Uniform(bounds["mean_params"][2][0], bounds["mean_params"][2][1]),
                # rng_key=rng_key,
            )
        return A * jnp.exp(-((t - mu) ** 2) / (2 * sigma**2))
