import time
from typing import Dict, List

import jax
import jax.numpy as jnp
import jaxopt
import numpyro

from mind_the_gaps.engines.gp_engine import BaseGPEngine
from mind_the_gaps.gp.celerite2_gaussian_process import Celerite2GP

# from mind_the_gaps.gp.tinygp_gaussian_process import TinyGP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve
from mind_the_gaps.models.kernel_spec import KernelSpec


class BaseNumpyroGPEngine(BaseGPEngine):
    posterior_params = {
        "fit",
        "max_steps",
        "num_chains",
        "num_warmup",
        "converge_steps",
        "progress",
    }

    def __init__(
        self,
        kernel_spec: KernelSpec,
        lightcurve: GappyLightcurve,
        meanmodel: str = None,
        mean_params: jax.Array = None,
        seed: int = 0,
        fit_mean: bool = True,
    ):
        """Base class for the Numpyro Gaussian Process Engine to define the
        shared methods for Jax/Numpyro based engine implementations.,

        Parameters
        ----------
        BaseGPEngine : _base class for Gaussian Process engines
        kernel_spec : KernelSpec
            Specification of the kernel to use, containing the terms and their parameters.
        lightcurve : GappyLightcurve
            The lightcurve data to fit the Gaussian Process to.
        meanmodel : str, optional
            The type of mean model to use, can be "Constant", "Linear", "Gaussian" or "Fixed", defaults to None.
        mean_params : jax.Array, optional
            Parameters for the mean model, if applicable. If None, defaults to a fixed mean based on the lightcurve.
        seed : int, optional
            Random seed for reproducibility, by default 0.
        fit_mean : bool, optional
            Whether to fit the mean model parameters, by default True.
        """

        self.kernel_spec = kernel_spec
        self._lightcurve = lightcurve
        self.seed = seed
        self.meanmodel = meanmodel
        self.fit_mean = fit_mean
        if self.meanmodel is None or self.meanmodel.lower() == "fixed":
            self.fit_mean = False

        self.mean_params = mean_params
        if seed is None:
            seed = int(time.time())

        self.seed = seed
        self.rng_key = jax.random.PRNGKey(self.seed)
        self.rng_key, subkey = jax.random.split(self.rng_key)

        numpyro.enable_x64()
        numpyro.set_platform("cpu")

        self.init_params = self.kernel_spec.get_param_array()
        if self.fit_mean:
            self.init_params = jnp.concatenate([mean_params, self.init_params])

        self._mcmc_samples = {}
        self._autocorr = []
        self.gp = None

    def numpyro_model(
        self,
        t: jax.Array,
        params: jax.Array | None = None,
        fit: bool = False,
    ) -> None:
        """Numpyro model for the Gaussian Process, which defines the prior distributions (Using the KernelSpec)
        and the likelihood of the model given the lightcurve data. This method computes

        Parameters
        ----------
        t : jax.Array
            The time array of the lightcurve.
        params : jax.Array, optional
            parameters for the Gaussian Process, by default None
        fit : bool, optional
            whether the model is being fitted, i.e. during optimisation, or sampled, by default False
        """

        self.gp.compute(t, params=params, fit=fit)

        log_likelihood = self.gp.log_likelihood(self._lightcurve.y)
        numpyro.deterministic("log_likelihood", log_likelihood)
        numpyro.sample(
            "obs",
            self.gp.numpyro_dist(),
            obs=self._lightcurve.y,
        )

    def minimize(self) -> jax.Array:
        """Minimize the negative log likelihood of the Gaussian Process using jaxopt L-BFGS-B method.
        This method optimises the parameters of the Gaussian Process to fit the lightcurve data.
        It uses the bounds defined in the kernel specification to ensure that the parameters remain within valid ranges.
        The method returns the optimised parameters as a jax Array.
        If the mean model is being fitted, it includes the mean model parameters in the optimisation.
        The optimisation is performed using the ScipyBoundedMinimize solver from jaxopt.
        The method also updates the Gaussian Process with the optimised parameters.

        Returns
        -------
        jax.Array
            The optimised parameters of the Gaussian Process after minimization.
        """

        bounds = self.kernel_spec.get_bounds_array()
        upper_bounds = jnp.array([values[1] for values in bounds])
        lower_bounds = jnp.array([values[0] for values in bounds])
        if self.fit_mean and self.meanmodel != "fixed":
            upper_bounds = jnp.concatenate([self.gp.meanmodel.bounds[1], upper_bounds])
            lower_bounds = jnp.concatenate([self.gp.meanmodel.bounds[0], lower_bounds])

        solver = jaxopt.ScipyBoundedMinimize(
            method="l-bfgs-b",
            fun=self.gp.negative_log_likelihood,
            maxiter=1000,
        )

        opt_params, res = solver.run(
            init_params=self.init_params,  # self.gp.params,
            bounds=(lower_bounds, upper_bounds),
        )
        self.gp.compute(params=opt_params, t=self._lightcurve.times, fit=True)

        return opt_params

    def initialise_params(self, num_chains: int, perc: float) -> Dict:
        """Initialise the parameters for the Gaussian Process by spreading
        the initial values around their base values using a normal distribution.
        This method generates a dictionary of parameters for each chain, where each parameter is perturbed
        by a small random value drawn from a normal distribution centered around the base value.

        Parameters
        ----------
        num_chains : int
            The number of chains to initialise parameters for.
        perc : float
            Percentage for the normal distribution used to spread the parameters, by default 0.1.

        Returns
        -------
        Dict
            A dictionary where keys are parameter names and values are jax arrays of spread values for each chain.
        """
        param_dict = {}
        i = 0
        for term in self.kernel_spec.terms:
            for param_name, param_spec in term.parameters.items():
                if param_spec.fixed:
                    continue

                base_value = param_spec.value
                low, high = param_spec.bounds

                spread_values = base_value + perc * jax.random.normal(
                    self.rng_key, shape=(num_chains,)
                )
                spread_values = jnp.clip(spread_values, float(low), float(high))

                param_dict[f"term{i}_{param_name}"] = jnp.array(spread_values)
            i += 1
        return param_dict

    @property
    def parameter_names(self) -> List[str]:
        """Return the names of the parameters in the Gaussian Process.

        Returns
        -------
        List[str]
            A list of parameter names used in the Gaussian Process.
        """
        self.gp.get_parameter_names()

    @property
    def tau(self) -> jax.Array:
        """Return the autocorrelation time of the chains.

        Returns
        -------
        jax.Array
            An array containing the autocorrelation time for each parameter in the chains.

        Raises
        ------
        AttributeError
            If the posteriors have not been derived yet, i.e., if `derive_posteriors` has not been called.
        """
        if self._mcmc_samples is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                derive_posteriors prior to populate the attributes."
            )
        return self._tau

    @property
    def autocorr(self) -> List[float]:
        """Return the autocorrelation time of the chains.

        Returns
        -------
        List[float]
            The autocorrelation time for each parameter in the chains.

        Raises
        ------
        AttributeError
            If the posteriors have not been derived yet, i.e., if `derive_posteriors` has not been called.
        """
        if self._autocorr is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to populate the attributes."
            )

        return self._autocorr

    @property
    def max_loglikelihood(self) -> float:
        """Return the maximum loglikelihood from the derived posteriors.

        Returns
        -------
        float
            The maximum loglikelihood value from the derived posteriors.

        Raises
        ------
        AttributeError
            If the posteriors have not been derived yet, i.e., if `derive_posteriors` has not been called.
        """
        if self._loglikelihoods is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to populate the attributes."
            )

        return self._loglikelihoods.max()
