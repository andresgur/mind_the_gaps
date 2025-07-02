import time
import warnings
from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import numpyro

from mind_the_gaps.engines.gp_engine import BaseGPEngine

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
        "perc",
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
        else:
            self.fit_mean = True

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
        self._ndim = len(self.init_params)
        self._converged = False

    def numpyro_model(
        self,
        t: jax.Array,
    ) -> None:
        """Numpyro model for the Gaussian Process, which defines the prior distributions (Using the KernelSpec)
        and the likelihood of the model given the lightcurve data. This method computes

        Parameters
        ----------
        t : jax.Array
            The time array of the lightcurve.
        """
        self.gp.compute_sample(t)
        numpyro.sample("obs", self.gp.numpyro_dist(), obs=self._lightcurve.y)

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
        if self.fit_mean:
            upper_bounds = jnp.concatenate([self.gp.meanmodel.bounds[1], upper_bounds])
            lower_bounds = jnp.concatenate([self.gp.meanmodel.bounds[0], lower_bounds])

        solver = jaxopt.ScipyBoundedMinimize(
            method="l-bfgs-b",
            fun=self.gp.negative_log_likelihood,
            maxiter=1000,
        )

        opt_params, res = solver.run(
            init_params=self.init_params,
            bounds=(lower_bounds, upper_bounds),
        )
        self.gp.compute_fit(params=opt_params, t=self._lightcurve.times)
        self.init_params = opt_params
        self.kernel_spec.update_params_from_array(
            self.init_params[self.gp.meanmodel.sampled_parameters :]
        )

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

    def _auto_window(self, taus: jax.Array, c: int) -> jax.Array:
        """Determine the window size for the autocorrelation function based on the specified threshold `c`.

        Parameters
        ----------
        taus : jax.Array
            The autocorrelation times as a jax array.
        c : int
            The threshold factor to determine the window size.

        Returns
        -------
        jax.Array
            The index of the first autocorrelation time that exceeds the threshold `c` times the autocorrelation time.
            If no such index exists, returns the last index of `taus`.
        """
        m = jnp.arange(len(taus)) < c * taus
        if jnp.any(m):
            return jnp.argmin(m)
        return len(taus) - 1

    def _function_1d(self, x: jax.Array) -> jax.Array:
        """Estimate the normalized autocorrelation function of a 1-D series

        Args:
            x: The series as a 1-D jax array.

        Returns:
            jax.array: The autocorrelation function of the time series.

        """

        x = jnp.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")
        n = self._next_pow_two(len(x))

        f = jnp.fft.fft(x - jnp.mean(x), n=2 * n)
        acf = jnp.fft.ifft(f * jnp.conjugate(f))[: len(x)].real
        acf /= acf[0]
        return jnp.asarray(acf)

    def _next_pow_two(self, n):
        """Returns the next power of two greater than or equal to `n`"""
        i = 1
        while i < n:
            i = i << 1
        return i

    def _auto_corr_time(
        self, samples: Dict, num_chains: int, c: int = 5, tol: int = 50, quiet=True
    ) -> float:
        """Estimate the autocorrelation time of the MCMC samples.
        Jax version of the Emcee approach to get autocorrelation time. (See https://emcee.readthedocs.io/en/stable/user/autocorr/#autocorrelation-analysis)

        Parameters
        ----------
        samples : Dict
            Dictionary of MCMC samples, where keys are parameter names and values are jax arrays of samples.
        num_chains : int
            Number of chains in the MCMC samples.
        c : int, optional
            Step size of the window search, by default 5
        tol : int, optional
            Minimum number of autocorrelation times to trust the estimate, by default 50
        quiet : bool, optional
            Warn when chain is too short rather than raise a ValueError, by default False

        Returns
        -------
        jax.Array
            An estimate of the integrated autocorrelation time of the chain for each parameter.

        Raises
        ------
        ValueError
            If the dimensions of the input samples are invalid or if the chain is shorter than the specified tolerance times the integrated autocorrelation time.
        """

        x = jnp.stack(
            [samples[key].T for key in samples.keys() if key != "log_likelihood"],
            axis=-1,
        )
        x = jnp.atleast_1d(x)

        if len(x.shape) == 2:
            x = x[:, :, jnp.newaxis]
        if len(x.shape) != 3:
            raise ValueError("invalid dimensions")

        n_t, n_c, n_d = x.shape
        tau_est = jnp.empty(n_d)
        windows = jnp.empty(n_d, dtype=int)

        for d in range(n_d):
            f = jnp.zeros(n_t)
            for c in range(n_c):
                f += self._function_1d(x[:, c, d])
            f /= n_c
            taus = 2.0 * jnp.cumsum(f) - 1.0
            windows = windows.at[d].set(self._auto_window(taus, c))
            tau_est = tau_est.at[d].set(taus[windows[d]])

        return tau_est

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
    def max_parameters(self) -> jax.Array:
        """Return the parameters corresponding to the maximum loglikelihood.

        Returns
        -------
        jax.Array
            An array containing the parameters corresponding to the maximum loglikelihood from the derived posteriors.

        Raises
        ------
        AttributeError
            If the posteriors have not been derived yet, i.e., if `derive_posteriors` has not been called.
        """
        if self._mcmc_samples is None or self._loglikelihoods is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run derive_posteriors first."
            )

        # Assume all parameters have the same shape: (num_chains, num_samples)
        sample_shape = next(iter(self._mcmc_samples.values())).shape
        num_chains, num_samples = sample_shape

        flat_idx = jnp.argmax(self._loglikelihoods)
        chain_idx, sample_idx = jnp.divmod(flat_idx, num_samples)

        # Extract parameter values at that (chain, sample) location
        values = [
            self._mcmc_samples[param][chain_idx, sample_idx]
            for param in self._mcmc_samples
            if param != "log_likelihood"  # handles "log_likelihood" or other keys
        ]

        return jnp.array(values)

    @property
    def median_parameters(self) -> jax.Array:
        """Return the median parameters from the derived posteriors.

        Returns
        -------
        jax.Array
            An array containing the median parameters from the derived posteriors.

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
        return [
            jnp.median(self._mcmc_samples[param].reshape(-1))
            for param in self._mcmc_samples
            if param != "log_likelihood"
        ]

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

    def generate_from_posteriors(
        self,
        nsims: int,
        par_workers: int = 1,
    ) -> List[GappyLightcurve]:
        """Generate lightcurves from the posterior samples of the Gaussian Process.
        This method samples from the posterior distributions of the parameters derived from MCMC sampling.
        It generates a specified number of lightcurves by randomly selecting parameter sets from the posterior samples.

        Parameters
        ----------
        nsims : int
            The number of lightcurves to generate from the posterior samples.


        Returns
        -------
        List[GappyLightcurve]
            A list of GappyLightcurve instances generated from the posterior samples.

        Raises
        ------
        RuntimeError
            If the posteriors have not been derived yet, i.e., if `derive_posteriors` has not been called.
        """
        if self._mcmc_samples is None:
            raise RuntimeError(
                "Posteriors have not been derived. Please run derive_posteriors prior to calling this method."
            )
        flat_samples = {
            key: v.reshape(-1)
            for key, v in self._mcmc_samples.items()
            if key != "log_likelihood"
        }

        total_samples = len(next(iter(flat_samples.values())))

        if nsims >= total_samples:
            warnings.warn(
                f"The number of simulation requested {nsims} is higher than the number of posterior samples {len(next(iter(self._mcmc_samples.values())))}, so many samples will be drawn more than once"
            )

        sampled_indices = np.random.choice(total_samples, nsims, replace=True)
        sampled_params = np.column_stack(
            [v[sampled_indices] for v in flat_samples.values()]
        )

        return [self._generate_lc_from_params(p) for p in sampled_params]

    @property
    def k(self):
        """
        Number of variable parameters

        Returns
        -------
        int
            Number of variable parameters
        """
        return self._ndim

    def _get_log_likes(self):
        from jax import lax

        samples = jax.tree_util.tree_map(lambda x: x.reshape(-1), self._mcmc_samples)
        param_names = sorted(samples.keys())
        param_matrix = jnp.stack([samples[k] for k in param_names], axis=-1)

        def compute_likelihood(p):
            return self.gp.compute_fit(
                params=p, t=self._lightcurve.times, log_like=True
            )

        self._loglikelihoods = lax.map(compute_likelihood, param_matrix)

    def _thin_and_discard_samples(self, max_steps: int):
        """
        Thin and discard MCMC samples based on autocorrelation time and convergence.

        Parameters
        ----------
        max_steps : int
            The total number of MCMC steps (iterations) performed.

        Returns
        -------
        tuple:
            discard (int): Number of samples discarded as burn-in.
            thin (int): Thinning factor applied.
        """
        mean_tau = jnp.mean(self._tau)

        if self._converged:

            discard = int(mean_tau * 40)
            if discard > max_steps:
                discard = int(mean_tau * 10)
            thin = int(mean_tau / 2)
        else:
            import warnings

            warnings.warn(
                "The chains did not converge after " + str(max_steps) + " iterations!"
            )
            thin = int(mean_tau / 4) if mean_tau > 0 else 1
            discard = int(mean_tau * 5) if mean_tau > 0 else 0

        discard = max(discard, 0)
        thin = max(thin, 1)

        # Apply thinning and discard burn-in
        self._mcmc_samples = {
            key: val[:, discard::thin] for key, val in self._mcmc_samples.items()
        }
