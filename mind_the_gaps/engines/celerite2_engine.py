import copy
import time
import warnings
from multiprocessing import Pool
from typing import Dict, List

import arviz as az
import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.initialization import init_to_value

from mind_the_gaps.engines.gp_engine import BaseGPEngine
from mind_the_gaps.gp.celerite2_gaussian_process import Celerite2GP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve
from mind_the_gaps.models.kernel_spec import KernelSpec


class Celerite2GPEngine(BaseGPEngine):
    """Celerite2 Gaussian Process Engine, used for modelling lightcurves using the Celerite2 library with Numpyro MCMC sampling.
    This engine wraps the CeleriteGP class and provides methods for fitting the GP, deriving posteriors, and generating lightcurves from the posteriors.
    It allows for parallelized MCMC sampling and provides methods to check convergence, calculate autocorrelation times, and generate lightcurves from the derived posteriors.

    Inherits from:
    ---------------
    BaseGP : Base class for Gaussian Process models
    """

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
        seed: int = None,
        fit_mean: bool = True,
    ):
        """Initialise the Celerite2 Gaussian Process Engine with a kernel specification,

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
        if seed == None:
            seed = int(time.time())

        self.seed = seed
        self.meanmodel = meanmodel
        self.fit_mean = fit_mean
        if self.meanmodel is None or self.meanmodel.lower() == "fixed":
            self.fit_mean = False

        self.mean_params = mean_params
        self.rng_key = jax.random.PRNGKey(self.seed)

        numpyro.enable_x64()
        numpyro.set_platform("cpu")
        #
        self.gp = Celerite2GP(
            kernel_spec=self.kernel_spec,
            lightcurve=self._lightcurve,
            meanmodel=self.meanmodel,
            mean_params=mean_params,
            # rng_key=self.rng_key,
        )
        self.init_params = self.kernel_spec.get_param_array()
        if self.fit_mean:
            self.init_params = jnp.concatenate([mean_params, self.init_params])

        self._mcmc_samples = {}
        self._autocorr = []

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

        gp = self.gp.compute(t, params=params, fit=fit)  # , rng_key=subkey)

        log_likelihood = gp.log_likelihood(self._lightcurve.y)
        numpyro.deterministic("log_likelihood", log_likelihood)
        numpyro.sample(
            "obs",
            gp.numpyro_dist(),
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
        self.gp.compute(params=opt_params, t=self._lightcurve.times, fit=True)
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

    def derive_posteriors(
        self,
        num_warmup: int,
        num_chains: int,
        max_steps: int,
        converge_steps: int,
        fit=True,
        progress=True,
        perc: float = 0.1,
    ) -> None:
        """Derive the posterior distributions of the Gaussian Process parameters using MCMC sampling.
        This method initialises the parameters, runs MCMC sampling using the Numpyro with the NUTS kernel,
        and checks for convergence based on the autocorrelation time of the samples.
        It updates the Gaussian Process with the sampled parameters and stores the samples for further analysis.

        Parameters
        ----------
        num_warmup : int
            Number of warmup steps for the MCMC sampling.
        num_chains : int
            Number of chains to run in parallel for MCMC sampling.
        max_steps : int
            Maximum number of steps for the MCMC sampling.
        converge_steps : int
            Number of steps to check for convergence in the MCMC sampling.
        fit : bool, optional
            Whether to fit the parameters before running MCMC, by default True.
        progress : bool, optional
            Whether to show a progress bar during MCMC sampling, by default True.
        perc : float
            Percentage for the normal distribution used to spread the parameters, by default 0.1.

        Raises
        ------
        ValueError
            If `max_steps` is less than `converge_steps`, as it would not allow for at least one iteration of MCMC sampling.

        """

        old_tau = jnp.inf
        if fit:
            self.minimize()
            fixed_params = self.initialise_params(num_chains=num_chains, perc=perc)

            kernel = NUTS(
                self.numpyro_model,
                adapt_step_size=True,
                dense_mass=False,
                init_strategy=init_to_value(values=fixed_params),
            )
        else:
            kernel = NUTS(
                self.numpyro_model,
                adapt_step_size=True,
                dense_mass=False,
            )

        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=1,
            num_chains=num_chains,
            chain_method="parallel",
            jit_model_args=False,
            progress_bar=progress,
        )
        self.rng_key, subkey = jax.random.split(self.rng_key)
        mcmc.run(subkey, t=self._lightcurve.times)
        state = mcmc.last_state

        mcmc = MCMC(
            kernel,
            num_warmup=0,
            num_samples=converge_steps,
            num_chains=num_chains,
            chain_method="parallel",
            progress_bar=progress,
            jit_model_args=True,
        )
        mcmc.post_warmup_state = state
        num_iterations = int(max_steps / converge_steps)
        if num_iterations < 1:
            raise ValueError(
                f"max_steps ({max_steps}) must be at least as large as converge_steps ({converge_steps}) to run at least one iteration."
            )
        for iteration in range(num_iterations):
            mcmc.run(
                mcmc.post_warmup_state.rng_key,
                t=self._lightcurve.times,
            )
            mcmc.post_warmup_state = mcmc.last_state
            idata = az.from_numpyro(mcmc)

            samples = mcmc.get_samples(group_by_chain=True)

            if iteration == 0:
                self._mcmc_samples = samples
            else:
                for key in samples:
                    self._mcmc_samples[key] = jnp.concatenate(
                        [self._mcmc_samples[key], samples[key]], axis=1
                    )

            tau = self._auto_corr_time(self._mcmc_samples, num_chains)

            self._autocorr.append(jnp.mean(tau))

            if jnp.all(tau * 100 < (iteration + 1) * converge_steps) and np.all(
                jnp.abs(old_tau - tau) / tau < 0.01
            ):
                print(f"MCMC converged after {(iteration+1)*converge_steps} steps.")
                break
            else:
                print(f"MCMC not converged after {(iteration+1)*converge_steps} steps.")
            old_tau = tau
        self._tau = tau
        self._mcmc = mcmc
        self._loglikelihoods = idata.posterior["log_likelihood"].values

    def _generate_lc_from_params(self, params: jax.Array) -> GappyLightcurve:
        """Generate a lightcurve from the Gaussian Process parameters.
        This method creates a new GappyLightcurve instance by sampling from the Gaussian Process defined by the kernel specification.

        Parameters
        ----------
        params : jax.Array
            Parameters for the Gaussian Process, which include both the mean model parameters and the kernel parameters.

        Returns
        -------
        GappyLightcurve
            A new GappyLightcurve instance generated from the Gaussian Process parameters.
        """
        if self.gp.meanmodel.sampled_mean:
            mean_params = params[: self.gp.meanmodel.sampled_parameters]
            kernel_params = params[self.gp.meanmodel.sampled_parameters :]

        else:
            mean_params = self.init_params[: self.gp.meanmodel.sampled_parameters]
            kernel_params = params

        self.kernel_spec.update_params_from_array(kernel_params)
        gp_sample = Celerite2GP(
            kernel_spec=self.kernel_spec,
            meanmodel=self.meanmodel,
            lightcurve=self._lightcurve,
            rng_key=self.rng_key,
            mean_params=mean_params,
        )
        psd_model = gp_sample.get_psd()
        simulator = self._lightcurve.get_simulator(psd_model, pdf="Gaussian")
        rates = simulator.generate_lightcurve()
        noisy_rates, dy = simulator.add_noise(rates)
        lc = GappyLightcurve(self._lightcurve.times, noisy_rates, dy)
        return lc

    def generate_from_posteriors(self, nsims: int) -> List[GappyLightcurve]:
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

        flat_samples = {
            key: v.reshape(-1)
            for key, v in self._mcmc_samples.items()
            if key != "log_likelihood"
        }

        total_samples = len(next(iter(flat_samples.values())))
        if self._mcmc_samples is None:
            raise RuntimeError(
                "Posteriors have not been derived. Please run derive_posteriors prior to calling this method."
            )
        if nsims >= total_samples:
            warnings.warn(
                f"The number of simulation requested {nsims} is higher than the number of posterior samples {len(next(iter(self._mcmc_samples.values())))}, so many samples will be drawn more than once"
            )

        sampled_indices = np.random.choice(total_samples, nsims, replace=True)
        sampled_params = np.column_stack(
            [v[sampled_indices] for v in flat_samples.values()]
        )

        lightcurves = []
        for params in sampled_params:
            lightcurves.append(self._generate_lc_from_params(params=params))

        return lightcurves

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
        return acf

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
        if self._mcmc_samples is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to populate the attributes."
            )
        flat_idx = jnp.argmax(self._loglikelihoods)
        idx = jnp.unravel_index(flat_idx, self._loglikelihoods.shape)
        chain_idx = int(idx[0])
        step_idx = int(idx[1])

        # Extract and stack the parameter values at that index (excluding log_likelihood)
        values = [
            self._mcmc_samples[param][chain_idx, step_idx]
            for param in self._mcmc_samples
            if param != "log_likelihood"
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
