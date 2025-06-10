import copy
import pprint as pp
import warnings
from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List

import arviz as az
import celerite
import celerite2.jax
import emcee
import jax
import jax.experimental
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import AIES, MCMC, NUTS
from numpyro.infer.ensemble import EnsembleSampler
from numpyro.infer.initialization import init_to_value

from mind_the_gaps.engines.base_numpyro_engine import BaseNumpyroGPEngine
from mind_the_gaps.engines.gp_engine import BaseGPEngine
from mind_the_gaps.gp.tinygp_gaussian_process import TinyGP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve
from mind_the_gaps.models.kernel_spec import KernelSpec


class TinyGPEngine(BaseNumpyroGPEngine):
    posterior_params = BaseNumpyroGPEngine.posterior_params | {"aies"}

    def __init__(
        self,
        kernel_spec: KernelSpec,
        lightcurve: GappyLightcurve,
        meanmodel: str = None,
        mean_params: jax.Array = None,
        seed: int = 0,
        fit_mean: bool = True,
    ):
        super().__init__(
            kernel_spec=kernel_spec,
            lightcurve=lightcurve,
            seed=seed,
            mean_params=mean_params,
            fit_mean=fit_mean,
            meanmodel=meanmodel,
        )
        self.gp = TinyGP(
            kernel_spec=self.kernel_spec,
            lightcurve=self._lightcurve,
            meanmodel=self.meanmodel,
            mean_params=mean_params,
        )

    def initialize_params(self, num_chains: int, std_dev=0.1) -> Dict:
        """
        Generate multiple initial parameter values for NUTS sampling, ensuring they
        respect bounds and have a Gaussian spread.

        Args:
            init_params (list or array): Initial parameter values.
            num_chains (int): Number of MCMC chains.
            bounds (dict): Dictionary with parameter names as keys and (min, max) tuples as values.
            std_dev (float): Standard deviation for Gaussian noise.

        Returns:
            dict: Dictionary of JAX arrays with shape (num_chains, param_dim).
        """
        param_dict = {}
        i = 0
        for term in self.kernel_spec.terms:
            for param_name, param_spec in term.parameters.items():
                if param_spec.fixed:
                    continue

                base_value = param_spec.value
                low, high = param_spec.bounds

                spread_values = base_value + np.random.normal(
                    0, std_dev, size=num_chains
                )

                spread_values = np.clip(spread_values, float(low), float(high))

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
        aies: bool = False,
    ) -> None:

        old_tau = jnp.inf
        if fit:
            self.init_params = self.minimize()
            self.kernel_spec.update_params_from_array(
                self.init_params[self.gp.meanmodel.sampled_parameters :]
            )

        fixed_params = self.initialize_params(num_chains=num_chains)

        if aies:
            kernel = AIES(
                self.numpyro_model,  # , init_strategy=init_to_value(values=fixed_params)
                moves={AIES.DEMove(): 0.5, AIES.StretchMove(): 0.5},
            )
            chain_method = "vectorized"
        else:
            kernel = NUTS(
                self.numpyro_model,
                adapt_step_size=True,
                dense_mass=False,
                init_strategy=init_to_value(values=fixed_params),
            )
            chain_method = "parallel"

        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=1,
            num_chains=num_chains,
            chain_method=chain_method,
            jit_model_args=True,
            progress_bar=progress,
        )

        mcmc.run(
            self.rng_key,
            t=self._lightcurve.times,
        )
        state = mcmc.last_state

        mcmc = MCMC(
            kernel,
            num_warmup=0,
            num_samples=converge_steps,
            num_chains=num_chains,
            chain_method=chain_method,
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
                np.abs(old_tau - tau) / tau < 0.01
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
        kernel_spec = copy.deepcopy(self.kernel_spec)
        if self.gp.meanmodel.sampled_mean:
            mean_params = params[: self.gp.meanmodel.sampled_parameters]
            kernel_params = params[self.gp.meanmodel.sampled_parameters :]

            # mean = self.meanmodel.compute_mean(fit=True, params=mean_params)
        else:
            mean_params = self.init_params[: self.gp.meanmodel.sampled_parameters]
            kernel_params = params

        kernel_spec.update_params_from_array(kernel_params)
        # Is this needed? Can get psd from kernel directly without getting the gp
        gp_sample = TinyGP(
            kernel_spec=kernel_spec,
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

    def generate_from_posteriors(self, nsims: int):
        if self._mcmc_samples is None:
            raise RuntimeError(
                "Posteriors have not been derived. Please run derive_posteriors prior to calling this method."
            )
        if nsims >= len(next(iter(self._mcmc_samples.values()))):
            warnings.warn(
                f"The number of simulation requested {nsims} is higher than the number of posterior samples {len(next(iter(self._mcmc_samples.values())))}, so many samples will be drawn more than once"
            )
        flat_samples = {
            key: v.reshape(-1)
            for key, v in self._mcmc_samples.items()
            if key != "log_likelihood"
        }

        total_samples = len(next(iter(flat_samples.values())))

        sampled_indices = np.random.choice(total_samples, nsims, replace=True)
        sampled_params = np.column_stack(
            [v[sampled_indices] for v in flat_samples.values()]
        )

        lightcurves = []
        for params in sampled_params:
            lightcurves.append(self._generate_lc_from_params(params=params))

        return lightcurves

    def auto_window(self, taus: jax.Array, c: int) -> jax.Array:
        m = jnp.arange(len(taus)) < c * taus
        if jnp.any(m):
            return jnp.argmin(m)
        return len(taus) - 1

    def function_1d(self, x: jax.Array) -> jax.Array:
        """Estimate the normalized autocorrelation function of a 1-D series

        Args:
            x: The series as a 1-D jax array.

        Returns:
            jax.array: The autocorrelation function of the time series.

        """

        x = jnp.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")
        n = self.next_pow_two(len(x))

        f = jnp.fft.fft(x - jnp.mean(x), n=2 * n)
        acf = jnp.fft.ifft(f * jnp.conjugate(f))[: len(x)].real
        acf /= acf[0]
        return acf

    def next_pow_two(self, n):
        """Returns the next power of two greater than or equal to `n`"""
        i = 1
        while i < n:
            i = i << 1
        return i

    def _auto_corr_time(self, samples, num_chains, c=5, tol=50, quiet=False):

        x = jnp.stack([samples[key].T for key in samples.keys()], axis=-1)
        x = jnp.atleast_1d(x)

        if len(x.shape) == 2:
            x = x[:, :, jnp.newaxis]
        if len(x.shape) != 3:
            raise ValueError("invalid dimensions")

        n_t, n_c, n_d = x.shape
        tau_est = jnp.empty(n_d)
        windows = jnp.empty(n_d, dtype=int)

        # Loop over parameters
        for d in range(n_d):
            f = jnp.zeros(n_t)
            for c in range(n_c):
                f += self.function_1d(x[:, c, d])
            f /= n_c
            taus = 2.0 * jnp.cumsum(f) - 1.0
            windows = windows.at[d].set(self.auto_window(taus, c))
            tau_est = tau_est.at[d].set(taus[windows[d]])

        # Check convergence
        flag = tol * tau_est > n_t

        # Warn or raise in the case of non-convergence
        if jnp.any(flag):
            msg = (
                "The chain is shorter than {0} times the integrated "
                "autocorrelation time for {1} parameter(s). Use this estimate "
                "with caution and run a longer chain!\n"
            ).format(tol, jnp.sum(flag))
            msg += "N/{0} = {1:.0f};\ntau: {2}".format(tol, n_t / tol, tau_est)
            if not quiet:
                pass
                # raise Exception

        return tau_est

    @property
    def max_loglikelihood(self):
        if self._loglikelihoods is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to populate the attributes."
            )

        return self._loglikelihoods.max()

    @property
    def autocorr(self) -> List[float]:
        if self._autocorr is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to populate the attributes."
            )

        return self._autocorr

    @property
    def max_parameters(self):
        """Return the parameters that maximize the loglikehood"""
        if self._mcmc_samples is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to populate the attributes."
            )
        return self._mcmc_samples[jnp.argmax(self._loglikelihoods)]

    @property
    def median_parameters(self):
        if self._mcmc_samples is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                derive_posteriors prior to populate the attributes."
            )
        return jnp.median(self._mcmc_samples, axis=0)

    @property
    def parameter_names(self):
        raise NotImplementedError

    @property
    def tau(self):
        """The autocorrelation time of the chains"""
        if self._mcmc_samples is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                derive_posteriors prior to populate the attributes."
            )
        return self._tau
