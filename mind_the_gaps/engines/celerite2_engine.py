import warnings
from functools import partial
from multiprocessing import Pool
from typing import Callable, List

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
from jaxopt import ScipyBoundedMinimize
from numpyro.infer import AIES, ESS, MCMC, NUTS
from numpyro.infer.initialization import init_to_value

from mind_the_gaps.engines.gp_engine import BaseGPEngine
from mind_the_gaps.gp.celerite2_gaussian_process import Celerite2GP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve


class Celerite2GPEngine(BaseGPEngine):
    posterior_params = {
        "fit",
        "max_steps",
        "num_chains",
        "num_warmup",
        "converge_steps",
    }

    def __init__(
        self,
        kernel_fn: Callable,
        lightcurve: GappyLightcurve,
        # max_steps: int,
        # num_chains: int,
        # num_warmup: int,
        # converge_step: int,
        # device: str,
        cpus: int,
        params: float,
        mean: str = None,
        # nsims: int = 10,
        seed: int = 0,
        bounds: dict = None,
    ):

        self.kernel_fn = kernel_fn

        self._lightcurve = lightcurve
        self.seed = seed
        self.mean = mean
        # self.mean = None
        # self.max_steps = max_steps
        # self.num_chains = num_chains
        # self.num_warmup = num_warmup
        # self.converge_step = converge_step
        # self.nsims = nsims
        self.rng_key = jax.random.PRNGKey(self.seed)
        # self.params = jnp.array(params)
        # self._ndim = len(self.params)
        self.bounds = bounds
        self.cpus = cpus

        numpyro.enable_x64()
        numpyro.set_platform("cpu")
        numpyro.set_host_device_count(self.cpus)
        self.gp = Celerite2GP(
            kernel_fn=self.kernel_fn,
            lightcurve=self._lightcurve,
            mean=self.mean,
            params=params,
            bounds=self.bounds,
            rng_key=self.rng_key,
        )
        self.gp.compute(params, self._lightcurve.times, fit=False)
        self._mcmc_samples = {}
        self._autocorr = []

    # def setup_gp(self, params: jnp.array, t: jnp.array, fit: bool):

    #    # self.gp.set_parameters(params=params, t=t, fit=fit)
    #    self.params = params
    #    kernel = self.kernel_fn(
    #        params=self.params, fit=fit, rng_key=self.rng_key, bounds=self.bounds
    #    )

    # self.gp = celerite2.jax.GaussianProcess(kernel, mean=100)

    #  self.gp.compute(t, yerr=self._lightcurve.dy, check_sorted=False)

    # def negative_log_likelihood(self, params: jnp.array, fit=True):
    #     self.setup_gp(params, self._lightcurve.times, fit=fit)
    #     # jax.debug.print("params {params}", params=params)
    #
    #        nll_value = -self.gp.log_likelihood(self._lightcurve.y)
    #        # jax.debug.print("nll_value: {nll_value}", nll_value=nll_value)
    #        return nll_value

    def numpyro_model(self, t, y=None, params=None, fit=False):

        self.gp.compute(params, t, fit=fit)

        log_likelihood = self.gp.log_likelihood(self._lightcurve.y)
        numpyro.deterministic("log_likelihood", log_likelihood)
        # with numpyro.plate("data", len(self._lightcurve.y)):
        numpyro.sample(
            "obs",
            self.gp.gp.numpyro_dist(),
            obs=self._lightcurve.y,
            rng_key=self.rng_key,
        )
        # numpyro.deterministic("psd", psd_model=self.gp.kernel.get_psd)

    def minimize(self):

        upper_bounds = jnp.array([values[1] for values in self.bounds.values()])
        lower_bounds = jnp.array([values[0] for values in self.bounds.values()])
        upper_bounds = jnp.concatenate([self.gp.mean_model.bounds[1], upper_bounds])
        lower_bounds = jnp.concatenate([self.gp.mean_model.bounds[0], lower_bounds])

        solver = jaxopt.ScipyBoundedMinimize(
            method="l-bfgs-b",
            fun=self.gp.negative_log_likelihood,
            maxiter=1000,
        )

        opt_params, res = solver.run(
            init_params=self.gp.params,
            bounds=(lower_bounds, upper_bounds),
        )
        self.gp.compute(params=opt_params, t=self._lightcurve.times, fit=False)
        return opt_params

    def initialize_params(self, init_params, num_chains, std_dev=0.1):
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

        for i, (param_name, (low, high)) in enumerate(self.bounds.items()):
            base_value = init_params[i]

            # Generate Gaussian noise and spread initial values
            spread_values = base_value + np.random.normal(0, std_dev, size=num_chains)

            # Clip to ensure values stay within bounds
            spread_values = np.clip(spread_values, low, high)

            # Convert to JAX array and store
            param_dict[param_name] = jnp.array(spread_values)

        return param_dict

    def derive_posteriors(
        self,
        num_warmup: int,
        num_chains: int,
        max_steps: int,
        converge_steps: int,
        fit=True,
        seed=0,
    ):

        converged = False
        old_tau = jnp.inf
        if fit:
            self.params = self.minimize()

        fixed_params = self.initialize_params(self.params, num_chains=num_chains)

        kernel = NUTS(
            self.numpyro_model,
            adapt_step_size=True,
            dense_mass=True,
            init_strategy=init_to_value(values=fixed_params),
        )
        # kernel = NUTS(
        #    self.numpyro_model,
        #    adapt_step_size=True,
        #    dense_mass=True,
        #    init_strategy=init_to_value(values=fixed_params),
        # )

        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=1,
            num_chains=num_chains,
            chain_method="parallel",
            jit_model_args=False,
            progress_bar=False,
        )

        mcmc.run(
            self.rng_key,
            t=self._lightcurve.times,
            y=self._lightcurve.y,
            params=self.params,
            # fit=False,
        )
        state = mcmc.last_state

        mcmc = MCMC(
            kernel,
            num_warmup=0,
            num_samples=converge_steps,
            num_chains=num_chains,
            chain_method="parallel",
            progress_bar=True,
            jit_model_args=True,
        )
        mcmc.post_warmup_state = state
        for iteration in range(int(max_steps / converge_steps)):
            mcmc.run(
                mcmc.post_warmup_state.rng_key,
                t=self._lightcurve.times,
                # y=self._lightcurve.y,
                # params=self.init_params,
            )
            mcmc.post_warmup_state = mcmc.last_state
            idata = az.from_numpyro(mcmc)
            # az_summary = az.summary(idata)
            az_summary = az.summary(
                idata,
                var_names=[
                    v for v in idata.posterior.data_vars if "log_likelihood" not in v
                ],
            )
            samples = mcmc.get_samples(group_by_chain=True)
            self.samples2 = mcmc.get_samples()

            if iteration == 0:
                self._mcmc_samples = samples  # {key: samples[key] for key in samples}
            else:
                for key in samples:
                    self._mcmc_samples[key] = jnp.concatenate(
                        [self._mcmc_samples[key], samples[key]], axis=0
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
                print(f"{az_summary['r_hat']} \nmean tau: {tau}")

        self._mcmc = mcmc
        self._loglikelihoods = idata.posterior["log_likelihood"].values

        # az.plot_autocorr(idata, var_names=list(idata.posterior.data_vars))
        # plt.savefig("autocorrellation.png")

        # self._mcmc_samples = mcmc.get_samples()

    def _get_chains(self, discard=0, thin=1, flat=True):
        samples_discarded = {k: v[discard:] for k, v in self.posterior_samples.items()}

        samples_thinned = {k: v[::thin] for k, v in samples_discarded.items()}

        if flat:
            samples_flat = {
                k: v.reshape(-1, *v.shape[2:]) for k, v in samples_thinned.items()
            }
            self._mcmc_samples = samples_flat

        self._mcmc_samples = samples_thinned

    def _generate_lc_from_params(self, params):

        kernel = self.kernel_fn(params=params[:-1], fit=True, rng_key=self.rng_key)
        gp_sample = celerite2.jax.GaussianProcess(kernel)  # , mean=sample["mean"])
        gp_sample.compute(self._lightcurve.times, check_sorted=False)
        psd_model = gp_sample.kernel.get_psd
        simulator = self._lightcurve.get_simulator(psd_model, pdf="Gaussian")
        rates = simulator.generate_lightcurve()
        noisy_rates, dy = simulator.add_noise(rates)
        lc = GappyLightcurve(self._lightcurve.times, noisy_rates, dy)
        return lc

    def _generate_lc_from_params2(self, params):
        gp_sample = Celerite2GP(
            kernel_fn=self.kernel_fn,
            mean=self.mean,
            params=params[:-1],
            lightcurve=self._lightcurve,
            rng_key=self.rng_key,
        )
        psd_model = gp_sample.get_psd()
        simulator = self._lightcurve.get_simulator(psd_model, pdf="Gaussian")
        rates = simulator.generate_lightcurve()
        noisy_rates, dy = simulator.add_noise(rates)
        lc = GappyLightcurve(self._lightcurve.times, noisy_rates, dy)
        return lc

    def generate_from_posteriors(self, nsims: int):
        flat_samples = {key: v.reshape(-1) for key, v in self._mcmc_samples.items()}

        total_samples = len(next(iter(flat_samples.values())))

        if self._mcmc_samples is None:
            raise RuntimeError(
                "Posteriors have not been derived. Please run derive_posteriors prior to calling this method."
            )
        if nsims >= len(next(iter(self._mcmc_samples.values()))):
            warnings.warn(
                "The number of simulation requested (%d) is higher than the number of posterior samples (%d), so many samples will be drawn more than once"
            )
        lcs = []
        sampled_indices = np.random.choice(total_samples, nsims, replace=True)
        sampled_params = np.column_stack(
            [v[sampled_indices] for v in flat_samples.values()]
        )

        samples = np.column_stack(
            [v[np.random.choice(len(v), nsims)] for v in self._mcmc_samples.values()]
        )
        samples2 = np.column_stack(
            [v[np.random.choice(len(v), nsims)] for v in self.samples2.values()]
        )

        lightcurves = []
        for params in sampled_params:
            lightcurves.append(self._generate_lc_from_params2(params=params))

        return lightcurves

    def auto_window(self, taus, c):
        m = jnp.arange(len(taus)) < c * taus
        if jnp.any(m):
            return jnp.argmin(m)
        return len(taus) - 1

    def function_1d(self, x):
        """Estimate the normalized autocorrelation function of a 1-D series

        Args:
            x: The series as a 1-D numpy array.

        Returns:
            array: The autocorrelation function of the time series.

        """

        x = jnp.atleast_1d(x)
        if len(x.shape) != 1:
            raise ValueError("invalid dimensions for 1D autocorrelation function")
        n = self.next_pow_two(len(x))

        # Compute the FFT and then (from that) the auto-correlation function

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
            # windows[d] = self.auto_window(taus, c)
            windows = windows.at[d].set(self.auto_window(taus, c))
            # tau_est[d] = taus[windows[d]]
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
        raise NotImplementedError

    @property
    def max_parameters(self):
        raise NotImplementedError

    @property
    def median_parameters(self):
        raise NotImplementedError

    @property
    def parameter_names(self):
        raise NotImplementedError

    @property
    def tau(self):
        raise NotImplementedError
