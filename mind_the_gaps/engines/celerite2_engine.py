import warnings
from functools import partial
from multiprocessing import Pool
from typing import Callable

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
from scipy.optimize import minimize

from mind_the_gaps.engines.gp_engine import BaseGPEngine

# from mind_the_gaps.gp.celerite2_gaussian_process import Celerite2GP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve


class Celerite2GPEngine(BaseGPEngine):
    posterior_params = {"fit"}

    def __init__(
        self,
        kernel_fn: Callable,
        lightcurve: GappyLightcurve,
        max_steps: int,
        num_chains: int,
        num_warmup: int,
        converge_step: int,
        device: str,
        devices: int,
        params: float,
        nsims: int = 10,
        seed: int = 0,
        bounds: dict = None,
    ):

        self.kernel_fn = kernel_fn
        self.lightcurve = lightcurve
        self.mean = None
        self.max_steps = max_steps
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.converge_step = converge_step
        self.nsims = nsims
        self.rng_key = jax.random.PRNGKey(seed)
        self.init_params = jnp.array(params)
        self.bounds = bounds

        numpyro.enable_x64()
        numpyro.set_platform(device)
        numpyro.set_host_device_count(devices)
        self.setup_gp(self.init_params, self.lightcurve.times, fit=False)

    def setup_gp(self, params: jnp.array, t: jnp.array, fit: bool):

        kernel = self.kernel_fn(
            params=params, fit=fit, rng_key=self.rng_key, bounds=self.bounds
        )
        self.gp = celerite2.jax.GaussianProcess(kernel, mean=100.0)

        self.gp.compute(t, yerr=self.lightcurve.dy, check_sorted=False)

    def negative_log_likelihood(self, params: jnp.array, fit=True):
        self.setup_gp(params, self.lightcurve.times, fit=fit)
        nll_value = -self.gp.log_likelihood(self.lightcurve.y)
        return nll_value

    def numpyro_model(self, t, y=None, params=None, fit=False):
        self.setup_gp(params, t, fit=fit)
        numpyro.sample("obs", self.gp.numpyro_dist(), obs=self.lightcurve.y)

    def minimize(self):

        upper_bounds = jnp.array([values[1] for values in self.bounds.values()])
        lower_bounds = jnp.array([values[0] for values in self.bounds.values()])

        solver = jaxopt.ScipyBoundedMinimize(
            method="l-bfgs-b",
            fun=self.negative_log_likelihood,
            maxiter=1000,
        )

        opt_params, res = solver.run(
            init_params=self.init_params,
            bounds=(lower_bounds, upper_bounds),
        )
        self.setup_gp(params=opt_params, t=self.lightcurve.times, fit=False)
        return opt_params

    def derive_posteriors(self, fit=True, seed=0):

        converged = False

        if fit:
            self.init_params = self.minimize()

        kernel = NUTS(
            self.numpyro_model,
            adapt_step_size=True,
            dense_mass=True,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=1,
            num_chains=self.num_chains,
            chain_method="parallel",
            progress_bar=True,
        )

        mcmc.run(
            self.rng_key,
            t=self.lightcurve.times,
            y=self.lightcurve.y,
            params=self.init_params,
            fit=False,
        )
        state = mcmc.last_state

        mcmc = MCMC(
            kernel,
            num_warmup=0,
            num_samples=self.converge_step,
            num_chains=self.num_chains,
            chain_method="parallel",
            progress_bar=False,
            jit_model_args=True,
        )
        mcmc.post_warmup_state = state
        for iteration in range(int(self.max_steps / self.converge_step)):
            mcmc.run(
                mcmc.post_warmup_state.rng_key,
                t=self.lightcurve.times,
                params=self.init_params,
            )
            mcmc.post_warmup_state = mcmc.last_state
            idata = az.from_numpyro(mcmc)
            az_summary = az.summary(idata)
            if all(az_summary["r_hat"] < 1.05):  # and all(
                # az_summary["ess_tail"] / self.converge_step > 0.1
                # ):
                print(f"MCMC converged after {(iteration+1)*self.converge_step} steps.")
                break
            else:
                print(
                    f"MCMC not converged after {(iteration+1)*self.converge_step} steps."
                )
                print(
                    f"{az_summary['r_hat']} {az_summary['ess_tail'] / self.converge_step}"
                )

        self.mcmc = mcmc

        # az.plot_autocorr(idata, var_names=list(idata.posterior.data_vars))
        # plt.savefig("autocorrellation.png")

        self.mcmc_samples = mcmc.get_samples()

    def _get_chains(self, discard=0, thin=1, flat=True):
        samples_discarded = {k: v[discard:] for k, v in self.posterior_samples.items()}

        samples_thinned = {k: v[::thin] for k, v in samples_discarded.items()}

        if flat:
            samples_flat = {
                k: v.reshape(-1, *v.shape[2:]) for k, v in samples_thinned.items()
            }
            self.mcmc_samples = samples_flat

        self.mcmc_samples = samples_thinned

    def generate_from_posteriors(self, nsims=10):

        if self.mcmc_samples is None:
            raise RuntimeError(
                "Posteriors have not been derived. Please run derive_posteriors prior to calling this method."
            )
        if nsims >= len(next(iter(self.mcmc_samples.values()))):
            warnings.warn(
                "The number of simulation requested (%d) is higher than the number of posterior samples (%d), so many samples will be drawn more than once"
            )
        lcs = []

        samples = np.column_stack(
            [v[np.random.choice(len(v), nsims)] for v in self.mcmc_samples.values()]
        )

        for params in samples:

            kernel = self.kernel_fn(params=params, fit=True, rng_key=self.rng_key)
            gp_sample = celerite2.jax.GaussianProcess(kernel)  # , mean=sample["mean"])
            gp_sample.compute(self.lightcurve.times, check_sorted=False)
            psd_model = gp_sample.kernel.get_psd
            simulator = self.lightcurve.get_simulator(psd_model, pdf="Gaussian")
            rates = simulator.generate_lightcurve()
            noisy_rates, dy = simulator.add_noise(rates)
            lc = GappyLightcurve(self.lightcurve.times, noisy_rates, dy)
            lcs.append(lc)

        return lcs
