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

from mind_the_gaps.engines.base_numpyro_engine import BaseNumpyroGPEngine
from mind_the_gaps.engines.gp_engine import BaseGPEngine
from mind_the_gaps.gp.celerite2_gaussian_process import Celerite2GP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve
from mind_the_gaps.models.kernel_spec import KernelSpec


class Celerite2GPEngine(BaseNumpyroGPEngine):
    """Celerite2 Gaussian Process Engine, used for modelling lightcurves using the Celerite2 library with Numpyro MCMC sampling.
    This engine wraps the CeleriteGP class and provides methods for fitting the GP, deriving posteriors, and generating lightcurves from the posteriors.
    It allows for parallelized MCMC sampling and provides methods to check convergence, calculate autocorrelation times, and generate lightcurves from the derived posteriors.

    Inherits from:
    ---------------
    BaseNumpyroGPEngine : Base class for Numpyro GP engines
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
        super().__init__(
            kernel_spec=kernel_spec,
            lightcurve=lightcurve,
            seed=seed,
            mean_params=mean_params,
            fit_mean=fit_mean,
            meanmodel=meanmodel,
        )
        self.gp = Celerite2GP(
            kernel_spec=self.kernel_spec,
            lightcurve=self._lightcurve,
            meanmodel=self.meanmodel,
            mean_params=mean_params,
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
        self.gp.compute_fit(params=opt_params, t=self._lightcurve.times)
        self.init_params = opt_params
        self.kernel_spec.update_params_from_array(
            self.init_params[self.gp.meanmodel.sampled_parameters :]
        )

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
            jit_model_args=True,
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
        kernel_spec = copy.deepcopy(self.kernel_spec)
        if self.gp.meanmodel.sampled_mean:
            mean_params = params[: self.gp.meanmodel.sampled_parameters]
            kernel_params = params[self.gp.meanmodel.sampled_parameters :]

        else:
            mean_params = self.init_params[: self.gp.meanmodel.sampled_parameters]
            kernel_params = params

        self.kernel_spec.update_params_from_array(kernel_params)
        gp_sample = Celerite2GP(
            kernel_spec=kernel_spec,
            meanmodel=self.meanmodel,
            lightcurve=self._lightcurve,
            mean_params=mean_params,
        )
        psd_model = gp_sample.get_psd()
        simulator = self._lightcurve.get_simulator(psd_model, pdf="Gaussian")
        rates = simulator.generate_lightcurve()
        noisy_rates, dy = simulator.add_noise(rates)
        lc = GappyLightcurve(self._lightcurve.times, noisy_rates, dy)
        return lc
