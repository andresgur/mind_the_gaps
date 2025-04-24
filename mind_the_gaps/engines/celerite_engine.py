import warnings
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import celerite
import emcee
import numpy as np
from celerite.modeling import ConstantModel, Model
from scipy.optimize import OptimizeResult, minimize

from mind_the_gaps.engines.gp_engine import BaseGPEngine
from mind_the_gaps.gp.celerite_gaussian_process import CeleriteGP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve
from mind_the_gaps.models.celerite.mean_models import (
    GaussianModel,
    LinearModel,
    SineModel,
)


class CeleriteGPEngine(BaseGPEngine):

    posterior_params = {
        "max_steps",
        "initial_chain_params",
        "fit",
        "converge",
        "max_steps",
        "walkers",
        "cores",
        "progress",
    }

    def __init__(
        self,
        kernel: celerite.modeling.Model,
        lightcurve: GappyLightcurve,
        mean_model: str,
        fit_mean: bool = True,
    ):

        self._lightcurve = lightcurve
        self.gp = CeleriteGP(
            kernel=kernel,
            lightcurve=lightcurve,
            mean_model=mean_model,
            fit_mean=fit_mean,
        )

        self.initial_params = self.gp.get_parameter_vector()
        self._ndim = len(self.initial_params)
        self._autocorr = []
        self._loglikelihoods = None
        self._mcmc_samples = None

    def _log_probability(self, params: List[float]) -> float:
        """_summary_

        Parameters
        ----------
        params : List[float]
            List of parameters of the GP at which to calculate the posteriors.

        Returns
        -------
        float
            The log of the posterior.
        """

        self.gp.set_parameter_vector(params)

        lp = self.gp.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.gp.log_likelihood(self._lightcurve.y)

    def _neg_log_like(self, params: np.array) -> float:
        """Returns the negative log likelihood

        Parameters
        ----------
        params : np.array
            Parameters of the GP.

        Returns
        -------
        float
            Negative log likelihood.
        """
        self.gp.set_parameter_vector(params)

        return -self.gp.log_likelihood(self._lightcurve.y)

    def _fit(self, initial_params: np.array = None) -> OptimizeResult:
        """Fit the GP by running a minimization routine.

        Parameters
        ----------
        initial_params : np.array, optional
            Set of initial paramteres. If not given uses those given at initialization, by default None.

        Returns
        -------
        OptimizeResult
            Result of scipy.minimize routine.
        """
        if initial_params is None:
            initial_params = self.initial_params
        solution = minimize(
            self._neg_log_like,
            initial_params,
            method="L-BFGS-B",
            bounds=self.gp.get_parameter_bounds(),
        )
        return solution

    def derive_posteriors(
        self,
        initial_chain_params: np.array = None,
        fit: bool = True,
        converge: bool = True,
        max_steps: int = 10000,
        walkers: int = 12,
        cores: int = 6,
        progress: bool = True,
    ):
        """Derive GP Posteriors

        Parameters
        ----------
        initial_chain_params : np.array, optional
            Set of initial parameters for the chains. If not given will
            use randomized from the parameters given at initialization, by default None
        fit : bool, optional
            Whether to run a minimization routine prior to running the MCMC chains, by default True
        converge : bool, optional
            Whether to stop the chains if convergence is detected, by default True
        max_steps : int, optional
            Maximum number of steps for the chains, by default 10000
        walkers : int, optional
            Number of walkers for the chains, by default 12
        cores : int, optional
            Number of cores for parallelization, by default 6
        progress : bool, optional
            Whether to show progress throught MCMC, by default True
        """

        # set the initial parameters if not given

        if initial_chain_params is None:
            if not fit:
                initial_params = self.initial_params

            else:
                solution = self._fit(self.initial_params)
                initial_params = solution.x

            initial_chain_params = self._spread_walkers(
                walkers, initial_params, np.array(self.gp.get_parameter_bounds())
            )

        every_samples = 500
        # This will be useful to testing convergence
        old_tau = np.inf

        # parallelize only if cores > 1

        pool = Pool(cores) if cores > 1 else None
        self.converged = False
        sampler = emcee.EnsembleSampler(
            walkers, self._ndim, self._log_probability, pool=pool
        )
        for sample in sampler.sample(
            initial_chain_params, iterations=max_steps, progress=progress
        ):
            # Only check convergence every 100 steps
            if sampler.iteration % every_samples:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            self._autocorr.append(np.mean(tau))

            # Check convergence
            if (
                np.all(tau * 100 < sampler.iteration)
                and np.all(np.abs(old_tau - tau) / tau < 0.01)
                and converge
            ):
                print("Convergence reached after %d samples!" % sampler.iteration)
                self.converged = True
                break
            old_tau = tau
        if pool is not None:
            pool.close()
            pool.join()  # Ensures all worker processes are cleaned up

        self._tau = tau
        mean_tau = np.mean(tau)

        if not self.converged:
            warnings.warn(
                f"The chains did not converge after {sampler.iteration} iterations!"
            )
            # tau will be very large here, so let's reduce the numbers
            thin = int(mean_tau / 4)
            discard = (
                int(mean_tau) * 5
            )  # avoid blowing up if discard is larger than the number of samples, this happens if the fit has not converged

        else:
            discard = int(mean_tau * 40)
            if discard > max_steps:
                discard = int(mean_tau * 10)
            thin = int(mean_tau / 2)

        self._loglikelihoods = sampler.get_log_prob(
            discard=discard, thin=thin, flat=True
        )
        self._mcmc_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        self._sampler = sampler

    def _spread_walkers(
        self,
        walkers: int,
        parameters: np.array,
        bounds: np.array,
        percent: float = 0.1,
        max_attempts: int = 20,
    ) -> np.array:
        """Spread the walkers using a Gaussian distribution around a given set of parameters.

        Parameters
        ----------
        walkers : int
            Number of walkers.
        parameters : np.array
            Parameters around which the chains need to be spread.
        bounds : np.array
            Corresponding bounds (min,max) for each of the parameters.
        percent : float, optional
            By which percentage factor (0-1) scale the initial parameters for the standard deviation, by default 0.1 (i.e. 10%)
        max_attempts : int, optional
            Attempts before giving up (if the parameters fall always outside the imposed bounds), by default 20.

        Returns
        -------
        np.array
            A set of randomized samples for the chains

        Raises
        ------
        ValueError
            If the percent parameter is not between 0 and 1 (inclusive).
        """

        if percent < 0 or percent > 1:
            raise ValueError(
                "The 'percent' parameter must be between 0 and 1 (inclusive)."
            )

        std = np.abs(parameters) * percent
        initial_samples = np.random.normal(
            parameters, std, size=(walkers, len(parameters))
        )
        # Replace None with -inf and inf
        bounds = np.array(
            [
                (
                    -np.inf if lower is None else lower,
                    np.inf if upper is None else upper,
                )
                for lower, upper in bounds
            ]
        )
        factors_lower = np.where(bounds[:, 0] > 0, 1.05, 0.95)
        factors_upper = np.where(bounds[:, 1] > 0, 0.95, 1.05)

        for i in range(walkers):

            for attempt in range(max_attempts):
                # Generate random values centered around the best-fit parameters
                if np.all(
                    np.logical_and(
                        bounds[:, 0] <= initial_samples[i],
                        initial_samples[i] <= bounds[:, 1],
                    )
                ):
                    break  # If the walker is within bounds, continue to the next walker
                else:
                    # Regenerate the walker if it is outside the bounds
                    initial_samples[i] = np.random.normal(parameters, std)
            if attempt == max_attempts - 1:
                warnings.warn(
                    "Some walkers are out of bounds! Setting them to values close to the bounds"
                )

                out_of_bounds_lower = initial_samples[i] < bounds[:, 0]
                out_of_bounds_upper = initial_samples[i] > bounds[:, 1]
                # Set the values for walkers outside the lower bounds to 1.05 * lower bound

                initial_samples[i][out_of_bounds_lower] = np.broadcast_to(
                    bounds[:, 0] * factors_lower, initial_samples[i].shape
                )[out_of_bounds_lower]
                # Set the values for walkers outside the upper bounds to 0.95 * upper bound

                initial_samples[i][out_of_bounds_upper] = np.broadcast_to(
                    bounds[:, 1] * factors_upper, initial_samples[i].shape
                )[out_of_bounds_upper]
        return initial_samples

    def _get_rstat(self, burnin: int = None) -> np.array:
        """Calculate convergence criterion from Gelman & Rubin 1992; Gelman et al. 2004.
        Values close to 1 indicate convergence.

        Parameters
        ----------
        burnin : int, optional
            Number of samples to burn prior to the estimation, by default uses the
            autocorrelation time to estimate a suitable number.

        Returns
        -------
        np.array
            Array containing the convergence criterion statistic per chain.

        Raises
        ------
        RuntimeError
            If the posteriors have not been derived.
        """

        if self._sampler is None:
            raise RuntimeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to populate the attributes."
            )
        # calculate R stat
        if burnin is None:
            # discard 10 times the mean autocorrelation time
            burnin = int(np.mean(self.tau)) * 10
        samples = self._sampler.get_chain(discard=burnin)

        whithin_chain_variances = np.var(
            samples, axis=0
        )  # this has nwalkers, ndim (one per chain and param)

        samples = self._sampler.get_chain(flat=True, discard=burnin)
        between_chain_variances = np.var(samples, axis=0)
        return whithin_chain_variances / between_chain_variances[np.newaxis, :]

    def generate_from_posteriors(
        self,
        nsims: int = 10,
        cpus: int = 8,
        pdf: str = "Gaussian",
        sigma_noise: float = None,
    ) -> List[GappyLightcurve]:
        """Generates lightcurves by sampling from the MCMC posteriors.

        Parameters
        ----------
        nsims : int, optional
            Number of lightcurve simulations to perform, by default 10.
        cpus : int, optional
            Number of cpus to use for parallelisation, by default 8
        pdf : str, optional
            PDF for the simulations: "Gaussian", "Lognormal" or "Uniform", by default "Gaussian"
        sigma_noise : float, optional
            Standard deviation for the (Gaussian) noise, by default None.

        Returns
        -------
        List[GappyLightcurve]
            List containing lightcurves sampled from the MCMC posterirors.


        Raises
        ------
        RuntimeError
            If the posteriors have not been derived.

        """

        if self._mcmc_samples is None:
            raise RuntimeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to calling this method."
            )
        if nsims >= len(self._mcmc_samples):
            warnings.warn(
                "The number of simulation requested (%d) is higher \
                    than the number of posterior samples (%d), \
                    so many samples will be drawn more than once"
            )

        # get some parameter combinations at random
        param_samples = self._mcmc_samples[
            np.random.randint(len(self._mcmc_samples), size=nsims)
        ]
        warnings.simplefilter("ignore")
        with Pool(processes=cpus, initializer=np.random.seed) as pool:
            lightcurves = pool.map(
                partial(
                    self._generate_lc_from_params,
                    pdf=pdf,
                    sigma_noise=sigma_noise,
                ),
                param_samples,
            )
        return lightcurves

    def _generate_lc_from_params(
        self, parameters: np.array, pdf: str, sigma_noise: float
    ) -> GappyLightcurve:
        """Generate lightcurve from a set of posterior parameter samples.

        Parameters
        ----------
        parameters : np.array
            MCMC posterior paramter sample.
        pdf : str
            A string defining the probability distribution (Gaussian, Lognormal or Uniform).
        sigma_noise : float
            Standard deviation for the (Gaussian) noise, by default None

        Returns
        -------
        GappyLightcurve
            Lightcurve sampled from the MCMC posterirors.
        """
        self.gp.set_parameter_vector(parameters)
        psd_model = self.gp.kernel.get_psd
        simulator = self._lightcurve.get_simulator(
            psd_model, pdf, sigma_noise=sigma_noise
        )
        rates = simulator.generate_lightcurve()
        noisy_rates, dy = simulator.add_noise(rates)
        lc = GappyLightcurve(self._lightcurve.times, noisy_rates, dy)
        return lc

    @property
    def autocorr(self) -> List[float]:
        """Get the autocorrelation time.

        Returns
        -------
        List[float]
            Autocorrelation time.
        """
        if self._autocorr is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to populate the attributes."
            )
        return self._autocorr

    @property
    def sampler(self):
        if self._loglikelihoods is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to populate the attributes."
            )
        return self._sampler

    @property
    def loglikelihoods(self):
        if self._loglikelihoods is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to populate the attributes."
            )
        return self._loglikelihoods

    @property
    def max_loglikelihood(self):
        if self._loglikelihoods is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to populate the attributes."
            )

        return np.max(self._loglikelihoods)

    @property
    def max_parameters(self):
        """Return the parameters that maximize the loglikehood"""
        if self._mcmc_samples is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                    derive_posteriors prior to populate the attributes."
            )
        return self._mcmc_samples[np.argmax(self._loglikelihoods)]

    @property
    def median_parameters(self):
        """Return the median parameters from the thinned and burned chains"""
        if self._mcmc_samples is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                derive_posteriors prior to populate the attributes."
            )
        return np.median(self._mcmc_samples, axis=0)

    @property
    def parameter_names(self):
        return self.gp.get_parameter_names()

    @property
    def tau(self):
        """The autocorrelation time of the chains"""
        if self._mcmc_samples is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                derive_posteriors prior to populate the attributes."
            )
        return self._tau

    @property
    def mcmc_samples(self):
        """The mcmc samples"""
        if self._mcmc_samples is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run \
                derive_posteriors prior to populate the attributes."
            )
        return self._mcmc_samples

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
