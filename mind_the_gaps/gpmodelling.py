# @Author: Andrés Gúrpide <agurpide>
# @Date:   05-02-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-05-2024
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike

from mind_the_gaps.lightcurves import GappyLightcurve
import celerite
import celerite.modeling
from celerite.modeling import ConstantModel
from mind_the_gaps.models import LinearModel, GaussianModel
import emcee
from multiprocessing import Pool
from functools import partial
from scipy.optimize import minimize
import warnings


class GPModelling:
    """
    The interface for Gaussian Process (GP) modeling. Currently uses celerite.
    """
    meanmodels = ["linear", "constant", "gaussian"]

    def __init__(
            self,
            lightcurve: GappyLightcurve,
            kernel,
            mean_model: str = None
    ):
        """

        Parameters
        ----------
        lightcurve
            An instance of a lightcurve
        model: celerite.terms.Term
            The model to be fitted to the lightcurve
        mean_model
            Mean model. If given it will be fitted, otherwise assumed the mean value.
            Available implementations are Constant, Linear and Gaussian.
        """
        self._lightcurve = lightcurve

        meanmodel, fit_mean = self._build_mean_model(mean_model)

        self.gp = celerite.GP(kernel, mean=meanmodel, fit_mean=fit_mean)
        # self.gp = GaussianProcess(kernel, self._lightcurve.times, diag=self._lightcurve.dy**2) --> Tiny GP
        # initialize GP ( # You always need to call compute once.)
        self.gp.compute(self._lightcurve.times, self._lightcurve.dy + 1e-12)
        self.initial_params = self.gp.get_parameter_vector()
        self._ndim = len(self.initial_params)
        self._autocorr = []
        self._loglikelihoods = None
        self._mcmc_samples = None


    def _build_mean_model(
            self,
            meanmodel: str = None
    ) -> Tuple[celerite.modeling.Model, bool]:
        """
        Construct the GP mean model based on lightcurve properties and input string

        Parameters
        ----------
        meanmodel
            A string indicating the mean model to be used. If None, a constant model is used.

        Returns
        -------
        celerite.modeling.Model
            A celerite model
        bool
            Whether the mean model is fitted or not
        """
        maxy = np.max(self._lightcurve.y)

        if meanmodel is None:
            # no fitting case
            meanmodel = ConstantModel(self._lightcurve.mean,
                        bounds=[(np.min(self._lightcurve.y), maxy)])
            return meanmodel, False

        elif meanmodel.lower() not in GPModelling.meanmodels:
            raise ValueError("Input mean model %s not implemented! Only \n %s \n are available" % (meanmodel, "\t".join(GPModelling.meanmodels)))

        elif meanmodel.lower()=="constant":
            meanlabels = ["$\mu$"]
            meanmodel = ConstantModel(self._lightcurve.mean,
                        bounds=[(np.min(self._lightcurve.y), maxy)])
            return meanmodel, True

        elif meanmodel.lower()=="linear":
            slope_guess = np.sign(self._lightcurve.y[-1] - self._lightcurve.y[0])
            minindex = np.argmin(self._lightcurve.times)
            maxindex = np.argmax(self._lightcurve.times)
            slope_bound = (self._lightcurve.y[maxindex] - self._lightcurve.y[minindex]) / (self._lightcurve.times[maxindex] - self._lightcurve.times[minindex])
            if slope_guess > 0 :
                min_slope = slope_bound
                max_slope = -slope_bound
            else:
                min_slope = -slope_bound
                max_slope = slope_bound
            slope = np.cov(self._lightcurve.times, self._lightcurve.y)[0, 1] / np.var(self._lightcurve.times)
            meanmodel = LinearModel(0, 1.5,
                                    bounds=[(-np.inf, np.inf), (-np.inf, np.inf)])
            meanlabels = ["$m$", "$b$"]

        elif meanmodel.lower()=="gaussian":
            sigma_guess = (self._lightcurve.duration) / 2
            amplitude_guess = (maxy - np.min(self._lightcurve.y)) * np.sqrt(2 * np.pi)* sigma_guess
            mean_guess = self._lightcurve.times[len(self._lightcurve.times)//2]
            meanmodel = GaussianModel(mean_guess, sigma_guess, amplitude_guess,
                                      bounds=[(self._lightcurve.times[0], self._lightcurve.times[-1]), (0, self._lightcurve.duration),
                                      (maxy * np.sqrt(2 * np.pi) * self._lightcurve.duration, 50 * maxy * np.sqrt(2 * np.pi) * self._lightcurve.duration)])

            meanlabels = ["$\mu$", "$\sigma$", "$A$"]

        return meanmodel, True


    def _log_probability(self, params: List) -> float:
        """
        Logarithm of the posteriors of the Gaussian process

        Parameters
        ----------
        params
            List of parameters of the GP at which to calculate the posteriors
        y: ArrayLike
            The dataset (count rates, lightcurves, etc)
        gp: celerite.GP
            An instance of the Gaussian Process solver

        Returns
        -------
        float
            Returns the log of the posterior
            https://celerite.readthedocs.io/en/stable/tutorials/modeling/
        """

        self.gp.set_parameter_vector(params)

        lp = self.gp.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.gp.log_likelihood(self._lightcurve.y)


    def _neg_log_like(self, params):
        """Function to calculate the negative log likelihood of the GP given a set of parameters
        
        Parameters
        ----------
        params
            List of parameters of the GP at which to calculate the posteriors
        
        Returns
        -------
        float
            Returns the negative log likelihood of the GP given the input parameters
        """
        self.gp.set_parameter_vector(params)
        return -self.gp.log_likelihood(self._lightcurve.y)


    def fit(
            self,
            initial_params: ArrayLike = None
    ):
        """
        Fit the GP by running a minimization rutine using scipy.optimize.minimize with the L-BFGS-B method.

        Parameters
        ----------
        initial_params
            Set of initial paramteres for the minimization. If not given uses those given at initialization of the class.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Returns the output of scipy.minimize
        """
        if initial_params is None:
            initial_params = self.initial_params

        solution = minimize(self._neg_log_like, initial_params, method="L-BFGS-B",
                            bounds=self.gp.get_parameter_bounds())
        return solution


    def derive_posteriors(
            self,
            initial_chain_params: ArrayLike = None,
            fit: bool = True,
            converge: bool = True,
            max_steps: int = 10000,
            convergence_steps: int = 500,
            walkers: int = 12,
            cores: int = 6,
            progress: bool = True
    ):
        """
        Derive GP Posteriors

        Parameters
        ----------
        initial_chain_params
            Set of initial parameters for the chains. If not given will use randomized from the parameters given at initialization
        fit:
            Whether to run a minimization routine prior to running the MCMC chains
        converge:
            Whether to stop the chains if convergence is detected based on the autocorrelation time
        max_steps:
            Maximum number of steps for the chains
        convergence_steps:
            Number of steps to check for convergence
        walkers:
            Number of walkers for the chains
        cores:
            Number of cores for parallelization
        progress:
            Whether to show a progress bar during the MCMC sampling. If fitting multiple lightcurves in parallel best to turn it off.
        """
        # set the initial parameters if not given
        if initial_chain_params is None:
            if not fit:
                initial_params = self.initial_params

            else:
                solution = self.fit(self.initial_params)
                initial_params = solution.x
        
            initial_chain_params = self.spread_walkers(walkers, initial_params, np.array(self.gp.get_parameter_bounds()))
        
        every_samples = convergence_steps
        # This will be useful to testing convergence
        old_tau = np.inf
        # parallelize only if cores > 1
        pool =  Pool(cores) if cores >1 else None   
        self.converged = False
        sampler = emcee.EnsembleSampler(walkers, self._ndim, self._log_probability, pool=pool)
        for sample in sampler.sample(initial_chain_params, iterations=max_steps, progress=progress):
            # Only check convergence every 100 steps
            if sampler.iteration % every_samples:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            self.autocorr.append(np.mean(tau))

            # Check convergence
            if np.all(tau * 100 < sampler.iteration) and np.all(np.abs(old_tau - tau) / tau < 0.01) and converge:
                print("Convergence reached after %d samples!" % sampler.iteration)
                self.converged = True
                break
            old_tau = tau
        if pool is not None:
            pool.close()
            pool.join() # Ensures all worker processes are cleaned up

        self._tau = tau
        mean_tau = np.mean(tau)

        if not self.converged:
            warnings.warn(f"The chains did not converge after {sampler.iteration} iterations!")
            # tau will be very large here, so let's reduce the numbers
            thin = int(mean_tau / 4)
            discard = int(mean_tau) * 5 # avoid blowing up if discard is larger than the number of samples, this happens if the fit has not converged

        else:
            discard = int(mean_tau * 40)
            if discard > max_steps:
                discard = int(mean_tau * 10)
            thin = int(mean_tau / 2)

        self._loglikelihoods = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
        self._mcmc_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        self._sampler = sampler


    def spread_walkers(
            self,
            walkers: int,
            parameters: ArrayLike,
            bounds: List[Tuple[float, float]],
            percent: float = 0.1,
            max_attempts: int = 20
    ) -> ArrayLike:
        """Spread the walkers using a Gaussian distribution around a given set of parameters

        Parameters
        ----------
        walkers
            Number of walkers
        parameters
            Parameters around which the chains need to be spread
        bounds
            Bounds (min, max) for each of the parameters (in same order)
        percent
            The percentage (0-1) by which to scale the initial parameters for the standard deviation
            of the Gaussians. Default: 0.1 (i.e., 10%).
        max_attempts
            Attempts before giving up (if the parameters fall always outside the imposed bounds)

        Returns
        -------
        ArrayLike
            Returns a set of randomized samples for the chains
        """
        if percent < 0 or percent > 1:
            raise ValueError("The 'percent' parameter must be between 0 and 1 (inclusive).")

        std = np.abs(parameters) * percent
        initial_samples = np.random.normal(parameters, std, size=(walkers, len(parameters)))
        # Replace None with -inf and inf
        bounds = np.array([(-np.inf if lower is None else lower, 
                        np.inf if upper is None else upper) 
                       for lower, upper in bounds])
        factors_lower = np.where(bounds[:, 0] > 0, 1.05, 0.95)
        factors_upper = np.where(bounds[:,1] > 0, 0.95, 1.05)

        for i in range(walkers):

            for attempt in range(max_attempts):
                # Generate random values centered around the best-fit parameters
                if np.all(np.logical_and(bounds[:, 0] <= initial_samples[i], initial_samples[i] <= bounds[:, 1])):
                    break  # If the walker is within bounds, continue to the next walker
                else:
                    # Regenerate the walker if it is outside the bounds
                    initial_samples[i] = np.random.normal(parameters, std)
            if attempt == max_attempts - 1:
                warnings.warn("Some walkers are out of bounds! Setting them to values close to the bounds")

                out_of_bounds_lower = initial_samples[i] < bounds[:, 0]
                out_of_bounds_upper = initial_samples[i] > bounds[:, 1]
                # Set the values for walkers outside the lower bounds to 1.05 * lower bound
                
                initial_samples[i][out_of_bounds_lower] = np.broadcast_to(bounds[:, 0]  * factors_lower , initial_samples[i].shape)[out_of_bounds_lower]
                # Set the values for walkers outside the upper bounds to 0.95 * upper bound
                
                initial_samples[i][out_of_bounds_upper] = np.broadcast_to(bounds[:, 1] * factors_upper, initial_samples[i].shape)[out_of_bounds_upper] 
        return initial_samples
    

    def standarized_residuals(
            self,
            include_noise: bool = True
    ):
        """
        Calculate the standarized residuals of the model (see e.g. Kelly et al. 2011) Eq. 49.
        You should set the gp parameters prior to calling this method

        Parameters
        ----------
        include_noise
            True to include any jitter term into the standard deviation calculation. False ignores this contribution.
        """
        pred_mean, pred_var = self.gp.predict(self._lightcurve.y, return_var=True, return_cov=False)
        if include_noise:
            pred_var+= self.gp.kernel.jitter
        std_res = (self._lightcurve.y - pred_mean) / np.sqrt(pred_var)
        return std_res


    def get_rstat(
            self,
            burnin: int = None
    ) -> ArrayLike:
        """Calculate convergence criterion from Gelman & Rubin 1992; Gelman et al. 2004.
        Values close to 1 indicate convergence.

        Parameters
        ----------
        burnin
            Number of samples to burn prior to the estimation. By default
            uses the autocorrelation time to estimate a suitable number

        Returns
        -------
        ArrayLike
            Returns an array of the value of the statistic per chain
        """
        if self._sampler is None:
            raise ValueError("Posteriors have not been derived. Please run derive_posteriors prior to populate the attributes.")
        # calculate R stat
        if burnin is None:
            # discard 10 times the mean autocorrelation time
            burnin = int(np.mean(self.tau)) * 10
        samples = self._sampler.get_chain(discard=burnin)

        whithin_chain_variances = np.var(samples, axis=0) # this has nwalkers, ndim (one per chain and param)

        samples = self._sampler.get_chain(flat=True, discard=burnin)
        between_chain_variances = np.var(samples, axis=0)
        return whithin_chain_variances / between_chain_variances[np.newaxis, :]

    @property
    def loglikelihoods(self):
        """An array containing the loglikelihoods of the MCMC samples"""
        if self._loglikelihoods is None:
            raise AttributeError("Posteriors have not been derived. Please run derive_posteriors prior to populate the attributes.")
        return self._loglikelihoods

    @property
    def autocorr(self):
        """An array containing the autocorrelation time of the MCMC samples as a function of step"""
        return self._autocorr

    @property
    def sampler(self):
        if self._loglikelihoods is None:
            raise AttributeError("Posteriors have not been derived. Please run derive_posteriors prior to populate the attributes.")
        return self._sampler

    @property
    def mcmc_samples(self):
        """An array containing the MCMC parameters for all the walkers and steps"""
        if self._mcmc_samples is None:
            raise AttributeError("Posteriors have not been derived. Please run derive_posteriors prior to populate the attributes.")
        return self._mcmc_samples

    @property
    def max_loglikelihood(self):
        """The maximum loglikelihood of the MCMC samples"""
        if self._loglikelihoods is None:
            raise AttributeError("Posteriors have not been derived. Please run derive_posteriors prior to populate the attributes.")

        return np.max(self._loglikelihoods)

    @property
    def max_parameters(self):
        """The parameters that maximize the loglikehood"""
        if self._mcmc_samples is None:
            raise AttributeError("Posteriors have not been derived. Please run derive_posteriors prior to populate the attributes.")
        return self._mcmc_samples[np.argmax(self._loglikelihoods)]

    @property
    def median_parameters(self):
        """The median parameters from the thinned and burned chains"""
        if self._mcmc_samples is None:
            raise AttributeError("Posteriors have not been derived. Please run derive_posteriors prior to populate the attributes.")
        return np.median(self._mcmc_samples, axis=0)

    @property
    def parameter_names(self):
        return self.gp.get_parameter_names()


    @property
    def k(self) -> int:
        """
        Number of variable parameters

        Returns
        -------
        int
            Number of variable parameters
        """
        return self._ndim


    @property
    def tau(self):
        """The autocorrelation time of the chains"""
        if self._mcmc_samples is None:
            raise AttributeError("Posteriors have not been derived. Please run derive_posteriors prior to populate the attributes.")
        return self._tau


    def generate_from_posteriors(
            self,
            nsims: int = 10,
            cpus: int = 8,
            pdf: str = "Gaussian",
            extension_factor: int = 2,
            sigma_noise = None
    ):
        """
        Generates lightcurves by sampling from the MCMC posteriors

        Parameters
        ----------
        nsims
            Number of lightcurve simulations to perform
        cpus
            Number of cpus to use for parallelization
        pdf
            Desired PDF for the generated lightcurves: Gaussian, Lognormal or Uniform
        extension_factor
            Extension factor for the generation of the lightcurves, to introduce rednoise leakage
        """

        if self._mcmc_samples is None:
            raise RuntimeError("Posteriors have not been derived. Please run derive_posteriors prior to calling this method.")
        if nsims >= len(self._mcmc_samples):
            warnings.warn("The number of simulation requested (%d) is higher than the number of posterior samples (%d), so many samples will be drawn more than once")

        # get some parameter combinations at random
        param_samples = self._mcmc_samples[np.random.randint(len(self._mcmc_samples), size=nsims)]
        # generate the simulator object, with a dummy kernel params for now
        simulator = self._lightcurve.get_simulator(self.gp.kernel.get_psd, pdf, sigma_noise=sigma_noise, extension_factor=extension_factor)
        warnings.simplefilter('ignore')
        with Pool(processes=cpus, initializer=np.random.seed) as pool:
            lightcurves = pool.map(partial(self._generate_lc_from_params, simulator=simulator), param_samples)
        return lightcurves
    
    def _generate_lc_from_params(
            self,
            parameters: ArrayLike,
            simulator,
    ) -> GappyLightcurve:
        """
        Internal method to generate a lightcurve from the given set of parameters
        Parameters
        ----------
        parameters
            Parameters for the GP PSD model
        simulator
            An instance of the simulator to generate the lightcurve
        Returns
        -------
        GappyLightcurve
            Returns a GappyLightcurve object with the generated lightcurve
        """
        self.gp.set_parameter_vector(parameters)
        # set the new PSD with update params
        simulator.psd_model = self.gp.kernel.get_psd
        rates = simulator.generate_lightcurve()
        noisy_rates, dy = simulator.add_noise(rates)
        lc = GappyLightcurve(self._lightcurve.times, noisy_rates, dy)
        return lc