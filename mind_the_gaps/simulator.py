# @Author: Andrés Gúrpide <agurpide>
# @Date:   28-03-2022
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 26-04-2025
from typing import Callable, Tuple, List, Literal, Dict, Union
from astropy.units import Quantity
from lmfit.model import ModelResult
from numpy.typing import ArrayLike, NDArray
from astropy.modeling import Model as AstropyModel

import astropy.modeling
import numpy as np
import warnings
from lmfit.models import LognormalModel
from scipy.stats import norm, rv_continuous
from stingray import Lightcurve
import numexpr as ne
import pyfftw
from math import sqrt, log
from mind_the_gaps.stats import create_log_normal, create_uniform_distribution
from mind_the_gaps.noise_models import PoissonNoise, GaussianNoise, KraftNoise


class BaseSimulatorMethod:
    def __init__(self, mean):
        self.meanrate = mean
        

    def adjust_pdf(self, segment)  -> Lightcurve:
        raise NotImplementedError("This method should be implemented by subclasses")


class TK95Simulator(BaseSimulatorMethod):
    def __init__(self, mean, random_state=None):
        super().__init__(mean)

    def adjust_pdf(self, segment):
        # Nothing to do here as the PDF is already Gaussian
        return segment


class E13Simulator(BaseSimulatorMethod):
    def __init__(
            self,
            mean: float,
            pdf: Literal["lognormal", "uniform", "gaussian"],
            max_iter: int = 1000,
            random_state: int = None
    ):
        super().__init__( mean)
        self.pdf = pdf
        self.max_iter = max_iter
        if self.pdf == "lognormal":
            self.pdfmethod = create_log_normal
        elif self.pdf== "uniform":
            self.pdfmethod = create_uniform_distribution
        elif self.pdf=="gaussian":
            self.pdfmethod = norm
        else:
            raise ValueError(
                "pdf must be one of 'lognormal', 'uniform', 'gaussian'"
            )

    def adjust_lightcurve_pdf(
            self,
            lc: Lightcurve,
            pdf: rv_continuous,
            max_iter: int = 400
    ) -> Lightcurve:
        """
        Adjust the PDF of the input lightcurve using the method from Emmanolopoulos+2013

        Parameters
        ----------
        lc
            Regularly sampled TK95 generated lightcurve with the desired PSD (and taking into account, if desired, rednoise leakage and aliasing effects)
        pdf
            Probability density distribution to be matched
        max_iter
            Number of iterations for the lightcurve creation if convergence is not reached (Greater beta values requirer larger max_iter e.g. 100 for beta < 1, 200 beta = 1, 500 or more for beta >=2)

        Returns
        ----------
        Lightcurve
            A lightcurve with the adjusted rates
        """
        n_datapoints = lc.n
        # step ii draw from distribution
        xsim = pdf.rvs(size=n_datapoints)

        dft_sim = pyfftw.interfaces.numpy_fft.rfft(xsim)

        phases_sim = np.angle(dft_sim)

        complex_fft = pyfftw.interfaces.numpy_fft.rfft(lc.countrate)

        amplitudes_norm = np.absolute(complex_fft) / (n_datapoints // 2 + 1) # see Equation A2 from Emmanolopoulos+13

        # step iii
        dft_adjust = ne.evaluate("amplitudes_norm * exp(1j * phases_sim)")

        xsim_adjust = pyfftw.interfaces.numpy_fft.irfft(dft_adjust, n=n_datapoints)
        # step iv
        xsim_adjust[np.argsort(-xsim_adjust)] = xsim[np.argsort(-xsim)]

        iteration = 0
        pyfftw.interfaces.cache.enable()
        #plt.figure()
        # start loop
        while (not np.allclose(xsim_adjust, xsim, rtol=1e-04) and iteration < max_iter):
            xsim = xsim_adjust
            # step ii Fourier transform of xsim
            dft_sim = pyfftw.interfaces.numpy_fft.rfft(xsim)
            phases_sim = np.angle(dft_sim)
            # replace amplitudes
            dft_adjust = ne.evaluate("amplitudes_norm * exp(1j * phases_sim)")
            # inverse fourier transform the spectrally adjusted time series
            xsim_adjust = pyfftw.interfaces.numpy_fft.irfft(dft_adjust, n=n_datapoints)
            # iv amplitude adjustment --> ranking sorted values
            xsim_adjust[np.argsort(-xsim_adjust)] = xsim[np.argsort(-xsim)]

            #diff = np.sum((xsim - xsim_adjust) ** 2)
            #plt.scatter(iteration, diff, color="black")
            iteration += 1
        if iteration == max_iter:
            warnings.warn("Lightcurve did not converge after %d iterations, PDF might be inaccurate. Try increase the maximum number of iterations" % iteration)
        # tesed until here
        lc.countrate = xsim

        return lc

    def adjust_pdf(
            self,
            segment
    ) -> Lightcurve:
        sample_std = np.std(segment.countrate)
        pdf = self.pdfmethod(self.meanrate, sample_std)
        adjusted_lc = self.adjust_lightcurve_pdf(segment, pdf, max_iter=self.max_iter)
        return adjusted_lc


class Simulator:
    """
    A class to simulate lightcurves from a given power spectral densities and flux probability density function
    """
    def __init__(
            self,
            psd_model: Union[Callable, AstropyModel],
            times: ArrayLike,
            exposures: Union[float, ArrayLike],
            mean: float,
            pdf: str = "gaussian",
            bkg_rate: Union[ArrayLike, None] = None,
            bkg_rate_err: Union[ArrayLike, None] = None,
            sigma_noise: Union[float, None] = None,
            aliasing_factor: float = 2,
            extension_factor: float = 10,
            epsilon: float = 1.001,
            max_iter: int = 400,
            random_state: Union[int, None] = None
    ):
        """
        Parameters
        ----------
        psd_model:
            PSD model to use for the lightcurve simulations. Astropy model or a function taking in angular frequencies and returning the power
        pdf:
            String defining the flux probability density function desired for the ligthcurve. If Gaussian, uses Timmer & König 1995 algorithm, otherwise uses Emmanolopoulos et al. 2013.
              Currently implemented: Gaussian, Lognormal and Uniform distributions.
        times:
            Timestamps of the lightcurve (i.e. times at the "center" of the sampling). Always in seconds
        exposures:
            Exposure time of each datapoint in seconds
        mean:
            Desired mean countrate for the lightcurve
        bkg_rate:
            Associated background rate (or flux) of each datapoint
        bkg_rate_err:
            uncertainty on background rate
        sigma_noise:
            Standard deviation for the (Gaussian) noise. If not given assumes Poisson (no bkg rates) or Kraft (when bkg rates are given) noise
        aliasing_factor:
            This defines the grid of the original, regularly-sampled lightcurve produced
            prior to resampling as min(exposure) / aliasing_factor
        extension_factor:
            Factor by which extend the initially generated lightcurve, to introduce rednoise leakage. 10 times by default
        epsilon:
            Factor (>1) by which expand the exposure times in the resampling to handle correctly numerically equal values. Normally there will be no need to vary it.
        max_inter:
            Maximum number of iterations for the E13 method. Not use if method is TK95 (Gaussian PDF)
        random_state:
            Random state to use for the random number generator. If None, uses the global random state.
        """
        if extension_factor < 1:
            raise ValueError("Extension factor must be greater than 1")
        
        if epsilon <1:
            raise ValueError("Epsilon needs to be greater than 1!")

        if np.any(exposures==0):
            raise ValueError("Some exposure times are 0!")
        else:
            # convert to array if scalar
            self._exposures = np.full(len(times), exposures) if np.isscalar(exposures) else exposures

        if pdf.lower() not in ["gaussian", "lognormal", "uniform"]:
            raise ValueError("%s not implemented! Currently implemented: Gaussian, Uniform or Lognormal")
        elif pdf.lower()=="gaussian":
            #print("Simulator will use TK95 algorithm with %s pdf" %pdf)
            self.simulator = TK95Simulator(mean)
        else:
            #print("Simulator will use E13 algorithm with %s pdf" % pdf)
            self.simulator = E13Simulator(mean, pdf.lower(), max_iter=max_iter)

        self.random_state = np.random.RandomState(random_state)
        self.sim_dt = np.min(self._exposures) / aliasing_factor

        dt = np.diff(times)
        # check that the sampling is consistent with the exposure times of each timestamp
        wrong = np.count_nonzero(dt < self.sim_dt * 0.99)
        if wrong > 0:
            raise ValueError("%d timestamps differences are below the exposure integration time! Either reduce the exposure times, or space your observations" % wrong)

                
        start_time = times[0] - dt[0] / 1.99
        end_time = times[-1] + dt[-1] # add small offset to ensure the first and last bins are properly behaved when imprinting the sampling pattern
        self.sim_duration = end_time - start_time

        # duration of the regularly and finely sampled lightcurve
        duration = (times[-1] - times[0]) * extension_factor

        # generate timesctamps for the regular, finely sampled grid and longer than input lightcurve by extending the end
        self.sim_timestamps = np.arange(start_time - self.sim_dt,
                                start_time + duration + self.sim_dt,
                                self.sim_dt)
        # variable for the fft
        self.fftndatapoints = len(self.sim_timestamps)
        self.pdf = pdf

        self.psd_model = psd_model
        self._times = times
        
        # noise
        if sigma_noise is None:
            if bkg_rate is None or np.all(bkg_rate==0):
                self.noise = PoissonNoise(self._exposures)
            else:
                self.noise = KraftNoise(self._exposures, bkg_rate * self._exposures, bkg_rate_err)
        else:
            self.noise = GaussianNoise(self._exposures, sigma_noise)


        half_bins = self._exposures / 2 * epsilon
        self.strategy = [(time - half_bin, time + half_bin) for time, half_bin in zip(times, half_bins)]

        self.mean = mean
        #print("Simulator will use %s noise" % self.noise.name)

    def __str__(self) -> str:
        sim_info = (f"Simulator(\n"
            f"  PSD Model: {self._psd_model}\n"
            f"  PDF: {self.pdf}\n)"
            f" Noise: {self.noise.name}")
        return sim_info
    
    @property
    def psd_model(self) -> Union[Callable, AstropyModel]:
        """Getter for the PSD model."""
        return self._psd_model

    @psd_model.setter
    def psd_model(
            self,
            new_psd_model: Union[Callable, AstropyModel]
    ):
        """Setter for the PSD model."""
        if not callable(new_psd_model):
            raise ValueError("PSD model must be callable (e.g., a function or Astropy model).")
        self._psd_model = new_psd_model

    def set_psd_params(
            self,
            psd_params: Dict
    ):
        """
        Set the parameters of the PSD

        Call this method prior to generate_lightcurve if you want to change the input params
        for the PSD

        Parameters
        ----------
        psd_params:dict
            Dictionary mapping parameter name to value
        """
        for par in psd_params.keys():
            setattr(self._psd_model, par, psd_params[par])

    def add_noise(
            self,
            rates: ArrayLike,
    ) -> (NDArray, NDArray):
        """
        Add noise to the input rates.

        This method applies a noise model to the input `rates`, returning the
        perturbed (noisy) rates and the associated uncertainties.

        Parameters
        ----------
        rates
            Input rates to which noise will be added.

        Returns
        -------
        noisy_rates
            The input rates after noise has been applied.
        
        dy
            The estimated uncertainties (standard deviation) on the noisy rates.
        """
        noisy_rates, dy = self.noise.add_noise(rates)
        #if self._noise_std is None:
         #   if np.all(self._bkg_rate==0):
          #      noisy_rates, dy = add_poisson_noise(rates, self._exposures)
           # else:
            #    noisy_rates, dy, upp_lims = add_kraft_noise(rates, self._exposures,
             #                                           self._bkg_rate * self._exposures,
              #                                          self._bkg_rate_err)
        #else:
            # add 0 mean Gaussian White Noise
         #   noisy_rates = rates + np.random.normal(scale=self._noise_std, size=len(rates))
            #dy = np.sqrt(rates * self._exposures) / self._exposures
          #  dy = self._noise_std * np.ones(len(rates))
            #dy = errors[np.argsort(noisy_rates)]

        return noisy_rates, dy
    
    def downsample(
            self,
            lc: Lightcurve
    ) -> NDArray:
        """
        Downsample the lightcurve into the new sampling pattern (regular or otherwise)

        Parameters
        ----------
        lc
            Regularly-sampled lightcurve to resample into the new sampling pattern

        Returns
        ----------
        NDArray
            The downsampled count rates at the input timestamps
        """

        rates = [ ]

        for bin in self.strategy:
            start, end = bin
            idxs = np.argwhere((lc.time >= start) & (lc.time < end))
        
            meanrate = np.mean(lc.countrate[idxs])
            rates.append(meanrate)
            
        return rates
    
    def simulate_regularly_sampled(self) -> Lightcurve:
        """
        Generate a regularly sampled lightcurve.

        Produce a lightcurve that is both longer in duration and 
        higher in temporal resolution than the final unevenly sampled lightcurve. 
        It simply applied the Timmer & Koenig (1995) algorithm, adjusting the mean 
        at the end.

        Returns
        -------
        lc
            A regularly sampled lightcurve generated using the TK95 algorithm.
        """
        # get a new realization of the PSD
        complex_fft = get_fft(self.fftndatapoints, self.sim_dt, self._psd_model)
        # invert it
        counts = pyfftw.interfaces.numpy_fft.irfft(complex_fft, n=self.fftndatapoints) # it does seem faster than numpy although only slightly
        # the power gets diluted due to the sampling, bring it back by applying this factor
        # the sqrt(2pi) factor enters because of celerite
        counts *= sqrt(self.fftndatapoints * self.sim_dt * sqrt(2 * np.pi))
        # set err_dist to None to avoid any warnings and issues
        lc =  Lightcurve(self.sim_timestamps, counts, input_counts=True, skip_checks=True, dt=self.sim_dt, err_dist="gauss") 
        # adjust mean, note variance is provided from the PSD
        lc.countrate = lc.countrate - lc.meanrate + self.mean 
        return lc


    def generate_lightcurve(self) -> NDArray:
        """
        Generates lightcurve with the last input PSD parameteres given.

        Note every call to this method will generate a different realization,
        even if the input parameters remain unchanged.
        All parameters are specified during the Simulator creation 

        Returns
        ----------
        NDArray
            The rates for the input timestamps
        """   
        lc = self.simulate_regularly_sampled()
        # cut a slightly longer segment than the original ligthcurve
        segment = cut_random_segment(lc, self.sim_duration)
        # shift it to match the timestamps
        shifted_lc = segment.shift(-segment.tstart + self.strategy[0][0])
        # adjust its rates to desired PDF, if TK95 (Gaussian) no need to adjust as the mean is already set
        lc_adjusted = self.simulator.adjust_pdf(shifted_lc)

        downsampled_rates = self.downsample(lc_adjusted)

        return downsampled_rates


def add_poisson_noise(
        rates: ArrayLike,
        exposures: Union[float, ArrayLike],
        background_counts: Union[ArrayLike, None] = None,
        bkg_rate_err: ArrayLike = None
) -> Tuple[ArrayLike, ArrayLike]:
    """Add Poisson noise and estimate uncertainties

    Parameters
    ----------
    rates
        Array of count rates
    exposures
        Exposure time at each bin (or a single value if same everywhere)
    background_counts
        The number of background counts at each bin (or a single value if same everywhere)
        If None, it is assumed to be zero
    bkg_rate_err
        The error on the background rate at each bin (or a single value if same everywhere)
        If None, it is assumed to be zero

    Returns
    -------
    ArrayLike
        New Poissonian modified rates
    ArrayLike
        Its uncertainties
    """

    if background_counts is None:
        background_counts = np.zeros(len(rates), dtype=int)
    if bkg_rate_err is None:
        bkg_rate_err = np.zeros(len(rates), dtype=int)

    total_counts = rates * exposures + background_counts

    total_counts_poiss = np.random.poisson(total_counts)

    net_counts = total_counts_poiss - background_counts #  frequentists

    dy = np.sqrt((np.sqrt(total_counts_poiss) / exposures)**2 + bkg_rate_err**2)

    return net_counts  / exposures, dy


def get_fft(
        N: int,
        dt: float,
        model: AstropyModel
) -> ArrayLike:
    """
    Get DFT

    Parameters
    ----------
    N
        Number of datapoints
    dt
        Binning in time in seconds
    model
        The model for the PSD

    Returns
    -------
    ArrayLike
        The complex FFT
    """
    freqs = np.fft.rfftfreq(N, dt) * 2 * np.pi
    #generate real and complex parts from gaussian distributions
    real, im = np.random.normal(0, size=(2, N // 2 + 1))
    complex_fft = np.empty(len(freqs), dtype=complex)
    complex_fft[1:] = (real + im * 1j)[1:] * np.sqrt(.5 * model(freqs[1:]))

    # assign whatever real number to the total number of photons, it does not matter as the lightcurve is normalized later
    complex_fft[0] = 1e6
    # In case of even number of data points f_nyquist is only real (see e.g. Emmanoulopoulos+2013 or Timmer & Koening+95)
    if N % 2 == 0:
        complex_fft[-1] = np.real(complex_fft[-1])
    return complex_fft

def get_segment(
        lc: Lightcurve,
        duration: float,
        N: int
) -> Lightcurve:
    """
    Get the Nth segment of the input lightcurve.

    Parameters
    ----------
    lc
        The input lightcurve from which a segment is to be drawn
    duration
        Duration of the desired segment
    N
        Integer indicating the segment to get (starting at N = 0 and ending in N +1) i.e. t_start = N * duration)

    Returns
    -------
    Lightcurve
        A selected segment of the lightcurve with the given duration

    Raises
    ------
    ValueError
        If N is negative.
    """
    if N < 0:
        raise ValueError("N must be a non-negative integer.")
    start = lc.time[0] + duration * N
    return lc.truncate(start=start, stop=start + duration, method="time")


def cut_random_segment(lc, duration):
    """Cut segment from the input lightcurve of given duration"""
    shift = np.random.uniform(lc.time[0], lc.time[-1] - duration)# - lc.time[0]
    return lc.truncate(start=shift, stop=shift + duration, method="time")