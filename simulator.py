# @Author: Andrés Gúrpide <agurpide>
# @Date:   28-03-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-03-2022
import numpy as np
import warnings
from lmfit.models import LognormalModel, Model
from scipy.stats import norm, poisson
from stingray import Lightcurve
from .stats import kraft_pdf
import numexpr as ne
import pyfftw
from astropy.stats import poisson_conf_interval


def add_poisson_noise(rates, exposures, background_counts=None, bkg_rate_err=None):
    """Add Poisson noise and estimate uncertainties"""

    if background_counts is None:
        background_counts = np.zeros(len(rates), dtype=int)
    if bkg_rate_err is None:
        bkg_rate_err = np.zeros(len(rates))

    total_counts = rates * exposures + background_counts

    total_counts_poiss = np.random.poisson(total_counts)

    net_counts = total_counts_poiss - background_counts #  frequentists

    dy = np.sqrt((np.sqrt(total_counts_poiss) / exposures)**2 + bkg_rate_err**2)

    return net_counts  / exposures, dy


def add_kraft_noise(rates, exposures, background_counts=None, bkg_rate_err=None, kraft_counts=15):
    """Add Poisson/Kraft noise to a given count rates rates based on a real lightcurve and estimate the uncertainty

    Parameters
    ----------
    rates: array
        The count rates per second
    bkg_counts: array
        The number of background counts. None will assume 0s
    exposures: array
        In seconds
    bkg_rate_err: array
        Error on the background rate. None will assume 0s
    kraft_counts: float
        Threshold counts below which to use Kraft+91 posterior probability distribution
    This method was tested for speed against a single for loop (instead of three list comprehension) and it was found to be faster using the lists (around 10% reduction in time from the for loop approach))
    """
    if bkg_rate_err is None:
        bkg_rate_err = np.zeros(len(rates))

    if background_counts is None:
        background_counts = np.zeros(len(rates), dtype=int)

    net_rates, dy = add_poisson_noise(rates, exposures, background_counts, bkg_rate_err)
    total_counts = net_rates * exposures + background_counts

    #  frequentists bins
    net_counts = total_counts - background_counts #  frequentists
    upper_limits = net_rates / bkg_rate_err < 1 # frequentists upper limit

    # bayesian bins
    expression = total_counts < kraft_counts
    if np.any(expression):
        # calculate the medians
        pdf = kraft_pdf(a=0, b=35)
        net_counts[expression] = [pdf(counts, bkg_counts_).median() for counts, bkg_counts_ in zip(np.round(total_counts[expression]), background_counts[expression])]
        net_rates[expression] = net_counts[expression] / exposures[expression]

        # uncertainties (round to nearest integer number of counts)
        lower_confs, upper_confs = poisson_conf_interval(total_counts[expression].astype(int), "kraft-burrows-nousek",
                                            background=background_counts[expression], confidence_level=0.68)
        dy[expression] = (upper_confs - lower_confs) / 2 / exposures[expression] # bayesian

        upper_limits[expression] = lower_confs==0 # bayesian upper limit

    return net_rates, dy, upper_limits


def add_kraft_noise_old_delete(rates, exposures, background_counts=None, bkg_rate_err=None, kraft_counts=15):
    """Add Poisson/Kraft noise to a given count rates rates based on a real lightcurve and estimate the uncertainty

    Parameters
    ----------
    rates: array
        The count rates per second
    bkg_counts: array
        The number of background counts. None will assume 0s
    exposures: array
        In seconds
    bkg_rate_err: array
        Error on the background rate. None will assume 0s
    kraft_counts: float
        Threshold counts below which to use Kraft+91 posterior probability distribution
    This method was tested for speed against a single for loop (instead of three list comprehension) and it was found to be faster using the lists (around 10% reduction in time from the for loop approach))
    """
    total_counts = rates * exposures + background_counts # here use the decimal values for the background np.round(

    total_counts_poiss = np.random.poisson(total_counts)

    # estimate uncertainties
    dy = np.zeros(len(total_counts_poiss))
    net_counts = np.zeros(len(total_counts_poiss))

    expression = total_counts_poiss < kraft_counts

    if bkg_rate_err is None:
        bkg_rate_err = np.zeros(len(rates[~expression]))
        dy[~expression] = np.sqrt((np.sqrt(total_counts_poiss[~expression]) / exposures[~expression])**2 + bkg_rate_err**2)
    else:
        dy[~expression] = np.sqrt((np.sqrt(total_counts_poiss[~expression]) / exposures[~expression])**2 + bkg_rate_err[~expression]**2)

    if background_counts is None:
        background_counts = np.zeros(len(rates))

    net_counts[~expression] = total_counts_poiss[~expression] - background_counts[~expression]

    pdf = kraft_pdf(a=0, b=35) # store pdf in memory
    # Replace by kraft+91 pdf where necessary
    dists = [pdf(counts, bkg_counts_) for counts, bkg_counts_ in zip(np.round(total_counts_poiss[expression]), np.ceil(background_counts[expression]))]
    net_counts[expression] = [dist.median() for dist in dists]
    dy[expression] = [(dist.ppf(0.84) - dist.ppf(0.16)) / 2 for dist in dists] # if subtract the two is equivalent to subtract the 50% value and doing the mean, this is faster than using "interval" / 2
    dy[expression] /=  exposures[expression]
    return net_counts / exposures, dy


def cut_downsample(lc, timestamps,bin_exposures, N=None):
    """Cut and downsample input lightcurve based on the input timestamps and exposures"""

    if np.isscalar(bin_exposures):
        start_time = timestamps[0] - bin_exposures
        end_time = timestamps[-1] + 1.5 * bin_exposures
    else:
        start_time = timestamps[0] - bin_exposures[0]
        end_time = timestamps[-1] + 1.5 * bin_exposures[-1] # add small offset to ensure the first and last bins are properly behaved when imprinting the sampling pattern

    duration = end_time - start_time

    if N is not None:
        lc_cut = get_segment(lc, duration, N)
    else:
        lc_cut = cut_random_segment(lc, duration)

    downsampled_rates = downsample(lc_cut, timestamps, bin_exposures)

    return downsampled_rates


def draw_positive(pdf):
    """Dummy method to ensure all samples of count rates are positively defined
    pdf: scipy.stats.rv_continuous
        PDF where to draw values from (which can contain negative values, e.g. a Gaussian function)

    Returns
    -------
    value:float
        One sample from the pdf
    """
    value = pdf.rvs()
    return(value if value>0 else draw_positive(pdf))


def PoissonModel(x, lamb, amplitude):
    return amplitude * poisson.pmf(x, lamb)


def fit_pdf(time_series, nbins=20):
    """
    Fit the time series probability density function.
    Parameters
    ---------
    time_series: array or Quantity
    n_bins: int
        Number of bins of the histogram to be fitted

    Returns
    -------
    out_fit: lmift.ModelResult
        Best fit model
    prefixes: list
        List of prefixes for each model component of the pdf
    """
    models = []
    if hasattr(time_series, "unit"):
        time_series_ = time_series.value
    else:
        time_series_ = time_series

    n, bins = np.histogram(time_series_, bins=nbins, density=True)
    x_values = bins[:-1] + np.diff(bins) / 2

    # Fit with one Gaussian Model
    pdf_model = LognormalModel(prefix='pdf')
    # from the Log Normal distribution (see e.g. https://en.wikipedia.org/wiki/Log-normal_distribution)
    sigma = np.sqrt(np.log(1 + (np.std(time_series_) / np.mean(time_series_))**2))
    center = np.log(np.mean(time_series_)**2 / (np.sqrt(np.var(time_series_) + np.mean(time_series_)**2)))

    pdf_model.set_param_hint("pdfcenter", value=center)
    pdf_model.set_param_hint("pdfsigma", value=sigma)
    pdf_model.set_param_hint("pdfamplitude", min=0, value=1)
    #pars = pdf_model.guess(n, x=x_values) # causes a bogus functionality
    out_fit = pdf_model.fit(n, x=x_values)
    bic_1_gauss = out_fit.bic
    print(out_fit.fit_report())
    print("BIC value for one Lognormal model: %.2f" % bic_1_gauss)
    # Fit with two Gaussian components
    pdf_model_2 = LognormalModel(prefix='pdf1') + LognormalModel(prefix='pdf2')
    pdf_model_2.set_param_hint("pdf1center", value=center)
    pdf_model_2.set_param_hint("pdf2center", value=center)
    pdf_model_2.set_param_hint("pdf1sigma", value=sigma)
    pdf_model_2.set_param_hint("pdf2sigma", value=sigma * 1.5)
    pdf_model_2.set_param_hint("pdf1amplitude", min=0, value=1)
    pdf_model_2.set_param_hint("pdf2amplitude", min=0, value=0.5)
    out_fit_2 = pdf_model_2.fit(n, x=x_values)
    bic_2_gauss = out_fit_2.bic
    print(out_fit_2.fit_report())
    print("BIC value for two Lognormal models: %.2f" % bic_2_gauss)

    if bic_1_gauss < bic_2_gauss:
        print("Best fit model: one Lognormal")
        return out_fit, ["pdf"]
    else:
        print("Best fit model: two Lognormals")
        return out_fit_2, ["pdf1", "pdf2"]


def get_fft_slow(N, dt, model):
    """Get DFT
    Parameters
    ----------
    N: int
        Number of datapoints
    dt: float
        Binning in time in seconds
    model: astropy.model
        The model for the PSD
    """
    freqs = np.fft.rfftfreq(N, dt) * 2 * np.pi
    #generate real and complex parts from gaussian distributions
    real, im = np.random.normal(0, size=(2, N // 2 + 1))
    complex_fft = np.empty(len(freqs), dtype=np.cfloat)
    complex_fft[1:] = (real + im * 1j)[1:] * np.sqrt(.5 * model(freqs[1:]))

    # assign whatever real number to the 0, it does not matter as the lightcurve is normalized later
    complex_fft[0] = 1
    # In case of even number of data points f_nyquist is only real (see e.g. Emmanoulopoulos+2013 or Timmer & Koening+95)
    if N % 2 == 0:
        complex_fft[-1] = np.real(complex_fft[-1])
    return complex_fft



def get_fft(N, dt, model):
    """Get DFT
    Parameters
    ----------
    N: int
        Number of datapoints
    dt: float
        Binning in time in seconds
    model: astropy.model
        The model for the PSD
    """
    freqs = np.fft.rfftfreq(N, dt) * 2 * np.pi
    #generate real and complex parts from gaussian distributions
    real, im = np.random.normal(0, size=(2, N // 2 + 1))
    complex_fft = np.empty(len(freqs), dtype=np.cfloat)
    complex_fft[1:] = (real + im * 1j)[1:] * np.sqrt(.5 * model(freqs[1:]))

    # assign whatever real number to the total number of photons, it does not matter as the lightcurve is normalized later
    complex_fft[0] = 1e6
    # In case of even number of data points f_nyquist is only real (see e.g. Emmanoulopoulos+2013 or Timmer & Koening+95)
    if N % 2 == 0:
        complex_fft[-1] = np.real(complex_fft[-1])
    return complex_fft


def simulate_lightcurve_numpy(timestamps, psd_model, dt, extension_factor=50):
    """Simulate a lightcurve regularly sampled N times longer than original using the algorithm of Timmer & Koenig+95

    Parameters
    ----------
    timestamps: array
        Timestamps in same units as dt
    psd_model: astropy.model
        The model for the PSD
    dt: float
        Binning with which simulate the lightcurve, same units as timestamps
    extension_factor: int
        How many times longer than original
    """
    if extension_factor < 1:
        raise ValueError("Extension factor needs to be higher than 1")

    duration = timestamps[-1] - timestamps[0]
    # generate timesctamps sampled at the median exposure longer than input lightcurve by extending the end
    sim_timestamps = np.arange(timestamps[0] - 2 * dt,
                               timestamps[0] + duration * extension_factor + dt,
                               dt)

    n_datapoints = len(sim_timestamps)

    complex_fft = get_fft_slow(n_datapoints, dt, psd_model)

    countrate = np.fft.irfft(complex_fft, n=n_datapoints)
    # the power gets diluted due to the sampling, bring it back by applying this factor
    # the sqrt(2pi) factor enters because of celerite
    countrate *= np.sqrt(n_datapoints) * np.sqrt(dt) * np.sqrt(np.sqrt(2 * np.pi))

    return Lightcurve(sim_timestamps, countrate, input_counts=True, skip_checks=True, dt=dt, err_dist="gauss")



def simulate_lightcurve(timestamps, psd_model, dt, extension_factor=50):
    """Simulate a lightcurve regularly sampled N times longer than original using the algorithm of Timmer & Koenig+95

    Parameters
    ----------
    timestamps: array
        Timestamps
    psd_model: astropy.model
        The model for the PSD. Has to take angular frequencies
    dt: float
        Binning to which simulate the lightcurve (same units as timestamps)
    extension_factor: int
        How many times longer than original
    """
    if extension_factor < 1:
        raise ValueError("Extension factor must be greater than 1")

    duration = timestamps[-1] - timestamps[0]
    # generate timesctamps sampled at the median exposure longer than input lightcurve by extending the end
    sim_timestamps = np.arange(timestamps[0] - 2 * dt,
                               timestamps[0] + duration * extension_factor + dt,
                               dt)

    n_datapoints = len(sim_timestamps)

    complex_fft = get_fft(n_datapoints, dt, psd_model)

    countrate = pyfftw.interfaces.numpy_fft.irfft(complex_fft, n=n_datapoints) # it does seem faster than numpy although only slightly

    # the power gets diluted due to the sampling, bring it back by applying this factor
    # the sqrt(2pi) factor enters because of celerite
    countrate *= np.sqrt(n_datapoints * dt * np.sqrt(2 * np.pi))

    return Lightcurve(sim_timestamps, countrate, input_counts=True, skip_checks=True, dt=dt, err_dist="gauss") # gauss is needed as some counts will be negative


def get_segment(lc, duration, N):
    """Get N segment of the input lightcurve. Starts at 0"""
    start = lc.time[0] + duration * (N)
    return lc.truncate(start=start, stop=start + duration, method="time")


def cut_random_segment(lc, duration):
    """Cut segment from the input lightcurve of given duration"""
    shift = np.random.uniform(lc.time[0], lc.time[-1] - duration)# - lc.time[0]
    return lc.truncate(start=shift, stop=shift + duration, method="time")


def imprint_sampling_pattern(lightcurve, timestamps, bin_exposures):
    """Modify the input lightcurve to have the input sampling pattern (timestamps and exposures) provided
    lightcurve: stingray.lightcurve
        Lightcurve to which imprint the given sampling pattern
    timestamps: array
        New timestamps of the new sampling
    bin_exposures: array or scalar
        Exposures of the timestamps. Either as a float or array (or 1 item array)
    Parameters
    ---------
    """
    half_bins = bin_exposures / 2

    if np.isscalar(half_bins):
        gti = [(time - half_bins, time + half_bins) for time in timestamps]
    elif len(half_bins) == len(timestamps):
        gti = [(time - half_bin, time + half_bin) for time, half_bin in zip(timestamps, half_bins)]
    else:
        raise ValueError("Half bins length (%d) must have same length as timestamps (%d) or be a scalar." % (len(half_bins), len(timestamps)))

    # get rid of all bins in between timestamps using Stingray
    lc_split = lightcurve.split_by_gti(gti, min_points=0)
    # stingray pints a warning for lightcurves with 0 data point that can be ignored
    # get average count rates for the entire subsegment corresponding to each timestamps
    return np.array([lc.meanrate for lc in lc_split])


def downsample(lc, timestamps, bin_exposures):
    """Downsample the lightcurve into the new binning (regular or not)
    Parameters
    ----------
    lc: Lightcurve
        With the old binning
    timestmaps: array or Quantity
        The new timestamps for the lightcurve
    bin_exposures: array or Quantity
        Exposure times of each new bin
    Returns the downsampled count rates
    """
    if len(lc.time) == len(timestamps):
        return lc.countrate

    if np.isscalar(bin_exposures):
        start_time = timestamps[0] - bin_exposures
    else:
        start_time = timestamps[0] - bin_exposures[0]
    # return the lightcurve as it is
    shifted = lc.shift(-lc.time[0] + start_time)
    downsampled_rates = imprint_sampling_pattern(shifted, timestamps, bin_exposures)
    return downsampled_rates


def tk95_sim(timestamps, psd_model, mean, sim_dt=None, extension_factor=50, bin_exposures=None):
    """Simulate one lightcurve using the method from Timmer and Koenig+95 with the input sampling (timestamps) and using a red-noise powerlaw model.

    Parameters
    ----------
    timestamps: array
        The timestamps of the lightcurve to simulate
    psd_model: astropy.model
        Model for the PSD
    mean: float or Quantity
        Desired mean for the final lightcurve
    std: float or Quanity
        Desierd standard deviation of the final lightcurve
    sim_dt: float
        Binning to use for the simulated lightcurve (desired final dt is given by bin_exposures).
        If not given it is computed from the mean difference of the input timestamps
    bin_exposures: float or array-like
        Exposure time of each bin in the final lightcurve (float for constant, array for irregular bins)
    """
    if extension_factor < 1:
        raise ValueError("Extension factor needs to be higher than 1")

    epsilon = 0.99 # to avoid numerically distinct but equal

    wrong = np.count_nonzero(np.diff(timestamps) < sim_dt * epsilon)
    if wrong >0:
        raise ValueError("%d timestamps differences are below sampling time!" % wrong)

    if sim_dt is None:
        sim_dt = np.mean(np.diff(timestamps))

    if bin_exposures is None:
        bin_exposures = np.mean(np.diff(timestamps))

    if np.any(bin_exposures < sim_dt):
        raise ValueError("Bin exposures must larger than sim_dt")

    lc = simulate_lightcurve(timestamps, psd_model, sim_dt, extension_factor)

    rates = cut_downsample(lc, timestamps, mean, bin_exposures)
    # adjust the mean, the variance is set on the PSD
    rates += mean - np.mean(rates)

    return rates


def check_pdfs(weights, pdfs):
    """Check input pdf parameters"""
    if len(pdfs) != len(weights):
        raise ValueError("Number of weights (%d) must match number of pdfs (%d)" % (weights.size, pdfs.size))

    if round(np.sum(weights),5) > 1:
        raise ValueError("Weights for the probability density distributions must be = 1. (%.20f)" % np.sum(weights))


def E13_sim(timestamps, psd_model, pdfs=[norm(0, 1)], weights=[1], sim_dt=None, extension_factor=20,
            bin_exposures=None, max_iter=300):
    """Simulate lightcurve using the method from Emmanolopoulos+2013

    timestamps: array_like
        Timestamps of the desired time series
    psd_model: astropy.model
        The model for the PSD
    pdfs: array_like
        Probability density distribution to be matched
    weights: array_like
        Array containing the weights of each of the input distributions
    sim_dt: float or Quantity
        Time sampling to use when simulating the data
    extension_factor: float
        Factor by which to extend the original lightcurve to introduce rednoise leakage.
    bin_exposures: array_like
        Exposure time of each sample
     max_iter: int
        Number of iterations for the lightcurve creation if convergence is not reached
        (More complex models (or greater beta values) or pdfs require larger max_iter e.g. 100 for beta < 1,
        200 beta = 1, 500 or more for beta >=2 or bending powerlaws)
    """

    if extension_factor < 1:
        raise ValueError("Extension factor needs to be higher than 1")

    check_pdfs(weights, pdfs)

    half_bins = bin_exposures / 2

    if sim_dt is None:
        sim_dt = np.mean(np.diff(timestamps))

    # step i
    lc = simulate_lightcurve(timestamps, psd_model, sim_dt, extension_factor=extension_factor) # aliasing and rednoise leakage effects are taken into account here

    if np.isscalar(half_bins):
        start_time = timestamps[0] - 2 * half_bins
        end_time = timestamps[-1] + 3 * half_bins
    else:
        start_time = timestamps[0] - 2 * half_bins[0]
        end_time = timestamps[-1] + 3 * half_bins[-1] # add small offset to ensure the first and last bins are properly behaved when imprinting the sampling pattern

    duration = end_time - start_time

    lc_cut = cut_random_segment(lc, duration)

    return E13_sim_TK95(lc_cut, timestamps, pdfs, weights, bin_exposures, max_iter)


def E13_sim_TK95(lc, timestamps, pdfs=[norm(0, 1)], weights=[1], exposures=None, max_iter=400):
    """Simulate lightcurve using the method from Emmanolopoulos+2013
    lc: Lightcurve
        Regularly sampled TK95 generated lightcurve with the desired PSD (and taking into account, if desired, rednoise leakage and aliasing effects)
    timestmaps: array
        New desired timestmaps (in seconds)
    pdfs: array_like
        Probability density distribution to be matched
    weights: array_like
        Array containing the weights of each of the input distributions
    exposures: array
        Exposures in seconds of the new bins. If not given assumed equal to lc.dt (it can be array or scalar)
     max_iter: int
        Number of iterations for the lightcurve creation if convergence is not reached (Greater beta values requirer larger max_iter e.g. 100 for beta < 1, 200 beta = 1, 500 or more for beta >=2)
    """

    check_pdfs(weights, pdfs)

    if exposures is None:
        exposures = lc.dt

    n_datapoints = len(lc)
    # step ii draw from distribution
    if len(pdfs) > 1:
        draw = np.random.choice(np.arange(len(weights)), n_datapoints, p=weights)
        xsim = np.array([pdfs[i].rvs() for i in draw])
    else:
        xsim = pdfs[0].rvs(size=n_datapoints)

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
    while (not np.allclose(xsim, xsim_adjust, rtol=1e-03) and iteration < max_iter):
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

    downsampled_rates = downsample(lc, timestamps, exposures)

    return downsampled_rates


def E13_sim_TK95_wrong(tk_rate, pdfs=[norm(0, 1)], weights=[1], max_iter=100):
    """Simulate lightcurve using the method from Emmanolopoulos+2013
    lc: Lightcurve
        Regularly sampled TK95 generated lightcurve with the desired PSD (and taking into account, if desired, rednoise leakage and aliasing effects)
    timestmaps: array
        New desired timestmaps (in seconds)
    pdfs: array_like
        Probability density distribution to be matched
    weights: array_like
        Array containing the weights of each of the input distributions
    half_bins: array
        Exposures in seconds (so new desired dt, it can be array or scalar)
     max_iter: int
        Number of iterations for the lightcurve creation if convergence is not reached (Greater beta values requirer larger max_iter e.g. 100 for beta < 1, 200 beta = 1, 500 or more for beta >=2)
    """
    check_pdfs(weights, pdfs)

    n_datapoints = len(tk_rate)
    # step ii draw from distribution
    draw = np.random.choice(np.arange(len(weights)), n_datapoints, p=weights)
    xsim = np.array([pdfs[i].rvs() for i in draw])
    dft_sim = np.fft.rfft(xsim)
    phases_sim = np.angle(dft_sim)

    complex_fft = np.fft.rfft(tk_rate)

    amplitudes_norm = np.absolute(complex_fft) / (n_datapoints // 2 + 1) # see Equation A2 from Emmanolopoulos+13

    # step iii
    dft_adjust = amplitudes_norm * np.exp(1j * phases_sim)
    xsim_adjust = pyfftw.interfaces.numpy_fft.irfft(dft_adjust, n=n_datapoints)
    # step iv
    xsim_adjust[np.argsort(-xsim_adjust)] = xsim[np.argsort(-xsim)]

    iteration = 0
    pyfftw.interfaces.cache.enable()
    # start loop
    while (not np.allclose(xsim, xsim_adjust, rtol=1e-02) and iteration < max_iter):
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
        print("Lightcurve did not converge after %d iterations, PDF might be inaccurate. Try increase the maximum number of iterations" % iteration)
    return xsim


def E13_sim_wrong(timestamps, psd_model, pdfs=[norm(0, 1)], weights=1, sim_dt=None, extension_factor=50,
            bin_exposures=None, max_iter=500):
    """Simulate lightcurve using the method from Emmanolopoulos+2013
    timestamps: array_like or Quantity
        Timestamps of the desired time series
    psd_model: astropy.model
        The model for the PSD
    pdfs: array_like
        Probability density distribution to be matched
    weights: array_like
        Array containing the weights of each of the input distributions
    sim_dt: float or Quantity
        Time sampling to use when simulating the data
    extension_factor: float
        Factor by which to extend the original lightcurve to introduce rednoise leakage.
     max_iter: int
        Number of iterations for the lightcurve creation if convergence is not reached (Greater beta values requirer larger max_iter e.g. 100 for beta < 1, 200 beta = 1, 500 or more for beta >=2)
    """

    if extension_factor < 1:
        raise ValueError("Extension factor needs to be higher than 1")

    weights = np.asarray(weights)
    pdfs = np.asarray(pdfs)
    if pdfs.size != weights.size:
        raise ValueError("Number of weights must match number of pdfs")

    if round(np.sum(weights),5) > 1:
        raise ValueError("Weights for the probability density distributions must be = 1. (%.20f)" % np.sum(weights))


    if sim_dt is None:
        sim_dt = np.mean(np.diff(timestamps))

    n_datapoints = len(timestamps)
    # step ii draw from distribution (we do this first to get the mean and std)
    draw = np.random.choice(np.arange(len(weights)), n_datapoints, p=weights)
    xsim = np.array([pdfs[i].rvs() for i in draw])
    dft_sim = np.fft.rfft(xsim)
    phases_sim = np.angle(dft_sim)

    # step i
    mean, std = np.mean(xsim), np.std(xsim)
    tk_rate = tk95_sim(timestamps, psd_model, mean, std, sim_dt, bin_exposures=bin_exposures,
                       extension_factor=extension_factor) # aliasing and rednoise leakage effects are taken into account here

    complex_fft = np.fft.rfft(tk_rate)

    amplitudes_norm = np.absolute(complex_fft) / (n_datapoints // 2 + 1) # see Equation A2 from Emmanolopoulos+13

    # step iii
    dft_adjust = amplitudes_norm * np.exp(1j * phases_sim)
    xsim_adjust = np.fft.irfft(dft_adjust, n=n_datapoints)
    # step iv
    xsim_adjust[np.argsort(-xsim_adjust)] = xsim[np.argsort(-xsim)]

    iteration = 0
    #plt.figure()
    # start loop
    while (not np.allclose(xsim, xsim_adjust, rtol=1e-02) and iteration < max_iter):
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
        print("Lightcurve did not converge after %d iterations, PDF might be inaccurate. Try increase the maximum number of iterations" % iteration)

    return xsim


def E13_sim_TK95_numpy(lc, timestamps, pdfs=[norm(0, 1)], weights=[1], half_bins=None, max_iter=100):
    """Simulate lightcurve using the method from Emmanolopoulos+2013
    lc: Lightcurve
        Regularly sampled TK95 generated lightcurve with the desired PSD (and taking into account, if desired, rednoise leakage and aliasing effects)
    timestmaps: array
        New desired timestmaps (in seconds)
    pdfs: array_like
        Probability density distribution to be matched
    weights: array_like
        Array containing the weights of each of the input distributions
    half_bins: array
        Exposures in seconds (so new desired dt, it can be array or scalar)
     max_iter: int
        Number of iterations for the lightcurve creation if convergence is not reached (Greater beta values requirer larger max_iter e.g. 100 for beta < 1, 200 beta = 1, 500 or more for beta >=2)
    """
    check_pdfs(weights, pdfs)

    if half_bins is None:
        half_bins = lc.dt

    n_datapoints = len(lc)
    # step ii draw from distribution
    draw = np.random.choice(np.arange(len(weights)), n_datapoints, p=weights)
    xsim = np.array([pdfs[i].rvs() for i in draw])
    dft_sim = np.fft.rfft(xsim)
    phases_sim = np.angle(dft_sim)

    complex_fft = np.fft.rfft(lc.countrate)

    amplitudes_norm = np.absolute(complex_fft) / (n_datapoints // 2 + 1) # see Equation A2 from Emmanolopoulos+13

    # step iii
    dft_adjust = ne.evaluate("amplitudes_norm * exp(1j * phases_sim)")
    xsim_adjust = np.fft.irfft(dft_adjust, n=n_datapoints)
    # step iv
    xsim_adjust[np.argsort(-xsim_adjust)] = xsim[np.argsort(-xsim)]

    iteration = 0
    #plt.figure()
    # start loop
    while (not np.allclose(xsim, xsim_adjust, rtol=1e-02) and iteration < max_iter):
        xsim = xsim_adjust
        # step ii Fourier transform of xsim
        dft_sim = np.fft.rfft(xsim)

        phases_sim = np.angle(dft_sim)
        # replace amplitudes
        dft_adjust = ne.evaluate("amplitudes_norm * exp(1j * phases_sim)")
        # inverse fourier transform the spectrally adjusted time series
        xsim_adjust = np.fft.irfft(dft_adjust, n=n_datapoints)
        # iv amplitude adjustment --> ranking sorted values
        xsim_adjust[np.argsort(-xsim_adjust)] = xsim[np.argsort(-xsim)]

        #diff = np.sum((xsim - xsim_adjust) ** 2)
        #plt.scatter(iteration, diff, color="black")
        iteration += 1
    if iteration == max_iter:
        print("Lightcurve did not converge after %d iterations, PDF might be inaccurate. Try increase the maximum number of iterations" % iteration)
    # UNTIL HERE IT WORKS METHOD TESTED!!
    lc.countrate = xsim

    downsampled_rates = downsample(lc, timestamps, 2 * half_bins)

    return downsampled_rates


def E13_sim_unused(timestamps, psd_model, pdfs=[norm(0, 1)], weights=1, sim_dt=None, extension_factor=50, bin_exposures=None, max_iter=500):
    """Simulate lightcurve using the method from Emmanolopoulos+2013

    timestamps: array_like or Quantity
        Timestamps of the desired time series
    psd_model: astropy.model
        The model for the PSD
    pdfs: array_like
        Probability density distribution to be matched
    weights: array_like
        Array containing the weights of each of the input distributions
    sim_dt: float or Quantity
        Time sampling to use when simulating the data
    extension_factor: float
        Factor by which to extend the original lightcurve to introduce rednoise leakage.
    max_iter: int
        Number of iterations for the lightcurve creation if convergence is not reached (Greater beta values requirer larger max_iter e.g. 100 for beta < 1, 200 beta = 1, 500 or more for beta >=2)
    """

    if extension_factor < 1:
        raise ValueError("Extension factor needs to be higher than 1")

    weights = np.asarray(weights)
    pdfs = np.asarray(pdfs)
    if pdfs.size != weights.size:
        raise ValueError("Number of weights must match number of pdfs")

    if round(np.sum(weights),5) > 1:
        raise ValueError("Weights for the probability density distributions must be = 1. (%.20f)" % np.sum(weights))


    if sim_dt is None:
        sim_dt = np.mean(np.diff(timestamps))

    if bin_exposures is None:
        bin_exposures = np.mean(np.diff(timestamps))

    duration = timestamps[-1] - timestamps[0]

    # generate timestamps sampled at the median exposure longer than input lightcurve by extending the end
    sim_timestamps = np.arange(timestamps[0],
                               timestamps[0] + duration * extension_factor + sim_dt,
                               sim_dt)

    n_datapoints = len(sim_timestamps)

    complex_fft = get_fft(n_datapoints, sim_dt, psd_model)

    amplitudes_norm = np.absolute(complex_fft) / (n_datapoints // 2 + 1) # see Equation A2 from Emmanolopoulos+13
    # step ii draw from distribution
    draw = np.random.choice(np.arange(len(weights)), n_datapoints, p=weights)
    xsim = np.array([pdfs[i].rvs() for i in draw])

    mean, std = np.mean(xsim), np.std(xsim)
    dft_sim = np.fft.rfft(xsim)
    phases_sim = np.angle(dft_sim)
    # step iii
    dft_adjust = amplitudes_norm * np.exp(1j * phases_sim)
    xsim_adjust = np.fft.irfft(dft_adjust, n_datapoints)
    # step iv
    xsim_adjust[np.argsort(-xsim_adjust)] = xsim[np.argsort(-xsim)]
    xsim_2 = xsim_adjust
    iteration = 0
    #plt.figure()
    # start loop
    while (not np.allclose(xsim, xsim_2, rtol=1e-02) and iteration < max_iter):
        xsim = xsim_2
        # step ii Fourier transform of xsim
        dft_sim = np.fft.rfft(xsim)
        phases_sim = np.angle(dft_sim)
        # replace amplitudes
        dft_adjust = amplitudes_norm * np.exp(1j * phases_sim)
        # inverse fourier transform the spectrally adjusted time series
        xsim_adjust = np.fft.irfft(dft_adjust, n_datapoints)
        # iv amplitude adjustment --> ranking sorted values
        xsim_adjust[np.argsort(-xsim_adjust)] = xsim[np.argsort(-xsim)]
        xsim_2 = xsim_adjust
        diff = np.sum((xsim - xsim_2) ** 2)
        #plt.scatter(iteration, diff, color="black")
        iteration += 1
    if iteration == max_iter:
        print("Lightcurve did not converge after %d iterations, PDF might be inaccurate. Try increase the maximum number of iterations" % iteration)
    # use ``input_counts=False`` to input the count range, i.e. counts/second, otherwise use counts/bin
    # in this case we have counts/bin
    lc = Lightcurve(sim_timestamps, xsim, skip_checks=True, dt=sim_dt, input_counts=True)
    return lc


def gauss_white_noise_sim(mean, std, N):
    samples = np.random.normal(mean, std, N)


def _normalize(time_series, mean, std):
    """Normalize lightcurve to a desired mean and standard deviation
    Parameters
    ----------
    time_series: Quantity or array
        Y values of the time series to which you want to normalize
    mean: Quantity or float
        Desired mean of the generated lightcurve to match.
    std_dev:Quantity or float
        Desired standard deviation of the simulated lightcurve to match.
    """
    # this sets the same variance
    if hasattr(mean, "unit"):
        time_series *= std.value / np.std(time_series)
        # this sets the same mean
        time_series += mean.value - np.mean(time_series)
        return time_series * mean.unit
    else:
        time_series *= std / np.std(time_series)
        # this sets the same mean
        time_series += mean - np.mean(time_series)
        return time_series
