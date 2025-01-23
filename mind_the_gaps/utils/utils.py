# @Author: Andrés Gúrpide <agurpide>
# @Date:   05-02-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-02-2022
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import SineModel, ConstantModel
from scipy.optimize import minimize


def phase_fold(timestamps, y, folding_frequency, dy=None, time_0=0, n_bins=10):
    """Fold the lightcurve using the frequency given
    Parameters
    ----------
    timestamps: array
        Timestamps of the lightcurve
    y: array
        Rates or y values of the lightcurve
    folding_frequency: float
        The frequency for the folding
    time_0: float
        Start time if any
    dy: array
        Uncertanties on the measurements
    n_bins: int
        Number of bins

    Returns
    -------
    bin_means:Array
        The mean of the bin in same units as rate
    bin_stds: Array
        The error of each bin (added in quadrature)
    two_phase_bins: Array
        The bins, for two cycles
    """
    phases = (timestamps - time_0) * folding_frequency
    phases = phases % 1
    phased_bins = np.array([int(np.floor(phase * n_bins)) for phase in phases])
    bins = np.arange(0, n_bins)
    bin_means = [y[phased_bins == i].mean() for i in bins]
    bin_stds = [np.sqrt(np.sum(dy[phased_bins == i] ** 2)) / len(dy[phased_bins == i]) for i in bins]
    bin_means = np.hstack([bin_means, bin_means])
    bin_stds = np.hstack([bin_stds, bin_stds])
    bins = bins / n_bins + 0.05
    two_phase_bins = np.hstack([bins, bins + 1])
    return bin_means, bin_stds, two_phase_bins


def fit_sines(timestamps, rates, frequency=None, dy=None, max_sines=5):
    """Fit a series of harmonics to the timeseries"""

    i = 0
    new_bic = 0
    old_bic = 1

    sinemodel = SineModel(prefix="sine%d_" %i) + ConstantModel(prefix="constant_")
    sinemodel.set_param_hint("constant_c", value=np.mean(rates), vary=True, min=0, max=np.max(rates))

    while i<max_sines and new_bic < old_bic:
        old_bic = new_bic
        if frequency is not None:
            sinemodel.set_param_hint("sine%d_frequency" % i, value=2 * np.pi * frequency * (i+1), vary=False)
        # tie the phases
        if i>0:
            sinemodel.set_param_hint("sine%d_shift" % i, expr="sine%d_shift" % (i-1))

        out_fit = sinemodel.fit(rates, x=timestamps, weights=1 / dy)
        new_bic = out_fit.bic
        i+= 1
        sinemodel += SineModel(prefix="sine%d_" %i)
    print("Best fit for %d sinusoids" % (i))

    return out_fit


def detrend(t, y, stds=None, plot=False):
    """Detrend the data by subtracting a linear fit

    Parameters
    ----------
    t: array
        Timestamps
    y: array
        Measurements
    stds: array
        Standard deviation of the measurements to be used as weights of the data measurements
    plot: boolean
        Whether to produce plots
    """
    if stds is not None:
        w = 1 / stds
    else:
        w = None
    # Highest power is returned first
    (p), sum_residuals, rank, singular_values, rcond = np.polyfit(t, y, deg=1, w=w, full=True)
    if plot:
        fig, (axes) = plt.subplots(2, 1, sharex=True, gridspec_kw={"hspace": 0, "wspace": 0})
        axes[0].errorbar(t, y, yerr=stds, ls="None", color="black", fmt="+")
        axes[0].plot(t, p[0] * t + p[1], color="red", lw=2, label="Linear fit")
        axes[0].set_ylabel("y")
        axes[0].legend()
        axes[1].errorbar(t, y - (p[0] * t + p[1]), yerr=stds, fmt="+", ls="None", color="black")
        axes[1].axhline(y=0, ls="--", color="blue")
        axes[1].set_ylabel("LinearFit - Data")
        axes[1].set_xlabel("Time")
        fig.savefig("linear_fit.png")
        fig, ax = plt.subplots(1)
        plt.errorbar(t, y - (p[0] * t + p[1]), yerr=stds, ls="None", color="black", fmt="+")
        plt.xlabel("Time")
        plt.ylabel("Detrended y")
        fig.savefig("detrended_data.png")

    return y - (p[0] * t + p[1]), p[0] * t + p[1]


def psd_ar1(tau, dt, df, frequencies, data_variance):
    """Derive theoretical psd of the AR(1) process as given in Equation 2 in Schulz and Mudelsee 2002

    Parameters
    ----------
    tau: float
        The autocorrelation timescale coefficient
    dt: float
        Average sampling time
    frequencies: array
        Array of frequencies to compute the PSD
    data_variance: float
        The variance of the data estimated from the PSD of the data (see Schulz and Mudelsee 2002)
    """
    rho = np.exp(- dt / tau)
    rho_2 = rho ** 2
    # gredth <- (1 - rhosq) /(1 + rhosq - 2 * rho * cos(seq(from = 0, to = pi, length.out = nfreq)))
    gredth = (1 - rho_2) / (1 - 2 * rho * np.cos((np.linspace(0, np.pi, len(frequencies)))) + rho_2)
    # scale G to have the same variance as for the real data
    gredth = (data_variance) / (sum(gredth) * df) * gredth
    return gredth


def mudelsee_residuals(timestamps, rates, tau):
    """Calculate residuals of the AR(1) fit using Equation 6 from Mudelsee et al. 2002.

    Parameters:
    -----------
    timestamps: array
        The timestamps of the time series
    rates: array
        The independent variable
    tau: float
        The tau obtained from the fit (Mudelsee et al. 2002)
    """

    return rates[1:] - rates[:-1] * np.exp(- (timestamps[1:] - timestamps[:-1]) / tau)


def mudelsee_fit(timestamps, detrended_data, errors=None):
    """Estimate a (tau) using the least-square method proposed by Mudelsee 2002 Equation 3
    Parameters
    timestamps: array Quantity
    detrended_data: array Quantity
        Detrended data
    erros: array Quantity
        The 1 sigma errors of the data (if any)
    """
    time_diff = np.diff(timestamps)
    avg_dt = np.mean(time_diff)
    a_0 = np.exp(-1) # 1 day
    std_rate = np.std(detrended_data)
    xscalt = detrended_data / std_rate
    timeseriesMNP = xscalt[:-1]
    timeseriesM1 = xscalt[1:]
    rho = sum(timeseriesMNP * timeseriesM1) / sum(timeseriesMNP * timeseriesMNP)
    print("rho = %.4f" % rho)
    scalt = -np.log(rho) / avg_dt
    tscalt = timestamps * scalt
    # the error is dropping the first one because the first term (x_1 - N(0,1))/sigma_1 does not contribute to determine a
    #errors[1:], taking into account the errors is tricky cause the data has been detrended
    print("Warning: errors are always ignore in the Mudelsee minimization")
    res_lsq = minimize(mudelsee_least_squares, a_0, args=(np.diff(tscalt), timeseriesMNP, timeseriesM1, None),
                       bounds=[(0, 1)], tol=3e-10)

    if res_lsq.success:
        a = res_lsq.x[0]
        print("a = %.10f" % a)
        tau = - 1 / (np.log(a) * scalt)
        #tau = 526299.2
        print("Minimization of tau successful. Tau found %.5f s, %.2f days" % (tau, tau / 3600 / 24))
        residuals = mudelsee_residuals(timestamps, detrended_data, tau)
        return tau, residuals
    else:
        raise Exception(res_lsq.message)


def mudelsee_least_squares(a, time_diff, timeseriesMNP, timeseriesM1, stds=None):
    """Estimate a (tau) using the least-square method proposed by Mudelsee 2002 Equation 3

    Parameters
    ---------
    a: float
        Initial guess for a (-1/ln(tau))
    time_diff: array
        The differences between the sampling timestamps
    timeseriesMNP: array
        The measurements except for the last data point
    timeseriesM1: array
        The measurements except for the first data point
    stds: array
        Standard deviation of the measurements (excluding first data point) to be used as weights for the minimization
    """
    if stds is None:
        stds = np.ones(len(timeseriesMNP))
    if (a > 0):
        tmp = timeseriesM1 - timeseriesMNP * a ** time_diff
    elif (a < 0):
        tmp = timeseriesM1 + timeseriesMNP * (-a) ** time_diff
    else:
        tmp = timeseriesM1

    return sum((tmp / stds) ** 2)


def simulate_ar1(tau, timestamps, variance, mean=0, trend=None, nsimulations=10, plot=False, n_plot=5, outdir="fake_ar1"):
    """Simulate a number of timeseries using an AR(1) process

    Parameters
    ----------
    tau: float
        Characteristic time of the AR(1) process
    timestamps: array
        Time stamps of the data
    variance: float
        Variance for the AR(1) process (tested)
    mean: float
        Mean of the AR(1) process (tested if no trend is added)
    trend: array or value
    Linear trend of the data to be added to the AR(1) simulations (optional)
    """
    ar1 = np.empty((nsimulations, len(timestamps)))
    for simulation in np.arange(nsimulations):
        noise = np.random.normal(0, np.sqrt(1 - np.exp(-2 * np.diff(timestamps) / tau)) * np.sqrt(variance))
        ar1_timeseries = np.empty(len(timestamps))
        ar1_timeseries[0] = np.random.normal(0, 1) * np.sqrt(variance)
        for i in np.arange(1, len(ar1_timeseries)):
            # there is a difference in the index of the noise component with respect to Mudelsee et al. 2002 (1) cause we do not have the fist created noise in the noise array
            ar1_timeseries[i] = ar1_timeseries[i - 1] * np.exp(-(timestamps[i] - timestamps[i - 1]) / tau) + noise[i - 1]
        #if trend is not None:
        ar1[simulation] = ar1_timeseries + mean
        fac_prev = float((simulation / nsimulations) * 100)
        print("\rProgress => {:2f}%\r".format(fac_prev), end ="\r")

    if plot:
        fig, ax = plt.subplots(1)
        plt.xlabel("Time (days)")
        plt.ylabel("Shifted count rate (ct/s)")
        shift = 0
        for simulation in np.arange(n_plot):
            plt.plot(timestamps / 3600 / 24, ar1[simulation] + shift, ls="None", marker="+")
            shift += np.mean(ar1[simulation])
        plt.savefig("%s/ar_1_fake.png" % outdir)
    return ar1
