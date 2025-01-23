# @Author: Andrés Gúrpide <agurpide>
# @Date:   05-02-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-02-2022
import numpy as np
import astropy.units as u
from scipy.optimize import minimize
from lmfit.models import LinearModel, Model


def chose_ls_model(frequencies, powers):
    outfit_break = linear_fit_break(frequencies, powers)
    outfit = linear_fit(frequencies, powers)
    if outfit_break.bic < outfit.bic:
            print("Best fit model: Linear Fit with Break")
            return outfit_break
    else:
        print("Best fit model: Linear Fit")
        return outfit


def chose_frequency_range(frequencies, power):
    """Chose frequency range based on a simple heuristic which consist of fitting a broken powerlaw to the LS. The break sets the high frequency end."""
    best_model = chose_ls_model(frequencies, power)
    if best_model.nvarys == 3:
        best_params = best_model.params
        maximum_frequency = np.exp(best_params.get("x_break"))
        if hasattr(frequencies, "unit"):
            maximum_frequency *= frequencies.unit
        print("Break found in the LS: restricting high frequency range to %.2f d" % ( 1/ maximum_frequency.to(1/u.d)).value)

    else:
        print("No break found in the LS: High frequency range won't be modified.")
        maximum_frequency =  np.max(frequencies)
    return best_model, maximum_frequency


def brokenpl(x, intercept, slope, x_break):
    "Broken power law that flattens above the break"
    res = np.zeros(x.shape)
    for ii, xx in enumerate(x):
        if xx < x_break:
            res[ii] = intercept + slope * (xx - x_break)
        else:
            res[ii] = intercept
    return res


def linear_fit_break(frequencies, powers):
    """Fit a linear model with a break to the power spectrum in log-log"""
    if hasattr(frequencies, "unit"):
        log_f = np.log(frequencies.value)
    else:
        log_f = np.log(frequencies)

    if hasattr(powers, "unit"):
        log_p = np.log(powers.value)
    else:
        log_p = np.log(powers)
    power_model = Model(brokenpl)
    power_model.set_param_hint("slope", min=-4, max=0., value=-1)
    power_model.set_param_hint("intercept", min=-np.inf, value=1)
    power_model.set_param_hint("x_break", min=np.min(log_f), max=np.max(log_f), value=np.mean(log_f))
    # fit
    params = power_model.make_params()
    out_fit = power_model.fit(log_p, params, x=log_f)
    return out_fit


def linear_fit(frequencies, powers):
    if hasattr(frequencies, "unit"):
        log_f = np.log10(frequencies.value)
    else:
        log_f = np.log10(frequencies)

    if hasattr(powers, "unit"):
        log_p = np.log10(powers.value)
    else:
        log_p = np.log10(powers)
    power_model = LinearModel(prefix="pow_")
    power_model.set_param_hint("pow_slope", min=-4, max=0.5, value=-1,
                             vary=True)
    power_model.set_param_hint("pow_intercept", min=-np.inf, value=np.mean(log_p),
                             vary=True)
    # fit
    params = power_model.guess(x=log_f, data=log_p)
    out_fit = power_model.fit(log_p, params, x=log_f)
    return out_fit


def s_statistic(observed_powers, model_powers):
    """Equation A.3 from Vaughan+2003 (see also Stella+1997)"""
    S = np.sum(np.log(model_powers) + observed_powers / model_powers)
    return S


def s_stat_powerlaw(x, frequencies, observed_powers):
    """Minimize S statistic"""
    model_powers = x[0] * frequencies ** x[1]
    return s_statistic(observed_powers, model_powers)


def minimize_powerlaw(frequencies, observed_powers):
    """Minimize powerlaw using S stat (Equation A.3 from Vaughan+2003)"""
    bnds = ((0, np.inf), (-4, 0))
    res = minimize(s_stat_powerlaw, [np.mean(observed_powers), -1], args=(frequencies, observed_powers), bounds=bnds, method='L-BFGS-B')
    return res.x


def fit_lomb_scargle(frequencies, powers, sigma=1):
    """Fit the lomb Scargle with a powerlaw model (it actually performs a linear fit in log-log space)

    Parameters
    ----------
    frequencies: array-like or Quantity
        Frequencies of the power spectrum, without the 0-frequency term (and the Nyquist term for even datasets).
    powers: array-like or Quantity
        Powers of the PSD with the same length as frequencies
    sigma: float
        Sigma confidence level for the uncertainty estimation. Set to 0 to skip uncertainty estimation
    """
    out_fit = linear_fit(frequencies, powers)
    # best params
    psd_slope = out_fit.params.get("pow_slope")
    psd_norm = 10**(out_fit.params.get("pow_intercept"))

    # uncertainties if lmfit converged
    if out_fit.errorbars and sigma> 0:
        conf_interval = out_fit.conf_interval(sigmas=[sigma])
        psd_slope_err = np.abs(conf_interval["pow_slope"][0][1] - conf_interval["pow_slope"][1][1])
        psd_norm_err = np.abs(10**(conf_interval["pow_intercept"][0][1])  -
                                       10**(conf_interval["pow_intercept"][1][1]))
    else:
        psd_slope_err = 0
        psd_norm_err = 0
    return psd_slope, psd_slope_err, psd_norm, psd_norm_err


def fit_psd_powerlaw(frequencies, powers):
    """Fit the PSD using the method from Vaughan+2005.
    The 0 frequency and power and the Nyquist frequency and power (only for even datasets) should be removed prior to calling this method. uncertanties are estimated using the analytical formulae
    given in the quoted paper.

    Parameters
    ----------
    frequencies: array-like or Quantity
        Frequencies of the power spectrum, without the 0-frequency term (and the Nyquist term for even datasets).
    powers: array-like or Quantity
        Powers of the PSD with the same length as frequencies
    Returns
    -------
    psd_slope:float
        Slope of the powerlaw of the PSD
    psd_slope_err: float
        Uncertainty on the slope
    psd_norm: float
        Powerlaw normalization
    psd_norm_err
        Uncertainty on the powerlaw normalization
    """
    out_fit = linear_fit(frequencies, powers)
    # determine uncertanties given by the same paper
    n_prime = len(powers)
    sigma_2 = np.pi**2 / (6 * np.log(10)**2)
    log_f = np.log10(frequencies)
    log_f_2_sum = np.sum(np.power(log_f, 2))
    delta = n_prime * log_f_2_sum  - np.sum(log_f) ** 2
    psd_slope_err = np.sqrt(n_prime * sigma_2 / delta)
    psd_log10norm_err = np.sqrt(sigma_2 * log_f_2_sum / delta)
    # best params
    psd_slope = out_fit.params.get("pow_slope")
    lognorm = out_fit.params.get("pow_intercept") + 0.25068
    dlog_conf = lognorm - psd_log10norm_err
    psd_norm = 10** lognorm
    psd_norm_err = psd_norm - 10**dlog_conf

    return psd_slope, psd_slope_err, psd_norm, psd_norm_err