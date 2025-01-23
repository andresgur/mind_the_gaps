import numpy as np
from scipy import stats, special
from scipy.stats import lognorm, uniform
import scipy as sp
from scipy.optimize import minimize

__all__ = ["kraft_pdf", "lognormal", "chi_cov", "chi_log_likehood", "chi_square", "create_log_normal", "create_uniform_distribution"]

class kraft_pdf(stats.rv_continuous):
    """Posterior probability function based on Kraft+1991"""
    def _argcheck(self, N, B):
        return (N >= 0) and (B>=0)

    def _pdf(self, x, N, B):
        n = np.arange(N + 1)
        C = (np.sum(np.exp(-B) * B ** n / special.factorial(n))) **-1 # tested
        return C * np.exp(-x - B) * (x + B) ** N / np.math.factorial(N)


class lognormal(stats.rv_continuous):
    """Posterior probability function for a log normal distribution as given in https://lmfit.github.io/lmfit-py/builtin_models.html#exponentialgaussianmodel"""
    def _argcheck(self, center, sigma):
        return (sigma >= 0)

    def _pdf(self, x, center, sigma):
        return  1 / (sigma * x * np.sqrt(2 * np.pi)) * np.exp(-(np.log(x) - center)**2 / (2 * sigma**2))



def fit_N(loglikehood, log_like_args=()):
    """Minimize the value of the normalization for the given log likehood
    Parameters
    ----------
    loglikehood: function
        Log likehood function
    log_like_args: arguments
        Arguments for the log likehood function
    """
    res = minimize(loglikehood, 1, args=(log_like_args), method='BFGS')
    return res.x


def chi_cov(powers_data, model_powers=None, inv_cov=None):
    """Estimate Chi^2 from Uttley+2002, but taking into account the covariance matrix, so the interdependency of the frequencies is taken into account

    powers_data: array_like
        The spectral powers of the data or the realization for which Chi^2 has to be estimated
    model_powers: array_like
        The model per frequency
    inv_cov: array_like
        The inverse covariance matrix
    Tested
    """
    data_model = powers_data - model_powers
    return np.matmul(np.matmul(data_model, inv_cov), data_model.T)


def chi_log_likehood(powers_data, model_pows=None, nyquist=False):
    """Statistic from Vaughan et al. 2005 Eq. A.3 / Emmanolopoulos+2013 A 11. The last frequency is assumed to be the Nyquist frequency
    and it is also assumed that the 0 frequency power is not given for the calculation of the parity

    powers_data: array_like
        The spectral powers of the data or the realization for the statistic has to be estimated
    model_pows: array_like
        The model
    nyquist: bool,
        Indicates whether the Nyquist frequency is included. Default False
    """

    if nyquist:
        log_like = chi_log_likehood_nonyq(powers_data[:-1], model_pows[:-1])
        # add nyquist contribution
        log_like += np.log(np.pi * powers_data[-1] * model_pows[-1]) + 2 * powers_data[-1] / model_pows[-1]
    else:
        log_like = chi_log_likehood_nonyq(powers_data, model_pows)

    return log_like


def chi_log_likehood_nonyq(powers_data, model_pows=None):
    """Statistic from Vaughan et al. 2005 Eq. A.3 / Emmanolopoulos+2013 A 11. The last frequency is assumed to be the Nyquist frequency
    and it is also assumed that the 0 frequency power is not given for the calculation of the parity

    powers_data: array_like
        The spectral powers of the data or the realization for the statistic has to be estimated
    model_pows: array_like
        The model
    """

    return 2. * np.sum(np.log(model_pows) + powers_data / model_pows)


def chi_square(powers_data, model_powers=None, sigmas=None):
    """Estimate Chi^2 from Uttley+2002

    powers_data: array_like
        The spectral powers of the data or the realization for which Chi^2 has to be estimated
    model_powers: array_like
        The value of the model per frequency
    sigmas: array_like
        The uncertainties on the model per frequency bin
    If the uncertainties are underestimated the model will fall below the data.
    """
    return np.sum(((model_powers - powers_data) / (sigmas)) **2)


def chi_square_N(powers_data, model_power=None, std_power=None):
    """Version of the chi_square log likehood but with a normalization factor that is minimized

    """
    N = fit_N(chi_square, (powers_data, model_power, std_power))
    return chi_square(N, powers_data, model_power, std_power)


def create_log_normal(mean, std):
    """Create a log normal with the desired mean and variance
    Parameters
    ----------
    mean: float,
        Mean
    std: float,
        Standard deviation
    """
    var = std**2
    mu = np.log((mean**2)/np.sqrt(var+mean**2))
    sigma = np.sqrt(np.log(var/(mean**2)+1))
    pdf = lognorm(sigma, scale=np.exp(mu))
    return pdf


def create_uniform_distribution(mean, std):
    """Create a uniform with the desired mean and variance
    Parameters
    ----------
    mean: float,
        Mean
    std: float,
        Standard deviation
    """
    var = std**2
    b = np.sqrt(3 * var) + mean
    a = 2 * mean - b
    #In the standard form, the distribution is uniform on [0, 1]. Using the parameters loc and scale, one obtains the uniform distribution on [loc, loc + scale].
    pdf = uniform(loc=a, scale=b-a)
    return pdf


def neg_log_like(params, y, gp):
    """Remove eventually"""
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)


def bic(loglikehood, n, k):
    """Calculate the Bayesian Information Criterion

    Parameters
    ----------
    loglikehood:float,
        The loglikehood of the model
    n:int,
        Number of datapoints
    k:int,
        Number of parameters

    """
    return -2. * loglikehood + k * np.log(n)

def aic(loglikehood, k):
    """Calculate the Akaike Information Criterion (Akaike 1998)
    loglikehood: float,
        The loglikehood of the model
    k: int,
        Number of model parameters

    Returns the aic
    """

    return 2 * k - 2 * loglikehood

def aicc(loglikehood, n, k):
    """Calculate the Akaike Information Criterion corrected for finite sample size

    Parameters
    ----------
    loglikehood:float,
        The loglikehood of the model
    n:int,
        Number of datapoints
    k:int,
        Number of parameters
    Returns the aic corrected for small sample size
    """
    return aic(loglikehood, k) + 2 * k * (k + 1) / (n - k - 1)
