from astropy.modeling.models import custom_model
from astropy.modeling import Model
import numpy as np
from scipy.special import gamma

@custom_model
def SHO(x, S_0=1, w_0=1, Q=10):
    """Equation 20 from Foreman-Mackey+2017 (tested)
    x and w_0 need to be given in the same units"""
    return np.sqrt(2 / np.pi) * S_0 * w_0**4 / ((x**2 - w_0**2)**2 + (x**2) * (w_0**2)/Q**2)


@custom_model
def Lorentzian(x, S_0=1, w_0=1, Q=10):
    """Equation 11 from Foreman-Mackey+2017 (tested)
    x and w_0 need to be given in the same units
    """
    a = S_0
    c = w_0 / 2 / Q
    return np.sqrt(1 / 2 / np.pi) * a/c * (1 / (1 + ((x - w_0)/c)**2) + 1 / (1 + ((x + w_0)/c)**2))


@custom_model
def BendingPowerlaw(x, S_0=1, w_0=1, Q=1/2):
    """The PSD of a DampedRandomWalk. Astropy gives issues when using threads, so create it in each new thread"""
    a = S_0
    c = 0.5 * w_0/Q
    return np.sqrt(2 / np.pi) * a/c * (1 / (1 + (x/c)**2))

@custom_model
def Matern32(x, sigma=1, rho=1):
    """Always angular units"""
    #https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
    n = 1
    nu = 3 / 2
    return 1 / np.sqrt(2 * np.pi) * sigma**2 * 2**n * np.pi ** (n/2) * gamma(nu + n /2) * (2 *nu)**nu / (gamma(nu) * rho**(2*nu)) * (2 * nu / rho**2 + x**2) ** -(nu + n/2)
