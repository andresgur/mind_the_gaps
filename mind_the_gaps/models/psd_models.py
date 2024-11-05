from astropy.modeling.models import custom_model
import numpy as np
from scipy.special import gamma
from math import pi, sqrt
@custom_model
def SHO(x, S0=1, Q=10, omega0=1):
    """Equation 20 from Foreman-Mackey+2017 (tested)
    x and omega0 need to be given in the same units"""
    return sqrt(2 / pi) * S0 * omega0**4 / ((x**2 - omega0**2)**2 + (x**2) * (omega0**2)/Q**2)


@custom_model
def Lorentzian(x, S0=1, Q=10, omega0=1):
    """Equation 11 from Foreman-Mackey+2017 (tested)
    x and omega0 need to be given in the same units
    """
    a = S0
    c = omega0 / 2 / Q
    return sqrt(1 / 2 / pi) * a/c * (1 / (1 + ((x - omega0)/c)**2) + 1 / (1 + ((x + omega0)/c)**2))


@custom_model
def BendingPowerlaw(x, S0=1, omega0=1, Q=1/2):
    """The PSD of a DampedRandomWalk. Astropy gives issues when using threads, so create it in each new thread
    
    Parameters
    ----------
    x: array-like
        Angular frequencies
    
    """
    a = S0
    c = 0.5 * omega0/Q
    return sqrt(2 / pi) * a/c * (1 / (1 + (x/c)**2))

def Matern(x, sigma:float=1, rho:float=1, n:int=1, nu=3/2):
    """Generalised form of the Matern kernel
        https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
    
    Parameters
    ----------
    x: array-like
        Angular frequencies
    n:int,
    nu: float
    Returns the Matern-3/2 by default
    """
    return 1 / sqrt(2 * pi) * sigma**2 * 2**n * pi ** (n/2) * gamma(nu + n /2) * (2 *nu)**nu / (gamma(nu) * rho**(2*nu)) * (2 * nu / rho**2 + x**2) ** -(nu + n/2)


@custom_model
def Matern32(x, sigma=1, rho=1, n=1):
    """The PSD of the Matern-3/2
    """
    return 1 / sqrt(2 * pi) * sigma**2 * 4 / sqrt(3) * rho * (1 / (1 + (x * rho / sqrt(3))**2))**2

@custom_model
def Matern52(x, sigma=1, rho=1):
    """The PSD of the Matern-5/2
    """
    return 1 / sqrt(2 * pi) * sigma**2 * 16/3 / sqrt(5) * rho * (1 / (1 + (x * rho / sqrt(5))**2))**3

@custom_model
def Jitter(x, sigma:float=1):
    """A jitter (white noise) kernel"""
    # the 2 is so when integrating the positive frequencies we get the right variance
    # the sqrt(2pi) enters because of celerite
    # the N so when integrating this tends to sigma^2 as opposed to sigma^2 x N
    # the df enters because we need to "dilute" the power
    N = len(x)
    df = np.diff(x)[0] # this is angular frequencies
    normalization_factor =  2 / sqrt(2 * pi)
    return np.ones(N) * sigma**2 / normalization_factor / (df) / N
