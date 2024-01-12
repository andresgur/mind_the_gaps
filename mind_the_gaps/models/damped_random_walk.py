"""
Model component for a damped random walk

Describes a damped random walk in both Astropy and Celerite formats
"""
import numpy as np
from astropy.modeling import Model as AstropyModel
from astropy.modeling.models import custom_model
from astropy.units import day, second
from celerite.terms import Term as CeleriteTerm
from mind_the_gaps.models import ModelComponent
from typing import Tuple


class DampedRandomWalk(CeleriteTerm):
    """
    Equation 13 from Foreman+2017
    """
    parameter_names = ("log_S0", "log_omega0")
    def get_real_coefficients(self, params):
        log_S0, log_omega0 = params
        Q = 1 / 2
        S0 = np.exp(log_S0)
        w0 = np.exp(log_omega0)
        return (
            S0,  # S0 * np.ones(2),
            0.5 * w0/Q # 0.5*w0/Q * np.ones(2)
        )

class DampedRandomWalk(ModelComponent):
    """
    Damped random walk lightcurve component

    Parameters
    ----------
    S_0: Tuple[float, float, float]
        Minimum, starting, and maximum S_0
    period: Tuple[float, float, float]
        Minimum, starting, and maximum period in days
    """
    def __init__(
            self, 
            S_0: Tuple[float, float, float], 
            period: Tuple[float, float, float],
        ):

        # Convert period to angular frequency in seconds
        w_0 = 2. * np.pi / (np.array(period, dtype=float) * day.to(second))

        self._set_values_and_ranges(
            ["S_0", "w_0"], [S_0, w_0]
        )

        log_S_0 = np.log(np.array(S_0, dtype=float))
        log_w_0 = np.log(w_0)

        self._celerite_component = DampedRandomWalk(
            log_S0=log_S_0[1], 
            log_omega0=log_w_0[1], 
            bounds={
                "log_S0": [log_S_0[0], log_S_0[2]],
                "log_omega0": [log_w_0[0], log_w_0[2]],
            }
        )

        @custom_model
        def AstropyDampedRandomWalk(x, S_0=S_0[1], w_0=w_0[1], Q=1/2):
            """
            The PSD of a DampedRandomWalk. 
            Astropy gives issues when using threads, so create it in each new thread
            """
            a = S_0
            c = 0.5 * w_0/Q
            return np.sqrt(2 / np.pi) * a/c * (1 / (1 + (x/c)**2))
        
        self._astropy_component = AstropyDampedRandomWalk()
