"""
Model component for a Lorentzian

Describes a Lorentzian in both Astropy and Celerite formats
"""
import numpy as np
from astropy.modeling import Model as AstropyModel
from astropy.modeling.models import custom_model
from astropy.units import day, second
from celerite.terms import Term as CeleriteTerm
from mind_the_gaps.models import ModelComponent
from typing import Tuple


class CeleriteLorentzian(CeleriteTerm):
    """
    Celerite-format Lorentzian model component
    """
    parameter_names = ("log_S0", "log_Q", "log_omega0")
    def get_real_coefficients(self, params):
        log_S0, log_Q, log_omega0 = params
        #f = np.sqrt(1.0 - 4.0 * Q**2)
        return (
            0,  # S0 * np.ones(2),
            0 # 0.5*w0/Q * np.ones(2)
        )

    def get_complex_coefficients(self, params):
        log_S0, log_Q, log_omega0 = params
        Q = np.exp(log_Q)
        S0 = np.exp(log_S0)
        w0 = np.exp(log_omega0)
        return (
            # np.ones(2) * S0, #a
            S0,
            0,
            0.5 * w0/Q,
            w0
            #np.zeros(2), #b
            #np.ones(2) * 0.5 * w0/Q, #c
            #[w0, -w0], #d
        )


class Lorentzian(ModelComponent):
    """
    Lorentzian lightcurve component

    Parameters
    ----------
    S_0: Tuple[float, float, float]
        Minimum, starting, and maximum S_0
    period: Tuple[float, float, float]
        Minimum, starting, and maximum period in days
    Q: Tuple[float, float, float]
        Minimum, starting, and maximum Q
    """
    def __init__(
            self, 
            S_0: Tuple[float, float, float], 
            period: Tuple[float, float, float], 
            Q: Tuple[float, float, float]
        ):

        # Convert period to angular frequency in seconds
        w_0 = 2. * np.pi / (np.array(period, dtype=float) * day.to(second))

        self._set_values_and_ranges(
            ["S_0", "w_0", "Q"], [S_0, w_0, Q]
        )

        log_S_0 = np.log(np.array(S_0, dtype=float))
        log_w_0 = np.log(w_0)
        log_Q = np.log(np.array(Q, dtype=float))

        self._celerite_component = CeleriteLorentzian(
            log_S0=log_S_0[1], 
            log_omega0=log_w_0[1],
            log_Q=log_Q[1],
            bounds={
                "log_S0": [log_S_0[0], log_S_0[2]],
                "log_omega0": [log_w_0[0], log_w_0[2]],
                "log_Q": [log_Q[0], log_Q[2]]
            }
        )

        @custom_model
        def AstropyLorentzian(
            x, 
            S_0=S_0[1], 
            w_0=w_0[1],
            Q=Q[1]
        ):
            """
            Astropy-format Lorentzian model component

            Equation 11 from Foreman-Mackey+2017 (tested)
            x and w_0 need to be given in the same units
            """
            a = S_0
            c = w_0 / 2 / Q
            return np.sqrt(1 / 2 / np.pi) * a/c * (1 / (1 + ((x - w_0)/c)**2) + 1 / (1 + ((x + w_0)/c)**2))

        self._astropy_component = AstropyLorentzian()
