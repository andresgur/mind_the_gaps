from celerite.terms import Term
import numpy as np
from celerite.modeling import Model

# Covariance models
class Lorentzian(Term):
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
    
    def __repr__(self):
        return "Lorentzian({0.log_S0}, {0.log_Q}, {0.log_omega0})".format(self)

class Cosinus(Term):
    parameter_names = ("log_S0", "log_omega0")

    def get_complex_coefficients(self, params):
        log_S0, log_omega0 = params
        S0 = np.exp(log_S0)
        w0 = np.exp(log_omega0)
        return (
            # np.ones(2) * S0, #a
            S0,
            0,
            0,
            w0
            #np.zeros(2), #b
            #np.ones(2) * 0.5 * w0/Q, #c
            #[w0, -w0], #d
        )


class DampedRandomWalk(Term):
    """Equation 13 from Foreman+2017"""
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
    def __repr__(self):
        return "DampedRandomWalk({0.log_S0}, {0.log_omega0})".format(self)


class BendingPowerlaw(Term):
    r"""
    For log_Q << log_S0 this term follows a w^-2 whereas for log_Q ~ log_S0 follows w^-4
    """
    parameter_names = ("log_S0", "log_Q", "log_omega0")

    def get_complex_coefficients(self, params):

        log_S0, log_Q, log_omega0 = params
        w0 = np.exp(log_omega0)
        return (
            np.exp(log_S0), np.exp(log_Q),  w0, w0
        )

    def log_prior(self):
        # Constraint required for term to be positive definite. Can be relaxed
        # with multiple terms but must be treated carefully.
        if self.log_S0 < self.log_Q:
            return -np.inf
        return super(BendingPowerlaw, self).log_prior()
