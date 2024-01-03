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


# Mean models
class GaussianModel(Model):
    parameter_names = ("mean", "sigma", "amplitude", "constant")

    def get_value(self, x):
        return self.amplitude / (2 * np.pi * self.sigma) * np.exp(-(x - self.mean)**2 /(2 * self.sigma**2)) + self.constant

class SineModel(Model):
    parameter_names = ("constant", "amplitude", "frequency", "phase")

    def get_value(self, x):
        return self.constant + self.amplitude * np.sin(self.frequency * x + self.phase)

class TwoSineModel(Model):
    parameter_names = ("constant", "amplitude0", "phase0", "amplitude1", "phase1", "frequency")

    def get_value(self, x):
        return self.constant + self.amplitude0 * np.sin(self.frequency * x + self.phase0) + self.amplitude1 * np.sin(2 * self.frequency * x + self.phase1)

class LinearModel(Model):
    parameter_names = ("slope", "intercept")
    def get_value(self,x):
        return self.slope * x + self.intercept

    def compute_gradient(self, x):
        # derive with respect to slope, and derive with respect to intercept
        return np.array([np.ones_like(x) * x, np.ones_like(x)])


class LensingProfile(Model):
    parameter_names = ("lense_mass", "stellar_mass", "")
    def get_value(self, x):
        # TODO: add calculation here
        return np.nan
