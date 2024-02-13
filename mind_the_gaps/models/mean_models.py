import numpy as np
from celerite.modeling import Model


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
