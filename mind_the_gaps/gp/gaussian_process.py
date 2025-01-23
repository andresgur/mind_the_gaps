from abc import ABCMeta, abstractmethod

import numpy as np


class BaseGP(metaclass=ABCMeta):

    @abstractmethod
    def compute(self, times: np.array, errors: np.array):
        raise NotImplementedError

    @abstractmethod
    def get_parameter_vector(self):
        raise NotImplementedError

    @abstractmethod
    def set_parameter_vector(self, params: np.array):
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(self, observations: np.array):
        raise NotImplementedError

    @abstractmethod
    def get_parameter_bounds(self):
        raise NotImplementedError

    @abstractmethod
    def log_prior(self):
        raise NotImplementedError

    @abstractmethod
    def get_parameter_names(self):
        raise NotImplementedError
