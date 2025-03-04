from abc import ABCMeta, abstractmethod
from typing import TypeVar

import jax.numpy as jnp
import numpy as np

ArrayType = TypeVar("Array", np.ndarray, jnp.ndarray)


class BaseGP(metaclass=ABCMeta):

    @abstractmethod
    def compute(self, times: ArrayType, errors: ArrayType):
        raise NotImplementedError

    @abstractmethod
    def get_parameter_vector(self):
        raise NotImplementedError

    @abstractmethod
    def set_parameter_vector(self, params: ArrayType):
        raise NotImplementedError

    @abstractmethod
    def log_likelihood(self, observations: ArrayType):
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
