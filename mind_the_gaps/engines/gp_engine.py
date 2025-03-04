from abc import ABCMeta, abstractmethod
from typing import Any, List, Mapping, TypeVar, Union

import jax.numpy as jnp
import numpy as np

from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve

ArrayType = TypeVar("Array", np.ndarray, jnp.ndarray)


class BaseGPEngine(metaclass=ABCMeta):
    """Base Gaussian Process Engine class.

    Parameters
    ----------
    metaclass : type, optional
        metaclass for the class, by default ABCMeta, an abstract base class.

    Raises
    ------
    NotImplementedError
        If a subclass doesnt define posterior_params.
    ValueError
        If params used in posterior_params is not one allowed for the engine.
    """

    posterior_params = {}
    meanmodels = ["linear", "constant", "gaussian"]
    _ndim = None
    _loglikelihoods = None
    _mcmc_samples = None

    def __init_subclass__(cls, **kwargs):

        super().__init_subclass__(**kwargs)

        if cls.posterior_params is None:
            raise NotImplementedError(
                f"Subclasses of {cls.__name__} must define `posterior_params`."
            )

    @classmethod
    def validate_kwargs(cls, kwargs):
        for key in kwargs:
            if key not in cls.posterior_params:
                raise ValueError(
                    f"Invalid parameter '{key}' for {cls.__name__}. "
                    f"Allowed parameters: {cls.posterior_params}"
                )

    @property
    def k(self) -> int:
        """
        Number of variable parameters

        Returns
        -------
        int
            Number of variable parameters
        """
        if self._ndim is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define `_ndim` in its `__init__`."
            )
        return self._ndim

    @property
    def loglikelihoods(self) -> ArrayType:
        """_summary_

        Returns
        -------
        Union[np.array|jnp.array]
            Array containing likelihoods.

        Raises
        ------
        AttributeError
            If the posteriors have not been derived the loglikelihoods will not exist
        """
        if self._loglikelihoods is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run "
                "derive_posteriors prior to populate the attributes."
            )
        return self._loglikelihoods

    @property
    def mcmc_samples(self) -> ArrayType:
        """_summary_

        Returns
        -------
        Union[np.array|jnp.array]
            Array containing likelihoods.

        Raises
        ------
        AttributeError
            If the posteriors have not been derived the loglikelihoods will not exist
        """
        if self._mcmc_samples is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run "
                "derive_posteriors prior to populate the attributes."
            )
        return self._mcmc_samples

    @abstractmethod
    def derive_posteriors(
        self, **engine_kwargs: Mapping[str, Any]
    ) -> List[GappyLightcurve]:
        raise NotImplementedError

    @abstractmethod
    def generate_from_posteriors(
        self, **engine_kwargs: Mapping[str, Any]
    ) -> List[GappyLightcurve]:
        raise NotImplementedError

    # @property
    # @abstractmethod
    # def parameters(self) -> ArrayType:
    #    raise NotImplementedError

    # @parameters.setter
    # @abstractmethod
    # def parameters(self, value: ArrayType) -> None:
    #    raise NotImplementedError

    @property
    @abstractmethod
    def autocorr(self) -> List[float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_loglikelihood(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def max_parameters(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def median_parameters(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def parameter_names(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def tau(self):
        raise NotImplementedError
