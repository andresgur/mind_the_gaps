import numpy as np
from abc import ABC
from astropy.modeling.models import Model as AstropyModel
from celerite.terms import Term as CeleriteTerm
# from mind_the_gaps.lightcurves import Lightcurve
# from mind_the_gaps.model_fit import ModelFit

from typing import List, Optional, Union, Dict, Tuple
from numpy.typing import NDArray


class ModelComponent(ABC):
    """
    Component that describes a model of a lightcurve

    Abstract class that is subclassed by specific model components
    """
    _astropy_component: AstropyModel
    _celerite_component: CeleriteTerm
    _parameter_values: Dict[str, float]
    _parameter_ranges: Dict[str, Tuple[float, float]]

    def get_celerite_kernel(self) -> CeleriteTerm:
        """
        Gets the Celerite representation of this component for the fitting process
        """
        return self._celerite_component

    def get_astropy_model(self) -> AstropyModel:
        """
        Gets the Astropy representation of this component for simulation
        """
        return self._astropy_component
    
    def _set_values_and_ranges(
            self, 
            parameters: List[str], 
            inputs: List[Tuple[float,float,float]]
    ):
        """
        Convenience method for sorting input lists to values and ranges
        for local storage

        Parameters
        ----------
        parameters: List[str]
            List of parameter names to store
        inputs: List[Tuple[float, float, float]]
            List of 3-long tuples of floats as minimum, starting, maximum
        """
        for parameter, input in zip(parameters, inputs):
            self._parameter_values[parameter] = input[1]
            self._parameter_ranges[parameter] = [input[0], input[2]]


class Model:
    """
    Lightcurve model

    Multi-component model that describes a lightcurve, 
    e.g. a combination of a Lorentzian and Damped Random Walk.

    Parameters
    ----------
    components: Union[ModelComponent, List[ModelComponents]]
        The components of the combined model.
        Can be either a single component, or a list.

    Raises
    ------
    TypeError: If the user attempts to declare a model of non-MTG-models.
    """
    _model_components: List[ModelComponent]

    def __init__(self, components: Union[ModelComponent, List[ModelComponent]]):
        if not isinstance(components, List):
            components = [components]

        for component in components:
            if not isinstance(component, ModelComponent):
                raise TypeError(
                    "A model must be created using Mind The Gaps versions of its components"
                )

        self._model_components = components

    def get_celerite_kernel(self) -> NDArray:
        """
        Returns the combined Celerite representation of the model components
        """
        return np.sum(
            component.get_celerite_kernel() for component in self._model_components
        )

    def get_astropy_model(self) -> List[AstropyModel]:
        """
        Returns the combined Astropy representation of the model components
        """
        return np.sum(
            component.get_astropy_model() for component in self._model_components
        )

    # def fit_lightcurve(
    #         lightcurve: Lightcurve, 
    #         limit: Optional[int] = 1000
    # ) -> ModelFit:
    #     """
    #     Fit the model to a lightcurve using Celerite
    #
    #     Parameters
    #     ----------
    #     lightcurve: Lightcurve
    #         The lightcurve to fit
    #     limit: Optional[int]
    #         The number of MCMC cycles to perform whilst fitting, 
    #         if None then continue until convergence
    #        
    #     Returns
    #     -------
    #     ModelFit: A fit of the lightcurve to this model
    #     """
    #     kernel = self.get_celerite_kernel()
    #     etc,
