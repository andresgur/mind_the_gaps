from abc import ABCMeta


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
