import celerite
import numpy as np
from celerite import terms
from celerite.modeling import ConstantModel, Model

from mind_the_gaps.gp.gaussian_process import BaseGP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve
from mind_the_gaps.models.celerite.mean_models import (
    GaussianModel,
    LinearModel,
    SineModel,
)


class CeleriteGP(BaseGP):

    def __init__(
        self,
        kernel: terms.Term,
        lightcurve: GappyLightcurve,
        fit_mean: bool,
        mean_model: str = None,
    ):
        self.mean_model, self.fit_mean = self._build_mean_model(
            lightcurve=lightcurve, mean_model=mean_model
        )
        self._lightcurve = lightcurve
        self.gp = celerite.GP(
            kernel=kernel, mean=self.mean_model, fit_mean=self.fit_mean
        )
        self.kernel = kernel
        self.compute()

    def compute(self) -> None:  # , times: np.array, errors: np.array):

        self.gp.compute(self._lightcurve.times, self._lightcurve.dy + 1e-12)

    def get_parameter_vector(self) -> np.array:
        return self.gp.get_parameter_vector()

    def set_parameter_vector(self, params: np.array) -> None:
        self.gp.set_parameter_vector(vector=params)

    def log_likelihood(self, observations: np.array) -> float:
        return self.gp.log_likelihood(y=observations)

    def log_prior(self) -> float:
        return self.gp.log_prior()

    def get_parameter_bounds(self) -> list:
        return self.gp.get_parameter_bounds()

    def _build_mean_model(
        self, lightcurve: GappyLightcurve, mean_model: str
    ) -> tuple[Model, bool]:
        maxy = np.max(lightcurve.y)

        if mean_model is None:
            # no fitting case
            meanmodel = ConstantModel(
                lightcurve.mean, bounds=[(np.min(lightcurve.y), maxy)]
            )
            return meanmodel, False

        elif mean_model.lower() == "linear":

            meanmodel = LinearModel(
                0, 1.5, bounds=[(-np.inf, np.inf), (-np.inf, np.inf)]
            )
            return meanmodel, False  # ?
        elif mean_model.lower() == "gaussian":
            sigma_guess = (lightcurve.duration) / 2
            amplitude_guess = (maxy - np.min(y)) * np.sqrt(2 * np.pi) * sigma_guess
            mean_guess = lightcurve.times[len(lightcurve.times) // 2]
            meanmodel = GaussianModel(
                mean_guess,
                sigma_guess,
                amplitude_guess,
                bounds=[
                    (lightcurve.times[0], lightcurve.times[-1]),
                    (0, lightcurve.duration),
                    (
                        maxy * np.sqrt(2 * np.pi) * lightcurve.duration,
                        50 * maxy * np.sqrt(2 * np.pi) * lightcurve.duration,
                    ),
                ],
            )

            return meanmodel, True
        else:
            raise

    def get_parameter_names(self) -> tuple:
        return self.gp.get_parameter_names()
