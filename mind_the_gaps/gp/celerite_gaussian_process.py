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
    meanmodels = ["linear", "constant", "gaussian"]

    def __init__(
        self,
        kernel: terms.Term,
        lightcurve: GappyLightcurve,
        fit_mean: bool,
        mean_model: str = None,
    ):
        self._lightcurve = lightcurve
        self.mean_model, self.fit_mean = self._build_mean_model(meanmodel=mean_model)
        self.fit_mean = fit_mean
        self.gp = celerite.GP(kernel=kernel, mean=self.mean_model, fit_mean=fit_mean)
        self.kernel = kernel
        self.compute()

    def compute(self) -> None:

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

    def _build_mean_model(self, meanmodel: str) -> tuple[Model, bool]:
        """Construct the GP mean model based on lightcurve properties and
        input string

        Parameters
        ----------
        meanmodel : str
            Mean model to construct. Valid options are "constant","linear","Gaussian". Defaults to Gaussian if meanmodel is None.

        Returns
        -------
        Tuple[Model, bool]
            Returns celerite.modelling.Model and a bool indicating whether to the mean model is fitted or not.

        Raises
        ------
        ValueError
            If meanmodel is not an accepted option.
        """

        if meanmodel is None:
            # no fitting case
            meanmodel = ConstantModel(
                self._lightcurve.mean,
                bounds=[(np.min(self._lightcurve.y), np.max(self._lightcurve.y))],
            )
            return meanmodel, False

        elif meanmodel.lower() == "constant":
            meanlabels = ["$\mu$"]

        elif meanmodel.lower() == "linear":
            slope_guess = np.sign(self._lightcurve.y[-1] - self._lightcurve.y[0])
            minindex = np.argmin(self._lightcurve.times)
            maxindex = np.argmax(self._lightcurve.times)
            slope_bound = (
                self._lightcurve.y[maxindex] - self._lightcurve.y[minindex]
            ) / (self._lightcurve.times[maxindex] - self._lightcurve.times[minindex])
            if slope_guess > 0:
                min_slope = slope_bound
                max_slope = -slope_bound
            else:
                min_slope = -slope_bound
                max_slope = slope_bound
            slope = np.cov(self._lightcurve.times, self._lightcurve.y)[0, 1] / np.var(
                self._lightcurve.times
            )
            meanmodel = LinearModel(
                0, 1.5, bounds=[(-np.inf, np.inf), (-np.inf, np.inf)]
            )
            meanlabels = ["$m$", "$b$"]

        elif meanmodel.lower() == "gaussian":
            sigma_guess = (self._lightcurve.duration) / 2
            amplitude_guess = (
                (np.max(self._lightcurve.y) - np.min(self._lightcurve.y))
                * np.sqrt(2 * np.pi)
                * sigma_guess
            )

            mean_guess = self._lightcurve.times[len(self._lightcurve.times) // 2]
            meanmodel = GaussianModel(
                mean_guess,
                sigma_guess,
                amplitude_guess,
                bounds=[
                    (self._lightcurve.times[0], self._lightcurve.times[-1]),
                    (0, self._lightcurve.duration),
                    (
                        np.max(self._lightcurve.y)
                        * np.sqrt(2 * np.pi)
                        * self._lightcurve.duration,
                        50
                        * np.max(self._lightcurve.y)
                        * np.sqrt(2 * np.pi)
                        * self._lightcurve.duration,
                    ),
                ],
            )

            meanlabels = ["$\mu$", "$\sigma$", "$A$"]

        return meanmodel, True

    def get_parameter_names(self) -> tuple:
        return self.gp.get_parameter_names()
