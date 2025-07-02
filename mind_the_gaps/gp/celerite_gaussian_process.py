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
from mind_the_gaps.models.kernel_spec import KernelSpec


class CeleriteGP(BaseGP):
    meanmodels = ["linear", "constant", "gaussian"]

    def __init__(
        self,
        kernel_spec: KernelSpec,
        lightcurve: GappyLightcurve,
        fit_mean: bool,
        meanmodel: str = None,
    ):
        self._lightcurve = lightcurve
        self.kernel_spec = kernel_spec
        self.kernel = self._get_kernel()
        self.mean_model, self.fit_mean = self._build_mean_model(meanmodel=meanmodel)

        self.fit_mean = fit_mean
        self.gp = celerite.GP(
            kernel=self.kernel, mean=self.mean_model, fit_mean=fit_mean
        )

        self.compute()

    def compute(self) -> None:

        self.gp.compute(
            self._lightcurve.times,
            self._lightcurve.dy + 1e-12,
        )

    def compute_fit():
        pass

    def compute_sample():
        pass

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
            meanmodel = ConstantModel(
                self._lightcurve.mean,
                bounds=[(np.min(self._lightcurve.y), np.max(self._lightcurve.y))],
            )
            return meanmodel, True

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

    def _get_kernel(self):

        terms = []
        bounds_dict = {}
        for i, term_spec in enumerate(self.kernel_spec.terms):
            kwargs = {}
            for name, param_spec in term_spec.parameters.items():
                kwargs[name] = param_spec.value
                if param_spec.bounds is not None:
                    bounds_dict[name] = param_spec.bounds

            if bounds_dict:
                kwargs["bounds"] = bounds_dict

            term = term_spec.term_class(**kwargs)
            terms.append(term)

        kernel = terms[0]
        for term in terms[1:]:
            kernel += term

        return kernel

    def standarized_residuals(self, include_noise=True):
        """Returns the standarized residuals (see e.g. Kelly et al. 2011) Eq. 49.
        You should set the gp parameters to your best or mean (median) parameter values prior to calling this method

        Parameters
        ----------
        include_noise: bool,
            True to include any jitter term into the standard deviation calculation. False ignores this contribution.
        """
        pred_mean, pred_var = self.gp.predict(
            self._lightcurve.y, return_var=True, return_cov=False
        )
        if include_noise:
            pred_var += self.gp.kernel.jitter
        std_res = (self._lightcurve.y - pred_mean) / np.sqrt(pred_var)
        return std_res

    def predict(self, y, **kwargs) -> tuple:
        """Compute the conditional predictive distribution of the model by calling celerite's predict method.

        Parameters
        ----------
        y : np.ndarray
            Observations at the coordinates of the lightcurve times.
        **kwargs : dict
            Additional keyword arguments to pass to the celerite predict method.
        Returns
        ------

        tuple
            mu, (mu, cov), or (mu, var) depending on the values of return_cov and
            return_var. See https://celerite.readthedocs.io/en/stable/python/gp/#celerite.GP.predict.

        """
        return self.gp.predict(y, **kwargs)
