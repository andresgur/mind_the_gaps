# @Author: Andrés Gúrpide <agurpide>
# @Date:   05-02-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-05-2024

from jax import config

config.update("jax_enable_x64", True)

from typing import Any, List, Mapping, Union

import celerite
import numpy as np
from celerite2.jax.terms import Term
from tinygp.kernels.base import Kernel

from mind_the_gaps.engines.celerite_engine import CeleriteGPEngine
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve


class GPModelling:
    """
    The interface for Gaussian Process (GP) modeling.
    """

    meanmodels = ["linear", "constant", "gaussian"]

    def __init__(
        self,
        kernel: Union[celerite.modeling.Model, Kernel, Term],
        lightcurve: GappyLightcurve,
        mean_model: str = None,
        fit_mean: bool = True,
        model_type: str = "auto",
        **modelling_kwargs: Mapping[str, Any],
    ):
        """Initialise GPModelling instance

        Parameters
        ----------
        kernel : Union[celerite.modeling.Model, Kernel, Term]
            GP kernel/Model to be fitted.
        lightcurve : GappyLightcurve
            An instance of a lightcurve.
        mean_model : str, optional
            Mean model. If given it will be fitted, otherwise assumed the mean value.
            Available implementations are Constant, Linear and Gaussian., by default None
        fit_mean : bool, optional
            Whether to fit the mean, by default True
        model_type : str, optional
            User can select the model/kernel type, "auto", "celerite", "celerite2", "tinygp"
            by default "auto" in which case the GP engine is selected based on the type of kernel.
        """

        self._lightcurve = lightcurve
        self.modelling_engine = self._select_modelling_engine(
            kernel=kernel,
            lightcurve=lightcurve,
            model_type=model_type,
            mean_model=mean_model,
            fit_mean=fit_mean,
            **modelling_kwargs,
        )

    def _select_modelling_engine(
        self,
        kernel: Union[celerite.modeling.Model, Kernel, Term],
        lightcurve: GappyLightcurve,
        model_type: str,
        mean_model: str,
        fit_mean: bool,
        **modelling_kwargs: Mapping[str, Any],
    ) -> Union[CeleriteGPEngine]:
        """Select GP  modelling engine.

        Parameters
        ----------
        kernel : Union[celerite.modeling.Model, Kernel, Term]
            GP kernel/Model to be fitted.
        lightcurve : GappyLightcurve
            An instance of a lightcurve.
        mean_model : str, optional
            Mean model. If given it will be fitted, otherwise assumed the mean value.
            Available implementations are Constant, Linear and Gaussian., by default None
        fit_mean : bool, optional
            Whether to fit the mean, by default True
        model_type : str, optional
            User can select the model/kernel type, "auto", "celerite", "celerite2", "tinygp"
            by default "auto" in which case the GP engine is selected based on the type of kernel.

        Returns
        -------
        Union[CeleriteGPEngine]
            GP modelling engine.

        Raises
        ------
        ValueError
            If model_type is not valid, or if the type of kernel is unrecognised.

        """
        if model_type.lower() == "auto":
            if isinstance(kernel, celerite.modeling.Model):
                return CeleriteGPEngine(
                    kernel=kernel,
                    lightcurve=lightcurve,
                    mean_model=mean_model,
                    fit_mean=fit_mean,
                )
            elif callable(kernel) and getattr(kernel, "_return_type", None) is Term:
                raise NotImplementedError(f"Celerite2GPEngine not implemented.")
            elif isinstance(kernel, Kernel):
                raise NotImplementedError(f"TinyGPEngine not implemented.")
            else:
                raise ValueError(f"Unrecognised kernel type: {type(kernel)}")
        elif model_type.lower() == "celerite":
            return CeleriteGPEngine(
                kernel=kernel,
                lightcurve=lightcurve,
                mean_model=mean_model,
                fit_mean=fit_mean,
            )
        elif model_type.lower() == "celerite2":
            raise NotImplementedError(f"Celerite2GPEngine not implemented.")
        elif model_type.lower() == "tinygp":
            raise NotImplementedError(f"TinyGPEngine not implemented.")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def derive_posteriors(self, **engine_kwargs: Mapping[str, Any]):
        """Vaidate the engine kwargs and derive the posteriors."""
        self.modelling_engine.validate_kwargs(engine_kwargs)
        self.modelling_engine.derive_posteriors(**engine_kwargs)

    def generate_from_posteriors(
        self, **engine_kwargs: Mapping[str, Any]
    ) -> List[GappyLightcurve]:
        """Generates lightcurves by sampling from the MCMC posteriors.

        Returns
        -------
        List[GappyLightcurve]
            List containing lightcurves sampled from the MCMC posterirors.
        """
        return self.modelling_engine.generate_from_posteriors(**engine_kwargs)

    @property
    def loglikelihoods(self):
        return self.modelling_engine.loglikelihoods

    @property
    def autocorr(self):
        return self.modelling_engine.autocorr

    @property
    def max_loglikelihood(self):
        return self.modelling_engine.max_loglikelihood

    @property
    def max_parameters(self):
        if self.modelling_engine._mcmc_samples is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run derive_posteriors prior to populate the attributes."
            )
        return self._mcmc_samples[np.argmax(self._loglikelihoods)]

    @property
    def median_parameters(self):
        return self.modelling_engine.median_parameters

    @property
    def parameter_names(self):
        return self.modelling_engine.gp.get_parameter_names()

    @property
    def k(self):
        return self.modelling_engine.k

    @property
    def tau(self):
        return self.modelling_engine.tau

    @property
    def mcmc_samples(self):
        if self.modelling_engine._mcmc_samples is None:
            raise AttributeError(
                "Posteriors have not been derived. Please run derive_posteriors prior to populate the attributes."
            )
        return self.modelling_engine._mcmc_samples
