# @Author: Andrés Gúrpide <agurpide>
# @Date:   05-02-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-05-2024

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from typing import Any, List, Mapping, Union

import celerite
import numpy as np
from celerite2.jax.terms import Term
from tinygp.kernels.base import Kernel

from mind_the_gaps.engines.celerite2_engine import Celerite2GPEngine
from mind_the_gaps.engines.celerite_engine import CeleriteGPEngine
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve


class GPModelling:
    """
    The interface for Gaussian Process (GP) modeling.
    """

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
    ) -> Union[CeleriteGPEngine | Celerite2GPEngine]:
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
                return Celerite2GPEngine(
                    kernel_fn=kernel,
                    lightcurve=lightcurve,
                    **modelling_kwargs,
                )
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
            return Celerite2GPEngine(
                kernel_fn=kernel,
                lightcurve=lightcurve,
                **modelling_kwargs,
            )
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
    def parameters(self):
        return self.modelling_engine.gp.get_parameter_vector()

    @parameters.setter
    def parameters(self, value):
        self.modelling_engine.gp.set_parameter_vector(value)

    @property
    def loglikelihoods(self) -> Union[np.array, jnp.array]:
        return self.modelling_engine._loglikelihoods

    @property
    def autocorr(self) -> List[float]:
        return self.modelling_engine._autocorr

    @property
    def mcmc_samples(self):
        return self.modelling_engine._mcmc_samples

    @property
    def max_loglikelihood(self):
        return self.modelling_engine.max_loglikelihood

    @property
    def max_parameters(self):
        return self.modelling_engine.max_parameters

    @property
    def median_parameters(self):
        return self.modelling_engine.median_parameters

    @property
    def parameter_names(self):
        return self.modelling_engine.parameter_names

    @property
    def k(self):
        return self.modelling_engine.k

    @property
    def tau(self):
        return self.modelling_engine.tau

    @property
    def bounds(self):
        return self.modelling_engine.gp.get_parameter_bounds()


class GPModellingComparison:
    """
    The interface for Gaussian Process (GP) modeling.
    """

    def __init__(
        self,
        null_kernel: Union[celerite.modeling.Model, Kernel, Term],
        alt_kernel: Union[celerite.modeling.Model, Kernel, Term],
        lightcurve: GappyLightcurve,
        **modelling_kwargs: Mapping[str, Any],
    ):

        self.null_kernel = null_kernel
        self.alt_kernel = alt_kernel
        self.lightcurve = lightcurve

        self.likelihoods = []
        self.null_modelling_kwargs = modelling_kwargs.pop("null_kwargs", {})
        self.alt_modelling_kwargs = modelling_kwargs.pop("alt_kwargs", {})

        self.null_model = GPModelling(
            kernel=null_kernel,
            lightcurve=lightcurve,
            **self.null_modelling_kwargs,
        )

        self.alt_model = GPModelling(
            kernel=alt_kernel, lightcurve=lightcurve, **self.alt_modelling_kwargs
        )

    def derive_posteriors(self, **engine_kwargs: Mapping[str, Any]):
        self.engine_kwargs = engine_kwargs
        self.null_model.derive_posteriors(**engine_kwargs)
        self.alt_model.derive_posteriors(**engine_kwargs)

    def generate_from_posteriors(self, nsims):
        return self.null_model.generate_from_posteriors(nsims)
        # self.alt_model.generate_from_posteriors(nsims)

    def process_lightcurves(self, nsims, **engine_kwargs):
        likelihoods_null = []
        likelihoods_alt = []

        lcs = self.null_model.generate_from_posteriors(nsims=nsims)

        for i, lc in enumerate(lcs):
            print("Processing lightcurve %d/%d" % (i + 1, len(lcs)), end="\r")

            # Run a small MCMC to make sure we find the global maximum of the likelihood
            # ideally we'd probably want to run more samples
            # null_modelling = GPModelling(kernel=null_kernel,lightcurve=lc)

            null_modelling = GPModelling(
                kernel=self.null_kernel, lightcurve=lc, **self.null_modelling_kwargs
            )
            null_modelling.derive_posteriors(**engine_kwargs)
            likelihoods_null.append(null_modelling.max_loglikelihood)

            alternative_modelling = GPModelling(
                kernel=self.alt_kernel, lightcurve=lc, **self.alt_modelling_kwargs
            )
            alternative_modelling.derive_posteriors(**engine_kwargs)
            likelihoods_alt.append(alternative_modelling.max_loglikelihood)
