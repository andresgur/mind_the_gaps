# @Author: Andrés Gúrpide <agurpide>
# @Date:   05-02-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-05-2024

import os
import sys

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, List, Mapping, Union

import celerite
import corner
import jax
import numpy as np
from celerite2.jax.terms import Term
from matplotlib import pyplot as plt
from scipy.stats import percentileofscore
from tinygp.kernels.base import Kernel

from mind_the_gaps.engines.celerite2_engine import Celerite2GPEngine
from mind_the_gaps.engines.celerite_engine import CeleriteGPEngine
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve
from mind_the_gaps.models.kernel_spec import KernelSpec


class GPModelling:
    """
    The interface for Gaussian Process (GP) modeling.
    """

    def __init__(
        self,
        # kernel: Union[celerite.modeling.Model, Kernel, Term],
        kernel_spec: KernelSpec,
        lightcurve: GappyLightcurve,
        meanmodel: str = None,
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
            kernel_spec=kernel_spec,
            lightcurve=lightcurve,
            model_type=model_type,
            meanmodel=meanmodel,
            fit_mean=fit_mean,
            **modelling_kwargs,
        )

    def _select_modelling_engine(
        self,
        kernel_spec: KernelSpec,
        lightcurve: GappyLightcurve,
        model_type: str = "auto",
        meanmodel: str = None,
        fit_mean: bool = True,
        mean_params: jax.Array = None,
        seed: int = None,
    ) -> Union[CeleriteGPEngine, Celerite2GPEngine]:
        """
        Factory method to select and instantiate the appropriate GP modelling engine.

        Parameters
        ----------
        kernel_spec : KernelSpec
            Specification of the GP kernel (e.g., celerite or celerite2).
        lightcurve : GappyLightcurve
            Lightcurve data to be modelled.
        model_type : str, optional
            Which model type to use: "auto", "celerite", "celerite2".
            Defaults to "auto", which selects the model based on kernel type.
        meanmodel : str, optional
            Mean model to use ("constant", "linear", etc.). Defaults to None.
        fit_mean : bool, optional
            Whether to fit the mean model parameters. Defaults to True.
        mean_params : jax.Array, optional
            Initial mean model parameters (used in celerite2 only).
        seed : int, optional
            Random seed (used in celerite2 only). Defaults to 0.
        **kwargs : dict
            Additional keyword arguments (currently unused).

        Returns
        -------
        Union[CeleriteGPEngine, Celerite2GPEngine]
            An instance of the appropriate GP modelling engine.

        Raises
        ------
        ValueError
            If model_type is invalid or kernel type is unrecognized.
        """
        if model_type.lower() == "auto":
            if issubclass(kernel_spec.terms[0].term_class, celerite.modeling.Model):
                model_type = "celerite"
            elif issubclass(kernel_spec.terms[0].term_class, Term):
                model_type = "celerite2"
            elif isinstance(kernel_spec.terms[0].term_class, Kernel):
                model_type = "tinygp"
            else:
                raise ValueError(
                    f"Unrecognized kernel type: {kernel_spec.terms[0].term_class}"
                )

        if model_type == "celerite":
            return CeleriteGPEngine(
                kernel_spec=kernel_spec,
                lightcurve=lightcurve,
                meanmodel=meanmodel,
                fit_mean=fit_mean,
            )
        elif model_type == "celerite2":
            return Celerite2GPEngine(
                kernel_spec=kernel_spec,
                lightcurve=lightcurve,
                meanmodel=meanmodel,
                mean_params=mean_params,
                seed=seed,
                fit_mean=fit_mean,
            )
        elif model_type == "tinygp":
            raise NotImplementedError(
                "TinyGP is not yet implemented in mind_the_gaps. Please use celerite or celerite2."
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def derive_posteriors(self, **engine_kwargs: Mapping[str, Any]):
        """Vaidate the engine kwargs and derive the posteriors."""
        self.modelling_engine.validate_posterior_kwargs(engine_kwargs)
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

    def plot_autocorrelation(self, path: str = "autocorr.png", dpi: int = 100) -> None:
        """Plots the autocorrelation statistic for the model.

        Parameters
        ----------
        path : str, optional
            File to save plot to, by default "autocorr.png"
        dpi : int, optional
            Define image quality in DPI, by default 100
        """

        path = os.path.abspath(path)

        plt.figure()
        n = np.arange(1, len(self.autocorr) + 1)
        plt.plot(n, self.autocorr, "-o")
        plt.ylabel("Mean $\\tau$")
        plt.xlabel("Number of steps")
        plt.savefig(fname=path, dpi=dpi)

    def corner_plot_samples(
        self, path: str = "corner_plot.png", dpi: int = 100
    ) -> None:
        """Plot a corner ploy of the samples for the model

        Parameters
        ----------
        path : str, optional
            File to save the plot to, by default "corner_plot.png"
        dpi : int, optional
            Define image quality in DPI, by default 100
        """
        samples = self.modelling_engine.mcmc_samples
        if isinstance(self.modelling_engine, Celerite2GPEngine):
            samples = {k: v for k, v in samples.items() if k != "log_likelihood"}
        corner_fig = corner.corner(
            samples,
            labels=self.modelling_engine.gp.get_parameter_names(),
            title_fmt=".1f",
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 18},
            max_n_ticks=3,
            labelpad=0.08,
            levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2**2)),
        )
        corner_fig.savefig(path, dpi=100)

    @property
    def parameters(self):
        return self.modelling_engine.gp.get_parameter_vector()

    @parameters.setter
    def parameters(self, value):
        self.modelling_engine.gp.set_parameter_vector(value)

    @property
    def loglikelihoods(self) -> Union[np.array, jnp.array]:
        return self.modelling_engine.loglikelihoods

    @property
    def autocorr(self) -> List[float]:
        return self.modelling_engine._autocorr

    @property
    def mcmc_samples(self):
        return self.modelling_engine.mcmc_samples

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

    @property
    def get_psd(self):
        return self.modelling_engine.gp.get_psd

    def standarized_residuals(self, include_noise: bool = True):
        """
        Returns the standarized residuals (see e.g. Kelly et al. 2011) Eq. 49.
        You should set the gp parameters to your best or mean (median) parameter values prior to calling this method

        Parameters
        ----------
        include_noise
            True to include any jitter term into the standard deviation calculation. False ignores this contribution.
        """
        return self.modelling_engine.gp.standarized_residuals()

    def predict(self, y, **kwargs):
        return self.modelling_engine.gp.predict(y, **kwargs)


class GPModellingComparison:
    """
    The interface for Gaussian Process (GP) modeling.
    """

    def __init__(
        self,
        null_kernel_spec: KernelSpec,
        alt_kernel_spec: KernelSpec,
        lightcurve: GappyLightcurve,
        null_mean_params: jax.Array = None,
        alt_mean_params: jax.Array = None,
        **modelling_kwargs: Mapping[str, Any],
    ):

        self.null_kernel_spec = null_kernel_spec
        self.alt_kernel_spec = alt_kernel_spec
        self.lightcurve = lightcurve

        self.likelihoods = []
        self.null_mean_params = null_mean_params
        self.alt_mean_params = alt_mean_params
        self.modelling_kwargs = modelling_kwargs

        self.null_model = GPModelling(
            kernel_spec=null_kernel_spec,
            lightcurve=lightcurve,
            mean_params=self.null_mean_params,
            **modelling_kwargs,
        )

        self.alt_model = GPModelling(
            kernel_spec=alt_kernel_spec,
            lightcurve=lightcurve,
            mean_params=self.alt_mean_params,
            **modelling_kwargs,
        )

    def derive_posteriors(self, **engine_kwargs: Mapping[str, Any]):
        # self.engine_kwargs = engine_kwargs
        self.null_model.derive_posteriors(**engine_kwargs)
        self.alt_model.derive_posteriors(**engine_kwargs)

    def generate_from_posteriors(self, nsims):
        return self.null_model.generate_from_posteriors(nsims)
        # self.alt_model.generate_from_posteriors(nsims)

    def process_lightcurves(self, nsims, **posterior_kwargs):
        self.likelihoods_null = []
        self.likelihoods_alt = []

        lcs = self.null_model.generate_from_posteriors(nsims=nsims)

        for i, lc in enumerate(lcs):
            print("Processing lightcurve %d/%d" % (i + 1, len(lcs)), end="\r")

            null_modelling = GPModelling(
                kernel_spec=self.null_kernel_spec,
                lightcurve=lc,
                mean_params=self.null_mean_params,
                **self.modelling_kwargs,
            )
            null_modelling.derive_posteriors(**posterior_kwargs)
            self.likelihoods_null.append(null_modelling.max_loglikelihood)

            alternative_modelling = GPModelling(
                kernel_spec=self.alt_kernel_spec,
                lightcurve=lc,
                mean_params=self.alt_mean_params,
                **self.modelling_kwargs,
            )
            alternative_modelling.derive_posteriors(**posterior_kwargs)
            self.likelihoods_alt.append(alternative_modelling.max_loglikelihood)

    def likelihood_ratio_test(self, path: str = "LRT.png") -> None:
        if self.likelihoods_null is None:
            raise ValueError(
                "You need to run process_lightcurves or process_lightcurves_par first."
            )
        path = os.path.abspath(path)
        plt.figure()
        T_dist = -2 * (np.array(self.likelihoods_null) - np.array(self.likelihoods_alt))
        plt.hist(T_dist, bins=10)
        T_obs = -2 * (
            self.null_model.max_loglikelihood - self.alt_model.max_loglikelihood
        )
        print("Observed LRT_stat: %.3f" % T_obs)
        perc = percentileofscore(T_dist, T_obs)
        print("p-value: %.4f" % (1 - perc / 100))
        plt.axvline(T_obs, label="%.2f%%" % perc, ls="--", color="black")

        sigmas = [95, 99.7]
        colors = ["red", "green"]
        for i, sigma in enumerate(sigmas):
            plt.axvline(np.percentile(T_dist, sigma), ls="--", color=colors[i])
        plt.legend()
        plt.xlabel("$T_\\mathrm{LRT}$")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Plot saved at: {path}")

    def process_lightcurves_par(self, nsims, par_workers=None, **posterior_kwargs):
        self.likelihoods_null = [None] * nsims
        self.likelihoods_alt = [None] * nsims
        lcs = self.null_model.generate_from_posteriors(nsims=nsims)

        args_list = [
            (
                i,
                lc,
                self.null_kernel_spec,
                self.alt_kernel_spec,
                self.modelling_kwargs,
                posterior_kwargs,
            )
            for i, lc in enumerate(lcs)
        ]

        with ProcessPoolExecutor(
            max_workers=par_workers  # or multiprocessing.cpu_count()
        ) as executor:
            futures = [
                executor.submit(_process_one_lightcurve, args) for args in args_list
            ]

            n_done = 0
        for future in as_completed(futures):
            i, null_ll, alt_ll = future.result()
            self.likelihoods_null[i] = null_ll
            self.likelihoods_alt[i] = alt_ll
            n_done += 1
            pct = (n_done / nsims) * 100
            print(f"Processed {n_done}/{nsims} lightcurves ({pct:.1f}%)", end="\r")
            sys.stdout.flush()

    def likelihood_ratio_test(self, path: str = "LRT.png") -> None:
        path = os.path.abspath(path)
        plt.figure()
        T_dist = -2 * (np.array(self.likelihoods_null) - np.array(self.likelihoods_alt))
        plt.hist(T_dist, bins=10)
        T_obs = -2 * (
            self.null_model.max_loglikelihood - self.alt_model.max_loglikelihood
        )
        print("Observed LRT_stat: %.3f" % T_obs)
        perc = percentileofscore(T_dist, T_obs)
        print("p-value: %.4f" % (1 - perc / 100))
        plt.axvline(T_obs, label="%.2f%%" % perc, ls="--", color="black")

        sigmas = [95, 99.7]
        colors = ["red", "green"]
        for i, sigma in enumerate(sigmas):
            plt.axvline(np.percentile(T_dist, sigma), ls="--", color=colors[i])
        plt.legend()
        plt.xlabel("$T_\\mathrm{LRT}$")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Plot saved at: {path}")


def _process_one_lightcurve(args):
    (i, lc, null_kernel_spec, alt_kernel_spec, modelling_kwargs, posterior_kwargs) = (
        args
    )
    null_modelling = GPModelling(
        kernel_spec=null_kernel_spec, lightcurve=lc, **modelling_kwargs
    )
    null_modelling.derive_posteriors(**posterior_kwargs)
    null_ll = null_modelling.max_loglikelihood

    alt_modelling = GPModelling(
        kernel_spec=alt_kernel_spec,
        lightcurve=lc,
        **modelling_kwargs,
    )
    alt_modelling.derive_posteriors(**posterior_kwargs)
    alt_ll = alt_modelling.max_loglikelihood

    return i, null_ll, alt_ll
