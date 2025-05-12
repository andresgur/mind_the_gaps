import sys

sys.path.append("/Users/connorourke/bin/src/mind_the_gaps")
import os
import sys
from dataclasses import fields

import arviz as az
import celerite
import celerite2
import corner
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from celerite2.jax.terms import RealTerm, SHOTerm
from celerite.modeling import Model
from scipy.stats import percentileofscore

from mind_the_gaps import GPModelling
from mind_the_gaps.gpmodelling import GPModelling, GPModellingComparison

# from mind_the_gaps.gp.processes.celerite2_gaussian_process import Celerite2GP
from mind_the_gaps.lightcurves import GappyLightcurve
from mind_the_gaps.models.celerite2.kernel_terms import (
    complex_real_kernel_fn,
    real_kernel_fn,
)
from mind_the_gaps.models.celerite.celerite_models import Lorentzian as Lor
from mind_the_gaps.models.kernel import KernelParameterSpec, KernelSpec, KernelTermSpec
from mind_the_gaps.models.psd_models import (
    SHO,
    BendingPowerlaw,
    Jitter,
    Lorentzian,
    Matern32,
)
from mind_the_gaps.simulator import Simulator

numpyro.set_host_device_count(10)
if __name__ == "__main__":
    cpus = 10

    times = np.arange(0, 500)
    dt = np.diff(times)[0]
    mean = 100

    variance_drw = (mean * 0.1) ** 2  # variance of the DRW (bending powerlaw)
    w_bend = 2 * np.pi / 20  # angular frequency of the DRW or Bending Powerlaw

    # define the PSD model
    psd_model = BendingPowerlaw(variance_drw, w_bend)

    # create simulator object
    simulator = Simulator(
        psd_model,
        times,
        np.ones(len(times)) * dt,
        mean,
        pdf="Gaussian",
        extension_factor=2,
    )
    # simulate noiseless count rates from the PSD, make the initial lightcurve 2 times as long as the original times
    countrates = simulator.generate_lightcurve()
    # add (Poisson) noise
    noisy_countrates, dy = simulator.add_noise(countrates)

    input_lc = GappyLightcurve(times, noisy_countrates, dy, exposures=dt)

    bounds_drw = dict(log_a=(-10, 50), log_c=(-10, 10))
    # you can use RealTerm from celerite or DampedRandomWalk from models.celerite_models
    null_kernel = celerite.terms.RealTerm(
        log_a=np.log(variance_drw), log_c=np.log(w_bend), bounds=bounds_drw
    )

    bounds_drw = dict(a=np.exp((-10, 50)), c=np.exp((-10, 10)))

    P = 10  # period of the QPO
    w = 2 * np.pi / P
    # Define starting parameters
    log_variance_qpo = np.log(variance_drw)
    Q = 80  # coherence
    log_c = np.log(0.5 * w / Q)
    log_d = np.log(w)
    print(
        f"log variance of the QPO: {log_variance_qpo:.2f}, log_c: {log_c:.2f}, log omega: {log_d:.2f}"
    )

    bounds_qpo = dict(log_a=(-10, 50), log_c=(-10, 10), log_d=(-5, 5))
    alternative_kernel = celerite.terms.ComplexTerm(
        log_a=log_variance_qpo, log_c=log_c, log_d=log_d, bounds=bounds_qpo
    ) + celerite.terms.RealTerm(
        log_a=np.log(variance_drw), log_c=np.log(w_bend), bounds=bounds_drw
    )

    comparison_kwargs = {
        "null_kwargs": {
            "fit_mean": True,
            "cpus": 10,
        },
        "alt_kwargs": {
            "fit_mean": True,
            "cpus": 10,
        },
    }
    null_spec = KernelSpec(
        engine="celerite",
        terms=[
            KernelTermSpec(
                term_class=celerite.terms.RealTerm,
                parameters={
                    "log_a": KernelParameterSpec(
                        value=np.log(variance_drw), bounds=(-10, 50)
                    ),
                    "log_c": KernelParameterSpec(
                        value=np.log(w_bend), bounds=(-10, 10)
                    ),
                },
            )
        ],
    )

    alternative_spec = KernelSpec(
        engine="celerite",
        terms=[
            KernelTermSpec(
                term_class=celerite.terms.ComplexTerm,
                parameters={
                    "log_a": KernelParameterSpec(
                        value=log_variance_qpo, bounds=(-10, 50)
                    ),
                    "log_c": KernelParameterSpec(value=log_c, bounds=(-10, 10)),
                    "log_d": KernelParameterSpec(value=log_d, bounds=(-5, 5)),
                },
            ),
            KernelTermSpec(
                term_class=celerite.terms.RealTerm,
                parameters={
                    "log_a": KernelParameterSpec(
                        value=np.log(variance_drw), bounds=(-10, 50)
                    ),
                    "log_c": KernelParameterSpec(
                        value=np.log(w_bend), bounds=(-10, 10)
                    ),
                },
            ),
        ],
    )
    gpmodel = GPModellingComparison(
        null_kernel_spec=null_spec,
        alt_kernel_spec=alternative_spec,
        lightcurve=input_lc,
        **comparison_kwargs,
    )

    gpmodel.derive_posteriors(fit=True, max_steps=50000, cores=cpus)
    gpmodel.process_lightcurves(nsims=100, fit=True, max_steps=1000, cores=cpus)
