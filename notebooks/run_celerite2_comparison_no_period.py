import sys

sys.path.append("/Users/connorourke/bin/src/mind_the_gaps")
from dataclasses import fields

import arviz as az
import celerite2
import corner
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from celerite2.jax.terms import RealTerm, SHOTerm
from scipy.stats import percentileofscore

from mind_the_gaps.gpmodelling import GPModelling, GPModellingComparison

# from mind_the_gaps.gp.processes.celerite2_gaussian_process import Celerite2GP
from mind_the_gaps.lightcurves import GappyLightcurve
from mind_the_gaps.models.celerite2.kernel_terms import (
    complex_real_kernel_fn,
    real_kernel_fn,
)
from mind_the_gaps.models.psd_models import BendingPowerlaw
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

    bounds_drw = dict(a=np.exp((-10, 50)), c=np.exp((-10, 10)))
    P = 10  # period of the QPO
    w = 2 * np.pi / P
    # Define starting parameters
    log_variance_qpo = np.log(variance_drw)
    Q = 80  # coherence
    c = 0.5 * w / Q
    log_c = np.log(c)
    log_d = np.log(w)
    print(
        f"log variance of the QPO: {log_variance_qpo:.2f}, log_c: {log_c:.2f}, log omega: {log_d:.2f}"
    )

    bounds_qpo = dict(
        a=np.exp((-10, 50)),
        c=np.exp((-10, 10)),
        d=np.exp((-5, 5)),
    )
    bounds_drw = dict(a2=np.exp((-10, 50)), c2=np.exp((-10, 10)))
    bounds_real = dict(a=np.exp((-10, 50)), c=np.exp((-10, 10)))

    comparison_kwargs = {
        "null_kwargs": {
            "params": jnp.array([input_lc.mean, variance_drw, w_bend]),
            "bounds": bounds_real,
            "fit_mean": True,
            "cpus": 10,
        },
        "alt_kwargs": {
            "fit_mean": True,
            "cpus": 10,
            "params": jnp.array(
                [input_lc.mean, variance_drw, c, w, variance_drw, w_bend]
            ),
            "bounds": bounds_qpo | bounds_drw,
        },
    }
    gpmodel = GPModellingComparison(
        null_kernel=real_kernel_fn,
        alt_kernel=complex_real_kernel_fn,
        lightcurve=input_lc,
        **comparison_kwargs,
    )

    gpmodel.derive_posteriors(
        fit=True, max_steps=5000, num_chains=10, num_warmup=500, converge_steps=1000
    )
    gpmodel.process_lightcurves(
        nsims=100,
        fit=True,
        max_steps=500,
        num_chains=6,
        num_warmup=500,
        converge_steps=500,
    )
