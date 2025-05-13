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

from mind_the_gaps.gpmodelling import GPModelling

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

    times = np.arange(0, 1000)
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

    null_model = GPModelling(
        kernel=real_kernel_fn,
        lightcurve=input_lc,
        mean_model=None,
        fit_mean=True,
        max_steps=5000,
        num_chains=6,
        num_warmup=500,
        converge_step=1000,
        device="cpu",
        devices=10,
        nsims=10,
        params=jnp.array([variance_drw, w_bend]),
        bounds=bounds_drw,
    )

    null_model.derive_posteriors(fit=True)

    samples = null_model.modelling_engine.mcmc_samples

    corner_fig = corner.corner(
        samples,
        title_fmt=".1f",
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        truths=[np.log(variance_drw), np.log(w_bend)],
        title_kwargs={"fontsize": 18},
        max_n_ticks=3,
        labelpad=0.08,
        levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2**2)),
    )

    corner_fig.savefig("corner_plot_celerite2.png", dpi=100)
    Nsims = 100  # typically 10,000
    lcs = null_model.generate_from_posteriors()

    print("Done!")
    likelihoods_null = []
    likelihoods_alt = []

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

    bounds_qpo = dict(a=np.exp((-10, 50)), c=np.exp((-10, 10)), d=np.exp((-5, 5)))
    bounds_drw = dict(a2=np.exp((-10, 50)), c2=np.exp((-10, 10)))
    alternative_model = GPModelling(
        kernel=complex_real_kernel_fn,
        lightcurve=input_lc,
        mean_model=None,
        fit_mean=True,
        max_steps=5000,
        num_chains=6,
        num_warmup=500,
        converge_step=1000,
        device="cpu",
        devices=10,
        params=[variance_drw, c, w, variance_drw, w_bend],
        bounds=bounds_qpo | bounds_drw,
    )

    print("Deriving posteriors for alternative model")
    alternative_model.derive_posteriors(fit=True)
    samples = alternative_model.modelling_engine.mcmc_samples

    corner_fig = corner.corner(
        alternative_model.modelling_engine.mcmc_samples,
        # labels=alternative_model.modelling_engine.gp.get_parameter_names(),
        title_fmt=".1f",
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 18},
        max_n_ticks=3,
        labelpad=0.08,
        levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2**2)),
    )  # plots 1 and 2 sigma levels
    corner_fig.savefig("corner_plot_celerite2_alt.png", dpi=100)
