import os
import sys

import celerite
import corner
import matplotlib.pyplot as plt
import numpy as np
from celerite.modeling import Model
from scipy.stats import percentileofscore

from mind_the_gaps import GPModelling
from mind_the_gaps.lightcurves import GappyLightcurve
from mind_the_gaps.models.celerite.celerite_models import Lorentzian as Lor
from mind_the_gaps.models.psd_models import (
    SHO,
    BendingPowerlaw,
    Jitter,
    Lorentzian,
    Matern32,
)
from mind_the_gaps.simulator import Simulator

if __name__ == "__main__":

    cpus = 12

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
    # null
    bounds_drw = dict(log_a=(-10, 50), log_c=(-10, 10))
    # you can use RealTerm from celerite or DampedRandomWalk from models.celerite_models
    null_kernel = celerite.terms.RealTerm(
        log_a=np.log(variance_drw), log_c=np.log(w_bend), bounds=bounds_drw
    )
    null_model = GPModelling(kernel=null_kernel, lightcurve=input_lc)
    null_model.derive_posteriors(max_steps=50000, fit=True, cores=cpus)
    autocorr = null_model.autocorr
    fig = plt.figure()
    n = np.arange(1, len(autocorr) + 1)
    plt.plot(n, autocorr, "-o")
    plt.ylabel("Mean $\\tau$")
    plt.xlabel("Number of steps")
    plt.savefig("autocorr_celerite_alt.png", dpi=100)
    corner_fig = corner.corner(
        null_model.mcmc_samples,
        labels=null_model.parameter_names,
        title_fmt=".1f",
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        truths=[np.log(variance_drw), np.log(w_bend), mean],
        title_kwargs={"fontsize": 18},
        max_n_ticks=3,
        labelpad=0.08,
        levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2**2), mean),
    )
    corner_fig.savefig("corner_plot_celerite.png", dpi=100)
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

    alternative_model = GPModelling(kernel=alternative_kernel, lightcurve=input_lc)

    alternative_model.derive_posteriors(max_steps=50000, fit=True, cores=cpus)
    # example of setting and getting params from model

    autocorr = alternative_model.autocorr
    fig = plt.figure()
    n = np.arange(1, len(autocorr) + 1)
    plt.plot(n, autocorr, "-o")
    plt.ylabel("Mean $\\tau$")
    plt.xlabel("Number of steps")
    plt.savefig("autocorr_celerite_alt.png", dpi=100)
    corner_fig = corner.corner(
        alternative_model.modelling_engine.mcmc_samples,
        labels=alternative_model.modelling_engine.gp.get_parameter_names(),
        title_fmt=".1f",
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 18},
        max_n_ticks=3,
        labelpad=0.08,
        levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2**2)),
    )  # plots 1 and 2 sigma levels
    corner_fig.savefig("corner_plot_celerite_alt.png", dpi=100)

    nsims = 100  # typically 10,000
    lcs = null_model.generate_from_posteriors(nsims=nsims)
    likelihoods_null = []
    likelihoods_alt = []

    for i, lc in enumerate(lcs):
        print("Processing lightcurve %d/%d" % (i + 1, len(lcs)), end="\r")
        # fig = plt.figure()
        # plt.errorbar(lc.times, lc.y, lc.dy)
        # plt.xlabel("Time (days)")
        # plt.ylabel("Rate (ct/s)")
        # plt.savefig("%d.png" % i, dpi=100)

        # Run a small MCMC to make sure we find the global maximum of the likelihood
        # ideally we'd probably want to run more samples
        null_modelling = GPModelling(kernel=null_kernel, lightcurve=lc)
        null_modelling.derive_posteriors(
            fit=False, cores=cpus, walkers=2 * cpus, max_steps=500, progress=False
        )
        likelihoods_null.append(null_modelling.max_loglikelihood)
        alternative_modelling = GPModelling(kernel=alternative_kernel, lightcurve=lc)
        alternative_modelling.derive_posteriors(
            fit=False, cores=cpus, walkers=2 * cpus, max_steps=500, progress=False
        )
        likelihoods_alt.append(alternative_modelling.max_loglikelihood)

    plt.figure()
    T_dist = -2 * (np.array(likelihoods_null) - np.array(likelihoods_alt))
    print(T_dist)
    plt.hist(T_dist, bins=10)
    T_obs = -2 * (null_model.max_loglikelihood - alternative_model.max_loglikelihood)
    print("Observed LRT_stat: %.3f" % T_obs)
    perc = percentileofscore(T_dist, T_obs)
    print("p-value: %.4f" % (1 - perc / 100))
    plt.axvline(T_obs, label="%.2f%%" % perc, ls="--", color="black")

    sigmas = [95, 99.7]
    colors = ["red", "green"]
    for i, sigma in enumerate(sigmas):
        plt.axvline(np.percentile(T_dist, sigma), ls="--", color=colors[i])
    plt.legend()
    # plt.axvline(np.percentile(T_dist, 99.97), color="green")
    plt.xlabel("$T_\\mathrm{LRT}$")

# plt.savefig("LRT_statistic.png", dpi=100)
