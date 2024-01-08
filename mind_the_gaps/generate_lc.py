#! /usr/bin/env python
# @Author: Andrés Gúrpide <agurpide>
# @Date:   01-09-2022
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 12-05-2022

import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import astropy.units as u
from astropy.visualization import quantity_support
import time
from multiprocessing import Pool
from functools import partial
from mind_the_gaps.simulator import tk95_sim, cut_random_segment, simulate_lightcurve, E13_sim_TK95, add_kraft_noise, add_poisson_noise
from mind_the_gaps.stats import create_log_normal, create_uniform_distribution
from mind_the_gaps.psd_models import BendingPowerlaw, Lorentzian, SHO, Matern32
from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling.functional_models import Const1D
from scipy.stats import norm
from mind_the_gaps.readingutils import read_standard_lightcurve
import warnings
import random


def read_command_line(args):
    """ Read command line arguments and builds the input PSD

    Parameters
    ----------
    args: argparse object,
        The dictionary containing the command line arguments

    Returns the input psd, the input parameters and the input command line used.
    """
    models = []
    outpars = ""
    input_command = ""

    if args.Lorentzian:
        # this will contain an array of several Lorentzian params, one entry of set of params for each Lorentzian
        parameters = args.Lorentzian
        for set_of_pars in parameters:
            model_component = Lorentzian()
            input_command += "--Lorentzian %s" % set_of_pars
            # set the parameters
            for parname, param in zip(model_component.param_names, set_of_pars):
                setattr(model_component, parname, np.exp(float(param)))
                outpars += "%s%.3f_" % (parname, float(param))
            models.append(model_component)


    if args.DampedRandomWalk:
        # this will contain an array of several Lorentzian params, one entry of set of params for each Lorentzian
        parameters = args.DampedRandomWalk
        for set_of_pars in parameters:
            model_component = BendingPowerlaw()
            input_command += "--BendingPowerlaw %s" % set_of_pars
            # set the parameters
            for parname, param in zip(model_component.param_names, set_of_pars):
                setattr(model_component, parname, np.exp(float(param)))
                outpars += "%s%.3f_" % (parname, float(param))
            models.append(model_component)


    if args.SHO or args.Granulation:
        # set granulation by default, otherwise it will be overwritten in the loop
        parameters = args.SHO if args.SHO is not None else args.Granulation
        for set_of_pars in parameters:
            model_component = SHO(Q = 1 / np.sqrt(2))
            input_command += "--SHO %s" % set_of_pars
            for parname, param in zip(model_component.param_names, set_of_pars):
                setattr(model_component, parname, np.exp(float(param)))
                outpars += "%s%.3f_" % (parname, float(param))
            models.append(model_component)

    if args.Matern32:
        parameters = args.Matern32
        for set_of_pars in parameters:
            model_component = Matern32()
            input_command += "--Matern32 %s" % args.Powerlaw
            # set the parameters
            for parname, param in zip(model_component.param_names, set_of_pars):
                setattr(model_component, parname, np.exp(float(param)))
                outpars += "%s%.3f_" % (parname, float(param))

            models.append(model_component)

    if args.Powerlaw:
        parameters = args.Powerlaw
        for set_of_pars in parameters:
            model_component = PowerLaw1D()
            input_command += "--Powerlaw %s" % args.Powerlaw
            # set the parameters
            for parname, param in zip(model_component.param_names, set_of_pars):
                setattr(model_component, parname, np.exp(float(param)))
                outpars += "%s%.3f_" % (parname, float(param))

            models.append(model_component)

    psd_model = np.sum(models)

    return psd_model, outpars, input_command


def read_config_file(config_file):
    """Read config file with model and parameter initial values and bounds.

    Parameters
    ----------
    config_file:str,
        The config file

    Returns the kernel,
    """
    data = np.genfromtxt(config_file, names=True, dtype="U25,U25,U25,U25", delimiter="\t", deletechars="")
    if len(np.atleast_1d(data))==1:
        data = np.array([data])

    models = []
    outpars = ""
    for i, row in enumerate(data):
        if row["model"] =="Lorentzian":
            model_component = Lorentzian()
            # set the parameters
            for parname in model_component.param_names:
                parameter_log = float(row["log%s" % parname])
                setattr(model_component, parname, np.exp(parameter_log))
                outpars += "%s%.3f_" % (parname, parameter_log)
        elif row["model"]=="SHO" or row["model"]=="Granulation":
            model_component = SHO(Q=1 / np.sqrt(2))
            # set the parameters
            for parname in model_component.param_names:
                parameter_log = float(row["log%s" % parname])
                setattr(model_component, parname, np.exp(parameter_log))
                outpars += "%s%.3f_" % (parname, parameter_log)

        elif row["model"] =="DampedRandomWalk":
            model_component = BendingPowerlaw()
            # set the parameters
            for parname in model_component.param_names:
                parameter_log = float(row["log%s" % parname])
                setattr(model_component, parname, np.exp(parameter_log))
                outpars += "%s%.3f_" % (parname, parameter_log)
        elif row["model"] =="Powerlaw":
            model_component = Powerlaw1D()
            # set the parameters
            for parname in model_component.param_names:
                parameter_log = float(row["log%s" % parname])
                setattr(model_component, parname, np.exp(parameter_log))
                outpars += "%s%.3f_" % (parname, parameter_log)
        else:
            warnings.warn("Model component %s not implemented! Skipping" % model_component)
            continue

        models.append(model_component)

    psd_model = np.sum(models)

    return psd_model, outpars


ap = argparse.ArgumentParser(description='Generate lightcurve from an input file and input model (either through the command line or through a config file). For the command line only one model component of each kind is allowed')
ap.add_argument('--Lorentzian', metavar="PARAM", nargs=3, help='Lorentzian model parameters.3 parameters: ln_S0, ln_w0 and ln_Q', action="append")
ap.add_argument('--DampedRandomWalk', metavar="PARAM", nargs=2, help='DampedRandomWalk model parameters. 2 parameters: ln_S0 and ln_w0. For several model components, just give the parameters as ln_S0 ln_w0 ln_S1 ln_w1 for model component0, component1, etc',
                action="append")
ap.add_argument('--Matern32', nargs=2, metavar="PARAM", help='Mattern-3/2 model parameters. 2 parameters: ln_S0, ln_rho', action="append")
ap.add_argument('--SHO', nargs=3, metavar="PARAM", help='SHO model parameters. 2 parameters: ln_S0, ln_w0 and ln_Q', action="append")
ap.add_argument('--Granulation', nargs=2, metavar="PARAM", help='Just a SHO model with Q=1/sqrt(2). 2 parameters: ln_S0 and ln_w0', action="append")
ap.add_argument('--Powerlaw', nargs=3, metavar="PARAM", help='Powerlaw (f(x)=S_0 (x/x_0)^-alpha) model parameters (https://docs.astropy.org/en/stable/api/astropy.modeling.powerlaws.PowerLaw1D.html). ln_S0 and ln_x0 and ln_alpha', action="append")
ap.add_argument("-o", "--outdir", nargs='?', help="Output dir", type=str, default="lightcurves")
ap.add_argument("--config", nargs='?', help="Config file with simulation models and parameters", type=str)
ap.add_argument("--rootfile", nargs='?', help="Any characters to append to the output file", type=str,
                default="")
ap.add_argument("-f", '--file', nargs='?', help="File with timestamps, rates, errors, exposures, bkg count rates and bkg rates uncertainties",
                type=str, required=True)
ap.add_argument("-e", '--extension_factor', nargs='?',
                help="Generate lightcurves initially e times longer than input lightcurve to introduce red noise leakage.", type=float, default=5)
ap.add_argument("-s", "--simulator", nargs="?", help="Light curve simulator to use. Shuffle will simply randomized the times of the fluxes based on the observing window",
                type=str, default="E13", choices=["E13", "TK95", "Shuffle"])
ap.add_argument("--pdf", nargs="?", help="PDF of the fluxes (For E13 method). Lognormal by default",
                type=str, choices=["Gaussian", "Lognormal", "Uniform"], default="Lognormal")
ap.add_argument("-n", "--npoints", nargs="?", help="Remove any number of datapoints from the input lightcurve before generating the new one. Default is 0. This is useful to degrade the quality of your lightcurve",
                type=int, default=0)
ap.add_argument("--noise_std", nargs="?", help="Standard deviation of the noise if Gaussian. Otherwise assume Poisson errors based on the counts. Useful in cases where the input lightcurves is in mag or fluxes",
                required=False, default=None, type=float)
args = ap.parse_args()

print("generate_lc:Parsed args: ", args)

count_rate_file = args.file
pdf = args.pdf
points_remove = args.npoints
python_command = "python %s -f %s -s %s -e %.1f --pdf %s -n %d -o %s" % (__file__, args.file, args.simulator, args.extension_factor, pdf, points_remove, args.outdir)

if args.noise_std is not None:
    python_command += "--noise_std %.5f" % args.noise_std

extension_factor = args.extension_factor

file_extension = ".png"
#
timestamps, rates, errors, exposures, bkg_counts, bkg_rate_err = read_standard_lightcurve(count_rate_file)

# remove any random number of points
if points_remove > 0:
    print("%d random datapoints will be removed from the lightcurve")
    ints = random.sample(range(len(timestamps)), points_remove)
    timestamps = np.array([timestamps[i].value for i in range(len(timestamps)) if i not in ints])  << u.s
    rates = np.array([rates[i] for i in range(len(rates)) if i not in ints])
    errors = np.array([errors[i] for i in range(len(rates)) if i not in ints])
    exposures = np.array([exposures[i].value for i in range(len(exposures)) if i not in ints]) << u.s
    bkg_counts = np.array([bkg_counts[i] for i in range(len(bkg_counts)) if i not in ints])
    bkg_rate_err = np.array([bkg_rate_err[i] for i in range(len(bkg_rate_err)) if i not in ints])


duration = timestamps[-1] - timestamps[0]
sim_dt = (np.min(exposures) / 2).to(u.s).value
meanrate = np.mean(rates)
maximum_frequency = 1 / (sim_dt * u.s)
minimum_frequency = 1 / (duration)
livetime = np.sum(exposures)
period_range = "%.1f-%.1f" % ((1 / (maximum_frequency.to("d**-1").value)), (1 / (minimum_frequency.to("d**-1").value)))
print("Duration: %.2f days" % (duration.to(u.d).value))
print("Period range explored for the integration of the variance: %s (days)" % period_range)
print("Live time %.2f days" % (livetime.to(u.d).value))

outdir = args.outdir

if not os.path.isdir(outdir):
    os.mkdir(outdir)
# strip units
timestamps = timestamps.value
exposures = exposures.value

if args.simulator == "E13" or args.simulator=="TK95":

    if args.config is None:
        psd_model, outpars, input_command = read_command_line(args)
        python_command += input_command
    else:
        psd_model, outpars = read_config_file(args.config)

    print("Simulating lightcurve with a length %.2f times longer than input lightcurve with the input model and parameters:" % extension_factor)

    print("PSD model\n---------")
    print(psd_model)
    print("------")
    print("PDF model\n--------\n %s\n---------" % pdf)

    # the 20 is arbitrary but to make sure we get the variance right for highly peaked components
    df_int = 1 / (duration.to(u.s).value * extension_factor)
    dw_int = df_int * 2 * np.pi # frequency differential in angular units
    fnyq = maximum_frequency.value # already divided by 2
    #fnyq = 1 / (2 * lc.dt)

    # prepare TK lightcurve with correct normalization
    lc_mc = simulate_lightcurve(timestamps, psd_model, sim_dt, extension_factor=extension_factor)


    if args.simulator == "E13" and pdf!="Gaussian":
        # obtain the variance and create PDF
        int_freq = np.arange(minimum_frequency.to(1 / u.s).value, fnyq, df_int) # frequencies over which to integrate
        w_int = int_freq * 2 * np.pi
        normalization_factor =  2 / np.sqrt(2 * np.pi) # this factor accounts for the fact that we only integrate positive frequencies and the 1 / sqrt(2pi) from the Fourier transform
        var = np.sum(psd_model(w_int)) * dw_int * normalization_factor

        if pdf == "Lognormal":
            pdfs = [create_log_normal(meanrate, var)]
        elif pdf== "Uniform":
            pdfs = [create_uniform_distribution(meanrate, var)]
        elif pdf=="Gaussian":
            pdfs = [norm(loc=meanrate, scale=np.sqrt(var))]

        half_bins = exposures / 2
        start_time = timestamps[0] - 2 * half_bins[0]
        end_time = timestamps[-1] + 3 * half_bins[-1] # add small offset to ensure the first and last bins are properly behaved when imprinting the sampling pattern
        duration = end_time - start_time

        segment = cut_random_segment(lc_mc, duration)

        lc_rates = E13_sim_TK95(segment, timestamps, pdfs, [1], exposures=exposures)

    elif args.simulator=="TK95" or pdf == "Gaussian":
        warnings.warn("Using TK95 since PDF is Gaussian")
        lc_rates = cut_downsample(lc_mc, timestamps, meanrate, exposures)
        # add the mean
        lc_rates += meanrate - np.mean(lc_rates)
    # add Gaussian or Poisson noise
    if args.noise_std is None:
        if np.all(bkg_counts==0):
            print("Assuming Poisson errors based on count rates")
            # the source is bright enough
            noisy_rates, dy = add_poisson_noise(lc_rates, exposures)
        else:
            print("Assuming Kraft errors based on count rates and background rates")
            noisy_rates, dy, upp_lims = add_kraft_noise(lc_rates, exposures, bkg_counts, bkg_rate_err)
    else:
        noise_std = args.noise_std
        print("Assuming Gaussian white noise with std: %.5f" % noise_std)
        noisy_rates = lc_rates + np.random.normal(scale=noise_std)
        dy = errors[np.argsort(noisy_rates)]

elif args.simulator=="Shuffle":
    outpars = ""
    var = np.var(rates)
    ind = np.random.randint(0, len(timestamps), len(timestamps))
    noisy_rates = rates[ind]
    dy = errors[ind]

outfile = "lc%s%sv%.3E_n%d.dat" % (args.rootfile, outpars, var, points_remove)
outputs = np.asarray([timestamps, noisy_rates, dy])
np.savetxt("%s/%s" % (outdir, outfile), outputs.T, delimiter="\t", fmt="%.6f", header="t\trate\terror")
print("Simulation completed", flush=True)

print("Results stored to %s" % outdir)
