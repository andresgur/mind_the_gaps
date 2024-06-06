#! /usr/bin/env python
# @Author: Andrés Gúrpide <agurpide>
# @Date:   01-09-2022
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 12-06-2024

import numpy as np
import os
from shutil import copyfile
import matplotlib.pyplot as plt
import argparse
import astropy.units as u
from astropy.visualization import quantity_support
import time
from mind_the_gaps.models.psd_models import BendingPowerlaw, Lorentzian, SHO, Matern32, Jitter
from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling.functional_models import Const1D
from mind_the_gaps.lightcurves import SimpleLightcurve, SwiftLightcurve
import warnings
from multiprocessing import Pool


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
    model_str = "m"
    # this will contain an array of several Lorentzian params, one entry of set of params for each Lorentzian
    if args.Lorentzian:
        # this will contain an array of several Lorentzian params, one entry of set of params for each Lorentzian
        parameters = args.Lorentzian
        for set_of_pars in parameters:
            model_str += "_Lorentzian"
            model_component = Lorentzian()
            input_command += "--Lorentzian %s" % set_of_pars
            # set the parameters
            for parname, param in zip(model_component.param_names, set_of_pars):
                setattr(model_component, parname, np.exp(float(param)))
                outpars += "%s%.3f_" % (parname, float(param))
            models.append(model_component)

    if args.DampedRandomWalk:
        parameters = args.DampedRandomWalk
        for set_of_pars in parameters:
            model_str += "_DampedRandomWalk"
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
            model_str += "_Granulation"
            model_component = SHO(Q = 1 / np.sqrt(2))
            input_command += "--SHO %s" % set_of_pars
            for parname, param in zip(model_component.param_names, set_of_pars):
                setattr(model_component, parname, np.exp(float(param)))
                outpars += "%s%.3f_" % (parname, float(param))
            models.append(model_component)

    if args.Matern32:
        parameters = args.Matern32
        for set_of_pars in parameters:
            model_str += "_Matern32"
            model_component = Matern32()
            input_command += "--Matern32 %s" % args.Matern32
            # set the parameters
            for parname, param in zip(model_component.param_names, set_of_pars):
                setattr(model_component, parname, np.exp(float(param)))
                outpars += "%s%.3f_" % (parname, float(param))

            models.append(model_component)

    if args.Powerlaw:
        parameters = args.Powerlaw
        for set_of_pars in parameters:
            model_str += "_Powerlaw"
            model_component = PowerLaw1D()
            input_command += "--Powerlaw %s" % args.Powerlaw
            # set the parameters
            for parname, param in zip(model_component.param_names, set_of_pars):
                setattr(model_component, parname, np.exp(float(param)))
                outpars += "%s%.3f_" % (parname, float(param))
            models.append(model_component)

    if args.Jitter:
        parameters = args.Jitter
        for set_of_pars in parameters:
            model_str += "_Jitter"
            model_component = Jitter()
            input_command += "--Jitter %s" % args.Jitter
            # set the parameters
            for parname, param in zip(model_component.param_names, set_of_pars):
                setattr(model_component, parname, np.exp(float(param)))
                outpars += "%s%.3f_" % (parname, float(param))

            models.append(model_component)

    psd_model = np.sum(models)

    return psd_model, outpars, input_command, model_str

def create_data_selection_figure(outdir, tmax, tmin):
    """Create figure with the data range used

    Parameters
    outdir: str,
        Output dir where to store the figure
    tmin: float,
        Minimum time to indicate on the figure (in same units as lightcurve timestamps)
    tmax: float,
        Maximum time to indicate on the figure (in same units as lightcurve timestamps)
    """

    data_selection_figure, data_selection_ax = plt.subplots(1, 1)
    plt.xlabel("Time (days)")
    plt.ylabel("Count rates (ct/s)")
    data_selection_ax.errorbar(lc.times / 3600 / 24, lc.y, yerr=lc.dy, color="black", ls="None", marker=".")

    data_selection_ax.axvline(tmax  / 3600 / 24, ls="--", color="blue")
    data_selection_ax.axvline(tmin   / 3600 / 24, ls="--", color="blue")
    ##data_selection_ax.set_xlim(left=0)
    plt.savefig("%s/data_selection.png" % outdir)
    plt.close(data_selection_figure)

    fig = plt.figure()
    plt.errorbar(lc.times / 3600 / 24, lc.y, yerr=lc.dy, color="black", ls="None", marker=".")
    plt.xlabel("Time (days)")
    plt.ylabel("Count rates (ct/s)")
    plt.savefig("%s/lc_segment.png" % outdir)
    plt.close(fig)


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
    model_str = "m"
    outpars = ""
    for i, row in enumerate(data):
        if row["model"] =="Lorentzian":
            model_str += "_Lorentzian"
            model_component = Lorentzian()
            # set the parameters
            for parname in model_component.param_names:
                parameter_log = float(row["log%s" % parname])
                setattr(model_component, parname, np.exp(parameter_log))
                outpars += "%s%.3f_" % (parname, parameter_log)
        elif row["model"]=="SHO" or row["model"]=="Granulation":
            model_str += "_Granulation"
            model_component = SHO(Q=1 / np.sqrt(2))
            # set the parameters
            for parname in model_component.param_names:
                parameter_log = float(row["log%s" % parname])
                setattr(model_component, parname, np.exp(parameter_log))
                outpars += "%s%.3f_" % (parname, parameter_log)

        elif row["model"] =="DampedRandomWalk":
            model_str += "_DampedRandomWalk"
            model_component = BendingPowerlaw()
            # set the parameters
            for parname in model_component.param_names:
                parameter_log = float(row["log%s" % parname])
                setattr(model_component, parname, np.exp(parameter_log))
                outpars += "%s%.3f_" % (parname, parameter_log)
        elif row["model"] =="Powerlaw":
            model_str += "_Powerlaw"
            model_component = PowerLaw1D()
            # set the parameters
            for parname in model_component.param_names:
                parameter_log = float(row["log%s" % parname])
                setattr(model_component, parname, np.exp(parameter_log))
                outpars += "%s%.3f_" % (parname, parameter_log)
        elif row["model"]=="Jitter":
            model_str += "_Jitter"
            model_component = Jitter()
            for parname in model_component.param_names:
                parameter_log = float(row["log%s" % parname])
                setattr(model_component, parname, np.exp(parameter_log))
                outpars += "%s%.3f_" % (parname, parameter_log)
        elif row["model"]=="Matern32":
            model_str += "_Matern32"
            model_component = Matern32()
            for parname in model_component.param_names:
                parameter_log = float(row["log%s" % parname])
                setattr(model_component, parname, np.exp(parameter_log))
                outpars += "%s%.3f_" % (parname, parameter_log)

        else:
            warnings.warn("Model component %s not implemented! Skipping" % model_component)
            continue

        models.append(model_component)

    psd_model = np.sum(models)

    return psd_model, outpars, model_str

def simulate_lcs(sim):
    """Create lightcurves"""
    rates = simulator.generate_lightcurve(extension_factor)
    noisy_rates, dy = simulator.add_noise(rates)
    save_lc(lc.times, noisy_rates, dy, sim)
    return

def shuffle_lcs(sim):
    """Create lightcurves by randomly shuffling the measurements"""
    ind = np.random.randint(0, lc.n, lc.n)
    noisy_rates = lc.y[ind]
    dy = lc.dy[ind]
    save_lc(lc.times, noisy_rates, dy, sim)
    return


def save_lc(times, rates, dy, sim):
    """Dummy function to store the generated lightcurve"""
    name = "%s%s" % (rootname, outpars)
    sample_variance = np.var(rates)
    outfile = "lc%d_%s_v%.3E_n%d.dat" % (sim, name, sample_variance, points_remove)
    outputs = np.asarray([times, rates, dy])
    np.savetxt(outdir + "/lightcurves/%s" % outfile, outputs.T, delimiter="\t",
                       fmt="%.6f", header="t\trate\terror")
    print("Simulation %d/%d completed" % (sim, n_sims), flush=True)


ap = argparse.ArgumentParser(description='Generate lightcurve(s) from an input file and input model (either through the command line or through a config file). For the command line only one model component of each kind is allowed')
ap.add_argument("--tmin", nargs='?', help="Minimum time in seconds to truncate the lightcurve", type=float,
                default=0)
ap.add_argument("--tmax", nargs='?', help="Maximum time in seconds to truncate the lightcurve",
                type=float, default=np.Infinity)
ap.add_argument('--Lorentzian', metavar="PARAM", nargs=3, help='Lorentzian model parameters.3 parameters: ln_S0, ln_Q and ln_w0', action="append")
ap.add_argument('--DampedRandomWalk', metavar="PARAM", nargs=2, help='DampedRandomWalk model parameters. 2 parameters: ln_S0 and ln_w0. For several model components, just give the parameters as ln_S0 ln_w0 ln_S1 ln_w1 for model component0, component1, etc',
                action="append")
ap.add_argument("-c", '--cores', nargs='?', help="Number of CPUs for parallelization. Default 11", type=int, default=11)
ap.add_argument("-n", "--n_sims", nargs='?', help="Number of simulations to perform. Default 1", type=int, default=1)
ap.add_argument('--Matern32', nargs=2, metavar="PARAM", help='Mattern-3/2 model parameters. 2 parameters: ln_S0, ln_rho', action="append")
ap.add_argument('--SHO', nargs=3, metavar="PARAM", help='SHO model parameters. 2 parameters: ln_S0, ln_Q and ln_w0', action="append")
ap.add_argument('--Granulation', nargs=2, metavar="PARAM", help='Just a SHO model with Q=1/sqrt(2). 2 parameters: ln_S0 and ln_w0', action="append")
ap.add_argument('--Powerlaw', nargs=3, metavar="PARAM", help='Powerlaw (f(x)=S_0 (x/x_0)^-alpha) model parameters (https://docs.astropy.org/en/stable/api/astropy.modeling.powerlaws.PowerLaw1D.html). ln_S0 and ln_x0 and ln_alpha', action="append")
ap.add_argument('--Jitter', nargs=1, metavar="PARAM", help='Jitter (white noise) model parameters: ln_sigma (log of the standard deviation of the noise)', action="append")
ap.add_argument("-o", "--outdir", nargs='?', help="Output dir", type=str, default="lightcurves")
ap.add_argument("--config", nargs='?', help="Config file with simulation models and parameters", type=str)
ap.add_argument("--rootname", nargs='?', help="Any characters to append to the output lightcurve files", type=str,
                default="")
ap.add_argument("-f", '--file', nargs='?', help="File with timestamps, rates, errors, exposures, bkg count rates and bkg rates uncertainties",
                type=str, required=True)
ap.add_argument("-e", '--extension_factor', nargs='?',
                help="Generate lightcurves initially e times longer than input lightcurve to introduce red noise leakage.", type=float, default=5)
ap.add_argument("-s", "--simulator", nargs="?", help="Light curve simulator to use. Shuffle will simply randomized the times of the fluxes based on the observing window",
                type=str, default="E13", choices=["E13", "TK95", "Shuffle"])
ap.add_argument("--pdf", nargs="?", help="PDF of the fluxes (For E13 method). Lognormal by default",
                type=str, choices=["Gaussian", "Lognormal", "Uniform"], default="Lognormal")
ap.add_argument("--npoints", nargs="?", help="Remove any number of datapoints from the input lightcurve before generating the new one. Default is 0. This is useful to degrade the quality of your lightcurve",
                type=int, default=0)
ap.add_argument("--noise_std", nargs="?", help="Standard deviation of the noise if Gaussian. Otherwise assume Poisson errors based on the counts. Useful in cases where the input lightcurves is in mag or fluxes",
                required=False, default=None, type=float)
args = ap.parse_args()

print("generate_lc:Parsed args: ", args)

count_rate_file = args.file
pdf = args.pdf
points_remove = args.npoints
n_sims = args.n_sims
cores = args.cores
outdir = args.outdir
rootname = args.rootname
python_command = "python %s -f %s -s %s -e %.1f --pdf %s -n %d -o %s" % (__file__, args.file, args.simulator, args.extension_factor, pdf, points_remove, args.outdir)

if args.noise_std is not None:
    python_command += "--noise_std %.5f" % args.noise_std

extension_factor = args.extension_factor

tmin = args.tmin
tmax = args.tmax
if tmin > tmax:
    raise ValueError("Error minimum time %.3f is larger than maximum time: %.3f" % (args.tmin, args.tmax))

file_extension = ".png"

if os.path.isfile(count_rate_file):
    try:
        lc = SwiftLightcurve(count_rate_file)
        print("Read as Swift lightcurve...")
    except:

        lc = SimpleLightcurve(count_rate_file)
        print("Read as SimpleLightcurve")

    lc = lc.truncate(tmin, tmax)

    # remove any random number of points
    if points_remove > 0:
        print("%d random datapoints will be removed from the lightcurve")
        lc = lc.rand_remove(points_remove)

    duration = lc.duration << u.s
    # in seconds and unitless
    livetime = np.sum(lc.exposures) * u.s
    duration = lc.duration * u.s

    time_range = "{:0.3f}-{:0.3f}{:s}".format(lc.times[0], lc.times[-1], "s")
    print("Time range considered: %s" % time_range)
    print("Duration: %.2f days" % (duration.to(u.d).value))
    print("Extension factor: %.1f" % extension_factor)
    print("Live time %.2f days" % (livetime.to(u.d).value))

    if "SLURM_CPUS_PER_TASK" in os.environ:
        cores = int(os.environ['SLURM_CPUS_PER_TASK'])
        warnings.warn("The number of cores is being reset to SLURM_CPUS_PER_TASK = %d " % cores )

    if args.simulator.lower() == "e13" or args.simulator.lower()=="tk95":

        if args.config is None:
            psd_model, outpars, input_command, model_str = read_command_line(args)
            python_command += input_command
        else:
            psd_model, outpars, model_str = read_config_file(args.config)

        outdir += "_" + model_str

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        if not os.path.isdir(outdir + "/lightcurves"):
            os.mkdir(outdir + "/lightcurves")
        print("Simulating lightcurve with a length %.2f times longer than input lightcurve with the input model and parameters:" % extension_factor)

        print("PSD model\n---------")
        print(psd_model)
        print("------")
        print("PDF model\n--------\n %s\n---------" % pdf)
        simulator = lc.get_simulator(psd_model, pdf, args.noise_std)

        start = time.time()

        with Pool(processes=cores, initializer=np.random.seed) as pool:
            pool.map(simulate_lcs, np.arange(n_sims))

        # the 20 is arbitrary but to make sure we get the variance right for highly peaked components
        # eventually remove these things
        #df_int = 1 / (duration.to(u.s).value * extension_factor)
        #dw_int = df_int * 2 * np.pi # frequency differential in angular units
        #fnyq = maximum_frequency.value # already divided by 2

        # prepare TK lightcurve with correct normalization
        #lc_mc = simulate_lightcurve(timestamps, psd_model, sim_dt, extension_factor=extension_factor)

        #half_bins = lc.exposures / 2
        #start_time = timestamps[0] - 2 * half_bins[0]
        #end_time = timestamps[-1] + 3 * half_bins[-1] # add small offset to ensure the first and last bins are properly behaved when imprinting the sampling pattern
        #duration = end_time - start_time

        #segment = cut_random_segment(lc_mc, duration)
        #sample_variance = np.var(segment.countrate)

        #if args.simulator == "E13" and pdf!="Gaussian":
            # obtain the variance and create PDF
            ##int_freq = np.arange(minimum_frequency.to(1 / u.s).value, fnyq, df_int) # frequencies over which to integrate
            ##w_int = int_freq * 2 * np.pi
            ##normalization_factor =  2 / np.sqrt(2 * np.pi) # this factor accounts for the fact that we only integrate positive frequencies and the 1 / sqrt(2pi) from the Fourier transform
            ##var = np.sum(psd_model(w_int)) * dw_int * normalization_factor
            #if pdf == "Lognormal":
        #        pdfs = [create_log_normal(meanrate, sample_variance)]
        #    elif pdf== "Uniform":
        #        pdfs = [create_uniform_distribution(meanrate, sample_variance)]
        #    elif pdf=="Gaussian":
        #        pdfs = [norm(loc=meanrate, scale=np.sqrt(sample_variance))]

        #    lc_rates = E13_sim_TK95(segment, timestamps, pdfs, [1],
        #                            exposures=lc.exposures, max_iter=400)

        #elif args.simulator=="TK95" or pdf == "Gaussian":
        #    warnings.warn("Using TK95 since PDF is Gaussian")
        #    lc_rates = downsample(segment, timestamps, lc.exposures)
            # adjust the mean
        #    lc_rates += meanrate - np.mean(lc_rates)

        # add Gaussian or Poisson noise
    #    if args.noise_std is None:
    #        if np.all(lc.bkg_rate==0):
    #            print("Assuming Poisson errors based on count rates")
                # the source is bright enough
    #            noisy_rates, dy = add_poisson_noise(lc_rates, lc.exposures)
    #        else:
    #            print("Assuming Kraft errors based on count rates and background rates")
    #            noisy_rates, dy, upp_lims = add_kraft_noise(lc_rates, lc.exposures,
    #                                                        lc.bkg_rate * lc.exposures,
    #                                                        lc.bkg_rate_err)
    #    else:
    #        noise_std = args.noise_std
    #        print("Assuming Gaussian white noise with std: %.5f" % noise_std)
    #        noisy_rates = lc_rates + np.random.normal(scale=noise_std, size=len(lc_rates))
    #        dy = errors[np.argsort(noisy_rates)]

    elif args.simulator.lower()=="shuffle":
        outdir += "_" + "shuffle"

        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        if not os.path.isdir(outdir + "/lightcurves"):
            os.mkdir(outdir + "/lightcurves")
        outpars = ""
        start = time.time()
        with Pool(processes=cores, initializer=np.random.seed) as pool:
            pool.map(shuffle_lcs, np.arange(n_sims))

    end = time.time()
    time_taken = (end - start) / 60
    print("Generated %d lightcurves in: %.2f minutes" % (n_sims, time_taken))
    # just for the plot
    if tmin==0:
        tmin = lc.times[0]
    if tmax==np.inf:
        tmax = lc.times[-1]

    create_data_selection_figure(outdir, tmin, tmax)
    # write the truncated data out
    lc.to_csv("%s/lc_data.dat" % outdir)
    # copy the input samples file
    lightcurve_file = os.path.basename(count_rate_file)
    copyfile(count_rate_file, "%s/%s" % (outdir, lightcurve_file))
    print("Results stored to %s" % outdir)
