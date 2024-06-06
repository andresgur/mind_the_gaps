#! /usr/bin/env python
# @Author: Andrés Gúrpide <agurpide>
# @Date:   01-09-2020
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 14-04-2022
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import re
import time
import warnings
import astropy.units as u
from astropy.visualization import quantity_support
from multiprocessing import Pool
from shutil import copyfile
from mind_the_gaps.models.psd_models import BendingPowerlaw, Lorentzian, SHO, Matern32, Jitter
from astropy.modeling.powerlaws import PowerLaw1D
from mind_the_gaps.lightcurves import SwiftLightcurve, SimpleLightcurve


def parse_model_name(directory):
    match = re.search(r'_m_([\w\s]+)(?:_fit)?', directory)
    if match:
        return match.group(1).replace("_fit","")
    return None


def create_model(model_names, kernel_names):
    """ Read command line arguments and builds the input PSD

    Parameters
    ----------
    model_names: array_like,
        List of strings with the model names

    Returns the model psd (a summation of astropy.model)
    """
    models = []
    for model_name, kernel_name in zip(model_names, kernel_names):
        name = kernel_name
        if model_name.lower()=="lorentzian":
            model_component = Lorentzian(name=kernel_name)
        elif model_name.lower()=="cosinus":
            model_component = Lorentzian(Q=np.exp(200), name=kernel_name)
        elif model_name.lower()=="dampedrandomwalk":
            model_component = BendingPowerlaw(name=kernel_name)
        elif model_name.lower()=="sho" or model_name.lower()=="granulation":
            model_component = SHO(Q = 1 / np.sqrt(2), name=kernel_name)
        elif model_name.lower()=="matern32":
            model_component=Matern32(name=kernel_name)
        elif model_name.lower()=="powerlaw":
            model_component = PowerLaw1D(name=kernel_name)
        elif model_name.lower()=="jitter":
            model_component = Jitter(name=kernel_name)

        models.append(model_component)

    psd_model = np.sum(models)

    return psd_model


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


def simulate_lcs(input_args): # don't pass the PDFs otherwise we the same pdf samples all the time!
    """Wrapping function for parallelization. It simply takes the samples and simulates the lightcurve"""
    sim, parameters = input_args # this is a dictionary with kernels:terms[X]:log_XX
    # set the parameters to the psd
    outpars = ""
    for kernel_name in kernel_names:
        if simulator.simulator.psd_model.n_submodels > 1:
            model_component = simulator.simulator.psd_model[kernel_name]
        else:
            model_component = simulator.simulator.psd_model
        # get the parameters for the Nth component
        kernel_param_names = [name for name in samples_data.dtype.names if kernel_name in name]

        for parname, kernel_parname in zip(model_component.param_names, kernel_param_names):
            setattr(model_component, parname, np.exp(parameters[kernel_parname]))
            outpars += "%s%.3f_" % (parname, parameters[kernel_parname])
    rates = simulator.generate_lightcurve(extension_factor)
    noisy_rates, dy = simulator.add_noise(rates)
    sample_variance = np.var(noisy_rates)
    rootfile = "_%d_%s_" % (sim, model_names_)
    outfile = "lc%s%sv%.3E.dat" % (rootfile, outpars, sample_variance)
    outputs = np.asarray([lc.times, noisy_rates, dy])
    np.savetxt("lightcurves/%s" % (outfile), outputs.T, delimiter="\t",
               fmt="%.6f", header="t\trate\terror")

    print("Simulation %d/%d completed" % (sim + 1, n_sims), flush=True)


ap = argparse.ArgumentParser(description='Generate lightcurve from a PCCURVE.qdp and celerite samples')
ap.add_argument("--tmin", nargs='?', help="Minimum time in same units as time in the data", type=float, default=0)
ap.add_argument("--tmax", nargs='?', help="Maximum time in same units as time in the data", type=float, default=np.Infinity)
ap.add_argument("-o", "--outdir", nargs='?', help="Output dir", type=str, default="lightcurves_significance")
ap.add_argument("--input_dir", nargs="?", help="Input directory from a celerite run, with the samples in samples.dat",
                type=str, required=False)
ap.add_argument("-n", "--n_sims", nargs='?', help="Number of simulations to perform", type=int, default=1000)
ap.add_argument("-f", '--file', nargs=1, help="Lightcurve file with count rates", type=str)
ap.add_argument("-c", '--cores', nargs='?', help="Number of CPUs for parallelization", type=int, default=10)
ap.add_argument("-s", "--simulator", nargs="?", help="Light curve simulator to use.", type=str, default="E13",
                choices=["E13", "TK95"])
ap.add_argument("-e", '--extension_factor', nargs='?', help="Generate lightcurves initially e times longer than input lightcurve to introduce red noise leakage.",
                type=float, default=5)
ap.add_argument("--pdf", nargs="?", help="PDF of the fluxes (For E13 method). Lognormal by default, ignore if simulator is Shuffle",
                type=str, default="Lognormal", choices=["Gaussian", "Lognormal"])
ap.add_argument("--noise_std", nargs="?", help="Standard deviation of the noise if Gaussian. Otherwise assume Poisson errors based on the counts. Useful in cases where the input lightcurves is in mag or fluxes",
                required=False, default=None, type=float)
args = ap.parse_args()

count_rate_file = args.file[0]
n_sims = args.n_sims
cores = args.cores
pdf = args.pdf
extension_factor = args.extension_factor
simulator = args.simulator
python_command = "python %s -n %d -f %s --tmin %.2f --tmax %.2f -c %d -s %s --input_dir %s --pdf %s -e %.2f -o %s" % (__file__, args.n_sims, args.file[0],
                    args.tmin, args.tmax, cores, simulator, args.input_dir, args.pdf, extension_factor, args.outdir)

quantity_support()

home = os.getenv("HOME")

tmin = args.tmin
tmax = args.tmax
noise_std = args.noise_std
celerite_dir = args.input_dir

celerite_file = "%s/samples.dat" % celerite_dir


if tmin > tmax:
    raise ValueError("Error minimum time %.3f is larger than maximum time: %.3f" % (args.tmin, args.tmax))

if os.path.isfile(count_rate_file):
    try:
        lc = SwiftLightcurve(count_rate_file)
        print("Read as Swift lightcurve...")
    except:
        lc = SimpleLightcurve(count_rate_file)
        print("Read as SimpleLightcurve")

    lc = lc.truncate(tmin, tmax)

    duration = lc.duration * u.s

    time_range = "{:0.3f}-{:0.3f}{:s}".format(lc.times[0], lc.times[-1], "s")
    print("Time range considered: %s" % time_range)
    print("Duration: %.2f days" % (duration.to(u.d).value))

    end_dir = "%s_t%s_N%d_s%s_pdf%s" % (args.outdir, time_range, n_sims, simulator, pdf)

    if args.outdir == "lightcurves_significance":
        outdir = end_dir
    else:
        outdir = "lightcurves_significance_%s" % end_dir

    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # write data out
    lc.to_csv("%s/lc_data.dat" % outdir)

    # just for the plot
    if tmin==0:
        tmin = lc.times[0]
    if tmax==np.inf:
        tmax = lc.times[-1]
    create_data_selection_figure(outdir, tmin, tmax)

    if not os.path.isdir("%s/lightcurves" % outdir):
        os.mkdir("%s/lightcurves" % outdir)

    print("Reading celerite samples from %s" % celerite_file)
    samples_data = np.genfromtxt(celerite_file, delimiter="\t", names=True, deletechars="")

    # read the headers of the parameters
    kernel_pattern = r'kernel(?::terms\[\d+\])?'

    kernel_names = np.unique([re.findall(kernel_pattern, name)[0] for name in samples_data.dtype.names[:-1]])
    model_names_ = parse_model_name(celerite_dir)
    model_names = model_names_.split("_")
    psd_model = create_model(model_names, kernel_names)

    n_kernels = len(kernel_names)

    if n_kernels!= psd_model.n_submodels:
        raise ValueError("Number of model components do not match! %d != %d" % (n_kernels, psd_model.n_submodels))

    print("Input model has %d model components" % n_kernels)

    rand_ints = np.random.randint(0, len(samples_data), n_sims)

    random_param_samples = samples_data[rand_ints]

    start_time = time.time()
    print("\nSimulating %d lightcurves on %d cores" % (n_sims, cores))
    os.chdir("%s" % outdir)

    simulator = lc.get_simulator(psd_model, pdf, noise_std=noise_std)
    # if on the cluster switch to the set number of tasks
    if "SLURM_CPUS_PER_TASK" in os.environ:
        cores = int(os.environ['SLURM_CPUS_PER_TASK'])
        warnings.warn("The numbe of cores is being reset to SLURM_CPUS_PER_TASK = %d " % cores )
    start = time.time()
    with Pool(processes=cores, initializer=np.random.seed) as pool:
        pool.map(
            simulate_lcs, enumerate(random_param_samples)
        )

    os.chdir("../")
    end = time.time()
    time_taken = (end - start) / 60
    print("Time taken: %.2f minutes" % (time_taken))
    # copy the input samples file
    samples_file = os.path.basename(celerite_file)
    copyfile(celerite_file, "%s/%s" % (outdir, samples_file))
    lc_file = os.path.basename(count_rate_file)
    copyfile(count_rate_file, "%s/%s" % (outdir, lc_file))
    file = open("%s/python_command.txt" % outdir, "w+")
    file.write(python_command)
    file.close()
    print("Results stored to %s" % outdir)

else:
    raise ValueError("Input file %s not found! Use -f option to supply your input lightcurve" % count_rate_file)
