# @Author: Andrés Gúrpide <agurpide>
# @Date:   01-09-2020
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 14-04-2022
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import astropy.units as u
from astropy.visualization import quantity_support
import time
from multiprocessing import Pool
from functools import partial
from shutil import copyfile
import warnings
import subprocess
from mind_the_gaps.lightcurves import SwiftLightcurve, SimpleLightcurve
import random


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


def simulate_lcs(sim): # don't pass the PDFs otherwise we the same pdf samples all the time!
    cmd = "python %s/scripts/pythonscripts/mind_the_gaps/scripts/generate_lc.py %s --rootfile _%d_" %  (home, generate_lc_args, sim)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    # Wait for the process to finish and get its output
    stdout, stderr = proc.communicate()

    print("Simulation %d/%d completed" % (sim + 1, n_sims), flush=True)


ap = argparse.ArgumentParser(description='Generate lightcurve from a PCCURVE.qdp and celerite samples')
ap.add_argument("--tmin", nargs='?', help="Minimum time in seconds to truncate the lightcurve", type=float,
                default=0)
ap.add_argument("--tmax", nargs='?', help="Maximum time in seconds to truncate the lightcurve",
                type=float, default=np.Infinity)
ap.add_argument("-o", "--outdir", nargs='?', help="Output dir", type=str, default="lightcurves_goodness")
ap.add_argument("--command", nargs="?", help="Command for the lightcurve simulation e.g. --Lorentzian X Y Z --DampedRandomWalk X Y. (For several model components with the same name, just give the parameters as ln_S0 ln_w0 ln_S1 ln_w1 for model component0, component1, etc)", type=str)
ap.add_argument("--config", nargs="?", help="Config file with the simulation models. Will be used over command argument.", type=str)
ap.add_argument("-n", "--n_sims", nargs='?', help="Number of simulations to perform", type=int, default=1000)
ap.add_argument("-f", '--file', nargs=1, help="File with lightcurve count rates, background rates, etc", type=str)
ap.add_argument("-c", '--cores', nargs='?', help="Number of CPUs for parallelization", type=int, default=11)
ap.add_argument("-up", '--upfile', nargs='?', help="Upper limits file. Default PCUL.qdp", type=str, default='PCUL.qdp')
ap.add_argument("-e", '--extension_factor', nargs='?', help="Generate lightcurves initially e times longer than input lightcurve to introduce red noise leakage.", type=float, default=5)
ap.add_argument("-s", "--simulator", nargs="?", help="Light curve simulator to use", type=str, default="E13",
                choices=["E13", "TK95"])
ap.add_argument("--pdf", nargs="?", help="PDF of the fluxes (For E13 method). Lognormal by default, ignore if simulator is Shuffle",
                type=str, default="Lognormal", choices=["Gaussian", "Lognormal", "Uniform"])
ap.add_argument("--npoints", nargs="?", help="Remove any number of datapoints from the input lightcurve before generating the new one. Default is 0. This is useful to degrade the quality of your lightcurve",
                type=int, default=0)
ap.add_argument("--noise_std", nargs="?", help="Standard deviation of the noise if Gaussian. Otherwise assume Poisson errors based on the counts. Useful in cases where the input lightcurves is in mag or fluxes",
                required=False, default=None, type=float)
args = ap.parse_args()

count_rate_file = args.file[0]
n_sims = args.n_sims
cores = args.cores
extension_factor = args.extension_factor
points_remove = args.npoints
noise_std = args.noise_std
simulator = args.simulator
pdf = args.pdf

base_command = "python %s -n %d -f %s --tmin %.2f --tmax %.2f -c %d -s %s -e %.1f --npoints %d --pdf %s -o %s" % (__file__, args.n_sims, args.file[0],
                args.tmin, args.tmax, cores, simulator, extension_factor, points_remove, pdf, args.outdir)
if args.command is not None:
    python_command = base_command + " --command %s" % args.command
if args.config is not None:
    python_command = base_command + " --config %s" % args.config
if args.noise_std is not None:
    python_command = base_command + " --noise_std %.3f" % args.noise_std

quantity_support()

home = os.getenv("HOME")

plt.style.use('%s/.config/matplotlib/stylelib/paper.mplstyle' % home)

tmin = args.tmin
tmax = args.tmax
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
    # remove any random number of points
    if points_remove > 0:
        print("%d random datapoints will be removed from the lightcurve" % points_remove)
        lc = lc.rand_remove(points_remove)

    duration = lc.duration * u.s
    dt = np.median(np.diff(lc.times))
    # in seconds
    sim_dt = np.min(lc.exposures) / 2

    time_range = "{:0.3f}-{:0.3f}{:s}".format(lc.times[0], lc.times[-1], "s")
    print("Time range considered: %s" % time_range)
    print("Duration: %.2f days" % (duration.to(u.d).value))
    print("Extension factor: %.1f" % extension_factor)

    # create folder name
    model_str = "m"
    if args.config is None:
        if "--Lorentzian" in args.command:
            model_str += "_Lorentzian"
        if "--DampedRandomWalk" in args.command:
            model_str += "_DampedRandomWalk"
        if "--Granulation" in args.command:
            model_str += "_Granulation"
        if "--Powerlaw" in args.command:
            model_str += "_Powerlaw"
        if "--Matern32" in args.command:
            model_str += "_Matern32"
    else:
        model_str += "_config"

    outdirname = "%s_%s_%s_e%.1ft%s_N%d_n%d" % (args.outdir, model_str, pdf, extension_factor,
                                                time_range, n_sims, points_remove)
    if args.outdir == "lightcurves_goodness":
        outdir = outdirname
    else:
        outdir = "lightcurves_goodness_%s" % (outdirname)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    if not os.path.isdir("%s/lightcurves" % outdir):
        os.mkdir("%s/lightcurves" % outdir)

    # write data out
    lc.to_csv("%s/lc_data.dat" % outdir)
    # just for the plot
    if tmin==0:
        tmin = lc.times[0]
    if tmax==np.inf:
        tmax = lc.times[-1]

    create_data_selection_figure(outdir, tmin, tmax)

    start_time = time.time()

    print("\nSimulating %d lightcurves on %d cores" % (n_sims, cores))
    os.chdir("%s" % outdir)

    generate_lc_args = "-f lc_data.dat -s %s -o lightcurves --pdf %s -e %.2f"  %(simulator, pdf, extension_factor)
    if noise_std is not None:
        generate_lc_args += " --noise_std %.5f" % noise_std
    # config file option
    if args.config:
        generate_lc_args += " --config %s" % args.config
    # command option
    else:
        generate_lc_args += " %s" % args.command

    # if on the cluster switch to the set number of tasks
    if "SLURM_CPUS_PER_TASK" in os.environ:
        cores = int(os.environ['SLURM_CPUS_PER_TASK'])
        warnings.warn("The number of cores is being reset to SLURM_CPUS_PER_TASK = %d " % cores )
    with Pool(processes=cores, initializer=np.random.seed) as pool:
        pool.map(simulate_lcs, np.arange(n_sims))
    os.chdir("../")
    # copy the input samples file
    lightcurve_file = os.path.basename(count_rate_file)
    copyfile(count_rate_file, "%s/%s" % (outdir, lightcurve_file))
    # copy any config files
    if args.config is not None:
        base_config = os.path.basename(args.config)
        copyfile(args.config, "%s/%s" % (outdir, base_config))
    file = open("%s/python_command.txt" % outdir, "w+")
    file.write(python_command)
    file.close()
    print("Results stored to %s" % outdir)

else:
    raise ValueError("Input file %s not found! Use -f option to supply your input lightcurve" % count_rate_file)
