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
import subprocess
import astropy.units as u
from astropy.visualization import quantity_support
from multiprocessing import Pool
from functools import partial
from shutil import copyfile
from mind_the_gaps.lightcurves import SwiftLightcurve, SimpleLightcurve


def parse_model_name(directory):
    match = re.search(r'_m_([\w\s]+)(?:_fit)?', directory)
    if match:
        return match.group(1).replace("_fit","")
    return None


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


def simulate_lcs(command): # don't pass the PDFs otherwise we the same pdf samples all the time!
    """Wrapping function for parallelization. It simply takes the samples and simulates the lightcurve"""
    param_command = command[0]
    sim = int(command[1])
    cmd = "generate_lc.py %s %s --rootfile _%d_%s_ " % (generate_lc_args, param_command, sim, model_names_)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, env=os.environ)

    # Wait for the process to finish and get its output
    stdout, stderr = proc.communicate()

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

model_names_ = parse_model_name(celerite_dir)
model_names = model_names_.split("_")

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
    dt = np.median(np.diff(lc.times))
    # in seconds
    sim_dt = np.min(lc.exposures) / 2

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

    n_kernels = len(kernel_names)

    print("Input model has %d model components" % n_kernels)

    rand_ints = np.random.randint(0, len(samples_data), n_sims)

    random_samples = samples_data[rand_ints]

    #if there is more than one kernel we skip the first one (the periodic component)
    #if (model_names[0]=="Lorentzian" or  model_names[0]=="Cosinus") and n_kernels>1:
    #    warnings.warn("Skipping the first model (%s component)" % model_names[0])
    #    model_names = model_names[1:]
    #    kernel_names = kernel_names[1:]


    commands = []
    for row in random_samples:
        command = ""
        for model_name, kernel_name in zip(model_names, kernel_names):
            command += " --%s" %model_name
            # match only the param bit of the kernel:terms:bla bla
            params = [name for name in samples_data.dtype.names if kernel_name in name]
            for par in params:
                command += " %.4f" % row[par]
        commands.append(command)
    print("Here are the first 10 commands")
    print(commands[0:10])
    start_time = time.time()
    print("\nSimulating %d lightcurves on %d cores" % (n_sims, cores))
    os.chdir("%s" % outdir)

    generate_lc_args = "-f lc_data.dat -s %s -o lightcurves --pdf %s -e %.2f"  %(simulator, pdf, extension_factor)
    if noise_std is not None:
        generate_lc_args += " --noise_std %.5f" % noise_std
    # if on the cluster switch to the set number of tasks
    if "SLURM_CPUS_PER_TASK" in os.environ:
        cores = int(os.environ['SLURM_CPUS_PER_TASK'])
        warnings.warn("The numbe of cores is being reset to SLURM_CPUS_PER_TASK = %d " % cores )

    with Pool(processes=cores, initializer=np.random.seed) as pool:
        pool.map(
            simulate_lcs, np.array([commands, np.arange(n_sims)]).T
        )

    os.chdir("../")
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
