# @Author: Andrés Gúrpide <agurpide>
# @Date:   01-09-2020
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 15-02-2023
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize
from shutil import copyfile
import celerite
from celerite.modeling import ConstantModel
import corner
import emcee
import astropy.units as u
from multiprocessing import Pool
from mind_the_gaps.celerite_models import Lorentzian, DampedRandomWalk, Cosinus, BendingPowerlaw, LinearModel
import readingutils as ru
import warnings
from scipy.stats import norm, kstest
from astropy.stats import sigma_clip
os.environ["OMP_NUM_THREADS"] = "1" # https://emcee.readthedocs.io/en/stable/tutorials/parallel/

def standarized_residuals(data, model, uncer,  ouput):

    std_res = (data - model) / uncer


    xrange = np.arange(np.min(std_res), np.max(std_res), 0.05)

    counts, bins = np.histogram(std_res, bins='auto')
    bin_widths = np.diff(bins)
    normalized_res = counts / (len(std_res) * bin_widths)

    fig, ax = plt.subplots()
    plt.bar(bins[:-1], normalized_res, width=bin_widths, edgecolor="black", facecolor="None", label="St residuals")
    mu, std = norm.fit(std_res, loc=0, scale=1)
    plt.axvline(mu, label="%.4f$\pm$%.4f" % (mu, std))

    # ks test
    gauss = norm(loc=0, scale=1)
    plt.plot(xrange, gauss.pdf(xrange))
    kstest_res = kstest(gauss.rvs, std_res[~np.isinf(std_res)], N=2000)
    plt.text(0.15, 0.8, "p-value = %.3f" %(kstest_res.pvalue),
             transform=ax.transAxes,fontsize=24)
    plt.legend()
    plt.savefig("%s/standarized_res_%s.png" % (outdir, ouput), dpi=100)
    plt.close(fig)

    print("Mu and std of the std resduals (%.3f+-%.3f)" % (mu, std))
    print("P-value (Values <0.05 indicate the fit is bad) for the std residuals: %.5f" % kstest_res.pvalue)
    np.savetxt("%s/std_res_%s.dat" % (outdir, ouput), std_res, header="res", fmt="%.5f")
    return



def read_config_file(config_file, walkers=32):
    """Read config file with model and parameter initial values and bounds.

    Parameters
    ----------
    config_file:str,
        The config file

    Returns the kernel,
    """
    model_info = np.genfromtxt(config_file, names=True, dtype="U25,U25,U25,U25",
                               delimiter="\t", deletechars="")
    if len(np.atleast_1d(model_info))==1:
        model_info = np.array([model_info])

    kernels = np.empty(len(model_info), dtype=celerite.terms.Term)

    outmodels = []
    initial_params = None
    columns = []
    labels = []
    # either nothing if only one kernel or start numbering the terms if more than one
    kernel_string = "" if len(model_info)==1 else "terms[0]:"

    for kernel_counter, row in enumerate(model_info):
        print("Adding %s component" % row["model"])
        outmodels.append("%s" % row["model"])

        w = np.log(two_pi / (np.array(row["P"].split(":")).astype(float) * days_to_seconds))
        S_0 = np.array(row["logS_0"].split(":")).astype(float) #already in log

        if row["model"] == "Lorentzian" or row["model"] =="SHO" or row["model"]=="Powerlaw":
            Q = np.log(np.array(row["Q"].split(":")).astype(float))
            bounds = dict(log_S0=(S_0[0], S_0[2]), log_Q=(Q[0], Q[2]), log_omega0=(w[2], w[0]))

            # create the variables if it's the first kernel
            kernel_columns = ["kernel:%slog_omega0" % kernel_string, "kernel:%slog_S0" %kernel_string, "kernel:%slog_Q"%kernel_string]
            kernel_labels = [r"$P$ (days)", r"log $S_0$", r"log Q"]
            if initial_params is None:
                initial_pars = np.array([np.random.uniform(S_0[0], S_0[2], walkers),
                                         np.random.uniform(Q[0], Q[2], walkers),
                                        np.random.uniform(w[2], w[0], walkers)])
                initial_params = initial_pars
            else:
                initial_params = np.append(initial_params, initial_pars, axis=0)

            if row["model"]=="SHO":
                kernel = celerite.terms.SHOTerm(log_S0=S_0[1], log_Q=Q[1], log_omega0=w[1], bounds=bounds)
            elif row["model"]=="Lorentzian":
                kernel = Lorentzian(log_S0=S_0[1], log_omega0=w[1], log_Q=Q[1], bounds=bounds)
            elif row["model"]=="Powerlaw":
                kernel = BendingPowerlaw(log_S0=S_0[1], log_omega0=w[1], log_Q=Q[1], bounds=bounds)

        # two param model components (S_0 and omega)
        elif row["model"] == "DampedRandomWalk" or row["model"] =="Granulation" or row["model"]=="Cosinus":
            bounds = dict(log_S0=(S_0[0], S_0[2]), log_omega0=(w[2], w[0]))

            if row["model"] == "DampedRandomWalk":
                kernel = DampedRandomWalk(log_S0=S_0[1], log_omega0=w[1],
                                                 bounds=bounds)
            elif row["model"] =="Granulation":
                Q = 1 / np.sqrt(2)
                kernel = celerite.terms.SHOTerm(log_S0=S_0[1], log_Q=np.log(Q), log_omega0=w[1],
                                                 bounds=bounds)
                kernel.freeze_parameter("log_Q")
            elif row["model"]=="Cosinus":
                kernel = Cosinus(log_S0=S_0[1], log_omega0=w[1], bounds=bounds)

            kernel_labels = [r"log $S_N$", r"log $\omega_N$"]
            kernel_columns = ["kernel:%s%s" %(kernel_string, name) for name in kernel.get_parameter_names()]
            if initial_params is None:
                initial_params = np.array([np.random.uniform(S_0[0], S_0[2], walkers),
                                            np.random.uniform(w[2], w[0], walkers)])
            else:
                initial_params = np.append(initial_params, np.array([np.random.uniform(S_0[0], S_0[2], walkers),
                                            np.random.uniform(w[2], w[0], walkers)]), axis=0)

        elif row["model"]=="Matern32":
            log_rho = np.log(np.array(row["P"].split(":")).astype(float) * days_to_seconds)
            bounds = dict(log_sigma=(S_0[0], S_0[2]), log_rho=(log_rho[0], log_rho[2]))
            kernel = celerite.terms.Matern32Term(log_sigma=S_0[1], log_rho=log_rho[1], eps=1e-7,
                                             bounds=bounds)

            kernel_columns = ["kernel:%s%s" %(kernel_string, name) for name in kernel.get_parameter_names()]
            kernel_labels = [r"log $\sigma$", r"log $\rho$"]

            if initial_params is None:
                initial_params = np.array([np.random.uniform(S_0[0], S_0[2], walkers),
                                            np.random.uniform(w[2], w[0], walkers)])

            else:
                initial_params = np.append(initial_params, np.array([np.random.uniform(S_0[0], S_0[2], walkers),
                                            np.random.uniform(w[2], w[0], walkers)]), axis=0)

        elif row["model"]=="Jitter":
            bounds = dict(log_sigma=(S_0[0], S_0[2]))
            kernel = celerite.terms.JitterTerm(log_sigma=S_0[1], bounds=bounds)
            kernel_columns = ["kernel:%s%s" %(kernel_string, name) for name in kernel.get_parameter_names()]
            kernel_labels = [r"log $\sigma$"]

            if initial_params is None:
                initial_params = np.array([np.random.uniform(S_0[0], S_0[2], walkers)])
            else:
                initial_params = np.append(initial_params, np.array([np.random.uniform(S_0[0], S_0[2], walkers)]), axis=0)

        else:
            warnings.warn("Component %s unrecognised. Skipping..." % row["model"])

        columns.extend(kernel_columns)
        labels.extend(kernel_labels)
        kernels[kernel_counter] = kernel
        kernel_string = "terms[%d]:" % (kernel_counter + 1)

    total_kernel = np.sum(kernels)
    return total_kernel, initial_params, labels, columns, outmodels

def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)


def log_probability(params, y, gp):
    """https://celerite.readthedocs.io/en/stable/tutorials/modeling/"""
    gp.set_parameter_vector(params)

    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    return lp + gp.log_likelihood(y)


def check_gaussianity(measurements):
        fig, ax = plt.subplots()
        bins, n, p = plt.hist(measurements, density=True)
        mu, std = norm.fit(measurements, loc=np.mean(measurements),
                           scale=np.std(measurements))
        rv = norm(loc=mu, scale=std)
        kstest_res = kstest(rv.rvs, measurements, N=10000)
        plt.text(0.02, 0.8, "p-value = %.3f" % (kstest_res.pvalue),
                 transform=ax.transAxes,fontsize=24)
        x = np.arange(min(measurements), max(measurements), 0.01)
        plt.plot(x, rv.pdf(x), label="Gaussian (%.3f$\pm$%.3f)" % (mu, std))
        plt.xlabel("Count-rate (ct/s)")
        plt.ylabel("PDF")
        plt.legend()
        plt.savefig("%s/pdf.png" % outdir)
        plt.close(fig)


def read_XRT_data(input_file, tmin=0, tmax=np.inf):
    """Modify this function to read your data and filter it by time"""
    try:
        data = ru.readPCCURVE("%s" % input_file, minSNR=0, minSigma=0, minCts=0)
    except ValueError:
        data = ru.readsimplePCCURVE("%s" % input_file, minSigma=0)
    time_column = data.dtype.names[0]
    rate_column = data.dtype.names[3]

    filtered_data = data[np.where((data["%s" % time_column] >= tmin) & (data["%s" % time_column] <=tmax))]

    time = filtered_data[time_column]
    y = filtered_data[rate_column]
    yerr = (-filtered_data["%sneg" % rate_column] + filtered_data["%spos" % rate_column]) / 2
    return time, y, yerr


def read_data2(input_file, tmin=0, tmax=np.inf):
    """Modify this function to read your data and filter it by time"""

    data = np.genfromtxt("%s" % input_file, names=True)
    time_column = data.dtype.names[0]
    rate_column = data.dtype.names[1]
    err_column = data.dtype.names[2]

    filtered_data = data[np.where((data["%s" % time_column] >= tmin) & (data["%s" % time_column] <=tmax))]
    if time_column.lower() in ["mjd", "jd", "day"]:
        print("Time in days")
        time = filtered_data[time_column] * days_to_seconds
    else:
        time = filtered_data[time_column]
    y = filtered_data[rate_column]
    yerr = filtered_data[err_column]
    return time, y, yerr


def read_rxte_data(input_file, tmin=0, tmax=np.inf):
    """Modify this function to read your data and filter it by time"""
    data = np.genfromtxt("%s" % input_file, names=True, skip_header=8)
    time_column = data.dtype.names[0]
    rate_column = data.dtype.names[1]
    err_column = data.dtype.names[2]

    filtered_data = data[np.where((data["%s" % time_column] >= tmin) & (data["%s" % time_column] <=tmax))]
    if time_column in ["mjd", "jd", "day"]:
        print("Time in days")
        time = filtered_data[time_column] * days_to_seconds
    else:
        time = filtered_data[time_column]

    y = filtered_data[rate_column]
    yerr = filtered_data[err_column]
    return time, y, yerr


def read_fermi_lat(input_file, tmin=0, tmax=np.inf):
    """Modify this function to read your data and filter it by time"""

    data = np.genfromtxt("%s" % input_file, names=True, delimiter=",")
    time_column = data.dtype.names[0]
    rate_column = data.dtype.names[1]

    filtered_data = data[np.where((data["%s" % time_column] >= tmin) & (data["%s" % time_column] <=tmax))]

    if "MJD" in time_column:
        time = filtered_data[time_column] * days_to_seconds
    else:
        time = filtered_data[time_column]

    y = filtered_data[rate_column]
    yerr = (np.abs(filtered_data["%s_err_neg" % rate_column]) + filtered_data["%s_err_pos" % rate_column]) / 2
    return time, y, yerr

two_pi = 2 * np.pi
days_to_seconds = 24 * 3600

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description='Perform fit and MCMC error estimation using celerite models (see Foreman-Mackey et al 2017. 10.3847/1538-3881/aa9332)')
    ap.add_argument("-c", "--config_file", nargs='?', help="Config file with initial parameter constraints", type=str, required=True)
    ap.add_argument("--cores", nargs='?', help="Number of cores for parallelization.Default 16", type=int, default=16)
    ap.add_argument("--tmin", nargs='?', help="Minimum time in data time units", type=float, default=0)
    ap.add_argument("--tmax", nargs='?', help="Maximum time in data time units", type=float, default=np.Infinity)
    ap.add_argument("-o", "--outdir", nargs='?', help="Output dir name", type=str, default="celerite")
    ap.add_argument("--fit", help="Whether to fit the data and start MCMC with best fit values. Default False", action="store_true")
    ap.add_argument("-m", "--meanmodel", help="A model for the mean. If provided the mean will be fitted. Default fix constant model (i.e. no fitting)",
                    choices=["Constant", "Linear"], default=None, nargs="?")
    ap.add_argument("--log", help="Whether to take the log of the data. Default False", action="store_true")
    ap.add_argument("input_file", nargs=1, help="Lightcurve '\t' separated file with timestamps, rates, errors", type=str)
    args = ap.parse_args()


    if "SLURM_CPUS_PER_TASK" in os.environ:
        cores = int(os.environ['SLURM_CPUS_PER_TASK'])
        warnings.warn("The numbe of cores is being reset to SLURM_CPUS_PER_TASK = %d " % cores )
    else:
        cores = args.cores

    nwalkers = 32 if args.cores < 64 else args.cores * 2 # https://stackoverflow.com/questions/69234421/multiprocessing-the-python-module-emcee-but-not-all-available-cores-on-the-ma

    home = os.getenv("HOME")
    # set style file
    if os.path.isfile('%s/.config/matplotlib/stylelib/paper.mplstyle' % home):
        plt.style.use('%s/.config/matplotlib/stylelib/paper.mplstyle' % home)
    else:
        warnings.warn("Style file not found")

    count_rate_file = args.input_file[0]
    tmin = args.tmin
    tmax = args.tmax

    if tmin >= tmax:
        raise ValueError("Minimum time (%.2es) is greater than or equal to maximum time (%.3es)!" %(tmin.value, tmax.value))

    try:
        time, y, yerr = read_data2(count_rate_file, tmin, tmax)
    except:
        try:
            time, y, yerr = read_XRT_data(count_rate_file, tmin, tmax)
        except:
            try:
                time, y, yerr = read_fermi_lat(count_rate_file, tmin, tmax)
            except:
                time, y, yerr = read_rxte_data(count_rate_file, tmin, tmax)

    # get only positive measurements
    time, y, yerr = time[y>0], y[y>0], yerr[y>0]
    if args.log:
        # this follows from error propagation
        yerr = yerr / y
        y = np.log(y)

    duration = time[-1] - time[0] # seconds
    print("Duration: %.2fd" % (duration / days_to_seconds))
    # filter data according to input parameters
    time_range = "%.3f-%.3f" % (time[0], time[-1])
    dt = np.mean(np.diff(time))
    print("Mean sampling: %.2ed" % (dt / days_to_seconds))
    print("Number of datapoints: %d" % len(time))
    # see https://dx.doi.org/10.3847/1538-4357/acbe37 see Vaughan 2003 Eq 9 and Appendix
    psd_noise_level = 2 * dt * np.mean(yerr**2)
    clipped_time_diff = sigma_clip(np.diff(time), sigma=2.5, maxiters=10, masked=False)
    psd_noise_level_median = 2 * np.median(np.diff(clipped_time_diff)) * np.mean(yerr**2)
    # timestamps to plot the model
    t_samples = 20 * len(time) if len(time) < 5000 else 6 * len(time)
    time_model = np.linspace(np.min(time), np.max(time), t_samples) # seconds
    # for plotting the PSD (if less than 500d extend a bit)
    if duration / days_to_seconds < 800 and len(time) < 5000:
        df = 0.5 / duration
        frequencies = np.arange(1 / (5 * duration), 1 / (0.1 * dt), df) #* two_pi 0.1 / duration)
    elif len(time) < 5000:
        df = 1 / duration
        frequencies = np.arange(1 / (duration), 1 / (0.1 * dt), df) #* two_pi 0.1 / duration)
    else:
        df = 1 / duration
        frequencies = np.arange(1 / (duration), 1 / (2 * dt), df) #* two_pi 0.1 / duration)
    days_freqs = frequencies * days_to_seconds
    w_frequencies = frequencies * two_pi

    prefix = "celerite_t%s" %(time_range)  if args.outdir=="celerite" else "celerite_%s_t%s" % (args.outdir, time_range)
    kernel, initial_samples, labels, cols, outmodels = read_config_file(args.config_file, nwalkers)

    outmodelsstr = "_" + "_".join(outmodels)

    outdir = "%s_m%s" % (prefix, outmodelsstr)
    if args.fit:
        outdir += "_fit"
    if args.log:
        outdir += "_log"
    if args.meanmodel:
        outdir += "_mean_%s" % args.meanmodel

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # check for Gaussianity in the distribution of fluxes
    check_gaussianity(y)

    time_range_file = open("%s/time_range.txt" % (outdir), "w+")
    time_range_file.write("%.5f-%.5f s" % (time[0], time[-1]))
    time_range_file.close()

    initial_samples = initial_samples.T
    data_mean = np.mean(y)
    meanmodel = ConstantModel(data_mean, bounds=[(np.min(y), np.max(y))])

    if args.meanmodel=="Constant":
        meanlabels = ["$\mu$"]
        cols.extend(["mean:mean"])
    elif args.meanmodel=="Linear":
        slope_guess = np.sign(y[-1] - y[0])
        minindex = np.argmin(time)
        maxindex = np.argmax(time)
        slope_bound = (y[maxindex] - y[minindex]) / (time[maxindex] - time[minindex])
        if slope_guess > 0 :
            min_slope = slope_bound
            max_slope = -slope_bound
        else:
            min_slope = -slope_bound
            max_slope = slope_bound

        meanmodel = LinearModel(np.mean(min_slope + max_slope), data_mean, bounds=[(-np.inf, np.inf), (-np.inf, np.inf)])
        meanlabels = ["$m$", "$b$"]
        cols.extend(["mean:slope", "mean:intercept"])

    if args.meanmodel:
        print("Fitting for the mean. Initial value %.3f" % np.mean(y))
        fit_mean = True
        if args.log:
            for meanlabel in meanlabels:
                labels.append("$\log$ " +meanlabel)
        else:
            labels.extend(meanlabels)

    else:
        print("Mean will be kept fixed at %.3f" % data_mean)
        fit_mean = False

    gp = celerite.GP(kernel, mean=meanmodel, fit_mean=fit_mean)

    print("parameter_dict:\n{0}\n".format(gp.get_parameter_dict()))
    print("parameter_names:\n{0}\n".format(gp.get_parameter_names()))
    print("parameter_vector:\n{0}\n".format(gp.get_parameter_vector()))
    print("parameter_bounds:\n{0}\n".format(gp.get_parameter_bounds()))
    gp.compute(time, yerr)  # You always need to call compute once.
    print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))
    initial_params = gp.get_parameter_vector()
    par_names = list(gp.get_parameter_names())
    bounds = np.array(gp.get_parameter_bounds())
    # create init model figure
    model_fig, ax = plt.subplots()
    plt.xlabel("Frequency (days$^{-1}$)")
    plt.ylabel("Power")
    # plot model components
    for term, model_name in zip(gp.kernel.terms, outmodels):
        ax.plot(days_freqs, term.get_psd(w_frequencies), ls="--",
                             label="%s" % model_name)
    # plot total model
    plt.plot(days_freqs, gp.kernel.get_psd(w_frequencies), ls="--", label="Total")
    # twin axis
    plt.legend()
    ax2 = ax.twiny()
    ax.set_xscale("log")
    xticks = ax.get_xticks()
    ax2.set_xscale("log")
    ax2.set_xlabel('Period (days)')
    x2labels = ["%.2f" %w for w in 1/xticks]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(x2labels)
    ax2.set_xlim(ax.get_xlim())
    plt.yscale("log")
    model_fig.savefig("%s/init_model.png" % outdir,
                      dpi=100)
    plt.close(model_fig)

    ndim = len(initial_params)

    if args.fit:
        # solution contains the information about the fit. .x is the best fit parameters
        solution = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=gp.get_parameter_bounds(), args=(y, gp))
        gp.set_parameter_vector(solution.x)
        print("Final log-likelihood (it is maximized so higher better): {0}".format(-solution.fun))
        print(solution)
        if not solution.success:
            raise Exception("The solver did not converge!\n %s" % solution.message)
        # bic information (equation 54 from Foreman et al 2017,
        # see also https://github.com/dfm/celerite/blob/ad3f471f06b18d233f3dab71bb1c20a316173cae/paper/figures/simulated/wrong-qpo.ipynb)
        k = len(gp.get_parameter_dict())
        bic = 2 * solution.fun + k * np.log(len(y))
        aicc = 2 * k + 2 * solution.fun + 2 * k * (k + 1) / (len(y) - k - 1)
        print("BIC (the smaller the better): %.2f" % bic)

        best_params = gp.get_parameter_dict()
        print("Best-fit params\n -------")
        print(best_params)

        for term in gp.kernel.terms:
            if "kernel:log_omega0" in term.get_parameter_names():
                omega_par = "kernel:log_omega0"
                period = (two_pi / (np.exp(best_params[omega_par]) * days_to_seconds))
                print("Best fit period\n----------\n %.3f days\n -----" % period)
            elif "kernel:log_rho" in term.get_parameter_names():
                omega_par = "kernel:log_rho"
                period = np.exp(best_params[omega_par] * days_to_seconds)
                print("Best fit period\n----------\n %.3f days\n -----" % period)

        header = ""
        outstring = ''
        for parname, best_par in zip(par_names, best_params):
            header += "%s\t" % parname
            outstring += '%.3f\t' % best_params[best_par]

        header += "BIC\tAICc\tloglikelihood\tparameters\tdatapoints"
        outstring += "%.1f\t%.1f\t%.2f\t%d\t%d" % (bic, aicc, gp.log_likelihood(y), k, len(y))
        out_file = open("%s/best_fit_par.dat" % (outdir), "w+")
        out_file.write("%s\n%s" % (header, outstring))
        out_file.close()

        celerite_figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
        # get model prediction
        pred_mean, pred_var = gp.predict(y, time_model, return_var=True)
        pred_std = np.sqrt(pred_var)

        standarized_residuals(y, pred_mean, pred_std, "best_fit")
        color = "orange"
        ax1.errorbar(time / days_to_seconds, y, yerr=yerr, fmt=".k", capsize=0)
        if args.log:
            ax1.set_ylabel(" log(Count-rate (ct/s))")
        else:
            ax1.set_ylabel("Count-rate (ct/s)")
        ax1.plot(time_model / days_to_seconds, pred_mean, color="orange")
        ax1.fill_between(time_model / days_to_seconds, pred_mean + pred_std, pred_mean - pred_std, color="orange", alpha=0.3,
                         edgecolor="none")

        outputs = np.array([time_model / days_to_seconds, pred_mean, pred_std])
        np.savetxt("%s/best_fit.dat" % outdir, outputs.T, header="time(d)\tmodel\tstd", fmt="%.6f")
        # no need to pass time here as if this is omitted the coordinates will be assumed to be x from GP.compute() and an efficient method will be used to compute the prediction (https://celerite.readthedocs.io/en/stable/python/gp/)
        pred_mean, pred_var = gp.predict(y, time, return_var=True)
        ax2.errorbar(time / days_to_seconds, (y - pred_mean) / yerr, yerr=1, fmt=".k", capsize=0)
        ax2.axhline(y=0, ls="--", color="#002FA7")
        ax2.set_ylabel("Residuals")
        ax2.set_xlabel("Time (days)")
        celerite_figure.savefig("%s/best_fit.png" % (outdir))
        plt.close(celerite_figure)

        # plot PSD from the best fit
        psd_best_fit_figure, psd_best_fit_ax = plt.subplots()
        psd_best_fit_ax.set_xlabel("Frequency [days$^{-1}$]")
        psd_best_fit_ax.set_ylabel("Power")
        psd_best_fit_ax.set_yscale("log")
        psd = gp.kernel.get_psd(w_frequencies)
        psd_best_fit_ax.plot(days_freqs, psd, color=color, label="Total model")
        psd_best_fit_ax.axhline(psd_noise_level_median, ls="solid", color="black", zorder=-10)
        psd_best_fit_ax.axhline(psd_noise_level, ls="solid", color="black", zorder=-10)
        ax2 = psd_best_fit_ax.twiny()
        psd_best_fit_ax.set_xscale("log")
        ax2.set_xscale("log")
        ax2.set_xlabel('Period (days)')
        xticks = psd_best_fit_ax.get_xticks()
        ax2.set_xticks(xticks)
        x2labels = ['{:01.0f}'.format(w) for w in 1/xticks]
        ax2.set_xticklabels(x2labels)
        ax2.set_xlim(psd_best_fit_ax.get_xlim())
        psd_best_fit_figure.savefig("%s/psd_best_fit.png" % outdir, bbox_inches="tight")

        for term in gp.kernel.terms:
            psd = term.get_psd(w_frequencies)
            psd_best_fit_ax.plot(days_freqs, psd, ls="--")

            if "log_omega0" in term.get_parameter_names():
                log_omega = term.get_parameter_dict()["log_omega0"]
                best_freq = np.exp(log_omega) / 2 / np.pi * days_to_seconds
                psd_best_fit_ax.axvline(best_freq, label="%.2f days" % (1 / best_freq), ls="--",
                                        color="black")
            elif "log_rho" in term.get_parameter_names():
                log_rho = term.get_parameter_dict()["log_rho"]
                # see bottom of https://celerite.readthedocs.io/en/stable/python/kernel/
                best_freq = np.sqrt(3) / np.exp(log_rho) * days_to_seconds
                psd_best_fit_ax.axvline(best_freq, label="%.2f days" % (1 / best_freq), ls="--",
                                                         color="black")

        psd_best_fit_ax.legend()
        psd_best_fit_figure.savefig("%s/psd_best_fit_components.png" % outdir, bbox_inches="tight")
        plt.close(psd_best_fit_figure)

        # reinitialize best fit parameters
        warnings.warn("Initial samples reset based on best-fit parameters")
        initial_samples = np.empty((nwalkers, ndim))
        #initial_samples = solution.x + 1e-2 * np.random.randn(nwalkers, ndim)
        #reshaped_params = np.resize(solution.x, (nwalkers, ndim))
        #initial_samples = solution.x + 1e-1 * np.random.randn(nwalkers, ndim)
        # Gaussian centered around the best params and 1+-sigma of 10%
        for i in range(nwalkers):
            accepted = False

            while not accepted:
                # Generate random values centered around the best-fit parameters
                perturbed_params = np.random.normal(solution.x, np.abs(solution.x) / 10.0)

                # Check if the perturbed parameters are within the bounds
                if np.all(np.logical_and(bounds[:, 0] <= perturbed_params, perturbed_params <= bounds[:, 1])):
                    initial_samples[i] = perturbed_params
                    accepted = True
    # set from the example, https://celerite.readthedocs.io/en/stable/tutorials/modeling/

    # start from the best fit values
    #initial_params * np.abs(np.random.randn(nwalkers, ndim)) # only positive values from the Gaussian

    #max_n = 100000
    max_n = 250000
    #max_n = 3000
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)
    every_samples = 800
    # This will be useful to testing convergence
    old_tau = np.inf

    print("Running chain for a maximum of %d samples with %d walkers until the chain has a length 100xtau using %d cores" % (max_n, nwalkers, cores))
    print("Initial samples\n----------")
    print(initial_samples)

    with Pool(cores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, args=(y, gp))

        # Now we'll sample for up to max_n steps
        for sample in sampler.sample(initial_samples, iterations=max_n, progress=True):
            # Only check convergence every 100 steps
            if sampler.iteration % every_samples:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                print("Convergence reached after %d samples!" % sampler.iteration)
                break
            old_tau = tau

    acceptance_ratio = sampler.acceptance_fraction
    print("Acceptance ratio: (%)")
    print(acceptance_ratio)
    print("Correlation parameters:")
    print(tau)
    mean_tau = np.mean(tau)
    print("Mean correlation time:")
    print(mean_tau)

    if not converged:
        warnings.warn("The chains did not converge!")
        # tau will be very large here, so let's reduce the numbers
        thin = int(mean_tau / 4)
        discard = int(mean_tau) * 10 # avoid blowing up if discard is larger than the number of samples, this happens if the fit has not converged

    else:
        discard = int(mean_tau * 40)
        if discard > max_n:
            discard = int(mean_tau * 10)
        thin = int(mean_tau / 2)

    fig = plt.figure()
    autocorr_index = autocorr[:index]
    n = every_samples * np.arange(1, index + 1)
    plt.plot(n, autocorr_index, "-o")
    plt.ylabel("Mean $\\tau$")
    plt.xlabel("Number of steps")
    plt.savefig("%s/autocorr.png" % outdir, dpi=100)
    plt.close(fig)

    # plot the entire chain
    chain = sampler.get_chain(flat=True)
    median_values = np.median(chain, axis=0)
    chain_fig, axes = plt.subplots(ndim, sharex=True, gridspec_kw={'hspace': 0.05, 'wspace': 0})
    if len(np.atleast_1d(axes))==1:
        axes = [axes]
    for param, parname, ax, median in zip(chain.T, par_names, axes, median_values):
        ax.plot(param, linestyle="None", marker="+", color="black")
        ax.set_ylabel(parname.replace("kernel:", "").replace("log_", ""))
        ax.axhline(y=median)

    print("Discarding the first %d samples" % discard)
    for ax in axes:
        ax.axvline(discard * nwalkers, ls="--", color="red")

    # calculate R stat
    samples = sampler.get_chain(discard=discard)

    whithin_chain_variances = np.var(samples, axis=0) # this has nwalkers, ndim (one per chain and param)

    samples = sampler.get_chain(flat=True, discard=discard)
    between_chain_variances = np.var(samples, axis=0)

    print("R-stat (values close to 1 indicate convergence)")
    print(whithin_chain_variances / between_chain_variances[np.newaxis, :]) # https://stackoverflow.com/questions/7140738/numpy-divide-along-axis

    final_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    loglikes = sampler.get_log_prob(discard=discard, thin=thin, flat=True)

    axes[-1].set_xlabel("Step Number")
    chain_fig.savefig("%s/chain_samples.png" % outdir, bbox_inches="tight", dpi=100)
    plt.close(chain_fig)

    # save samples
    inds = [par_names.index("%s" % c) for c in cols]

    samples = final_samples[:, inds]

    outputs = np.vstack((samples.T, loglikes))

    header = "\t".join(cols) + "\tloglikehood"

    np.savetxt("%s/samples.dat" % outdir, outputs.T, delimiter="\t", fmt="%.5f",
              header=header)

    best_loglikehood = np.argmax(loglikes)
    # save contours for each param indicating the maximum
    for i, parname in enumerate(cols):
        plt.figure()
        par_vals = samples[:,i]
        plt.scatter(par_vals, loglikes)
        plt.scatter(par_vals[best_loglikehood], loglikes[best_loglikehood],
                    label="%.2f, L = %.2f" % (par_vals[best_loglikehood], loglikes[best_loglikehood]))
        plt.legend()
        plt.xlabel("%s" % parname)
        plt.ylabel("$L$")
        plt.savefig("%s/%s.png" % (outdir, parname), dpi=100)
        plt.close()
    # remove the log dependency in the parameters
    #samples[:, :] = np.exp(samples[:, :])
    # ind_omega = par_names.index("kernel:%s" % "log_omega0")

    # save max params and standarized residuals
    best_pars = final_samples[best_loglikehood]

    np.savetxt("%s/parameter_max.dat" % outdir, best_pars.T,
               header="\t".join(gp.get_parameter_names()), fmt="%.2f")
    max_loglikehood = loglikes[best_loglikehood]
    k = len(gp.get_parameter_dict())
    bic = - 2 * loglikes[best_loglikehood] + k * np.log(len(y))
    aicc = 2 * k - 2 * loglikes[best_loglikehood] + 2 * k * (k + 1) / (len(y) - k - 1)

    out_file = open("%s/max_log_likehood.txt" % (outdir), "w+")
    out_file.write("#loglikehood\tBIC\tAICc\tn\tp\n")
    out_file.write("%.3f\t%.2f\t%.2f\t%d\t%d" % (max_loglikehood, bic, aicc, len(y), k))
    out_file.close()

    gp.set_parameter_vector(best_pars)
    gp.compute(time, yerr)
    # (No need to pass time as it'll be assumed the same datapoints as GP compute (https://celerite.readthedocs.io/en/stable/python/gp/)
    best_model, var = gp.predict(y, return_var=True, return_cov=False) # omit time for faster computing time

    standarized_residuals(y, best_model, np.sqrt(var), "max")

    median_parameters = np.median(final_samples, axis=0)
    distances = np.linalg.norm(final_samples - median_parameters, axis=1)
    closest_index = np.argmin(distances)
    median_log_likelihood = loglikes[closest_index]

    header = ""
    outstring = ''

    for i, parname in enumerate(cols):
        q_16, q_50, q_84 = corner.quantile(samples[:,i], [0.16, 0.5, 0.84]) # your x is q_50
        header += "%s\t" % parname
        dx_down, dx_up = q_50-q_16, q_84-q_50
        if "slope" in parname:
            outstring += '%.2e$_{-%.2e}^{+%.2e}$ & ' % (q_50, dx_down, dx_up)
        else:
            outstring += '%.2f$_{-%.2f}^{+%.2f}$ & ' % (q_50, dx_down, dx_up)
        if "Q" in parname:
            header += "%s & " % parname.replace("log_Q", "Q")
            q_16, q_50, q_84 = np.exp(q_16), np.exp(q_50), np.exp(q_84)
            dx_down, dx_up = q_50-q_16, q_84-q_50
            outstring += '%.1f$^{+%.1f}_{-%.1f}$ & ' % (q_50, dx_up, dx_down)

        # store also in days
        if "omega" in parname:
            header += "%s & " % parname.replace("log_omega", "P")
            periods = two_pi / (np.exp(samples[:,i]) * days_to_seconds)
            q_16, q_50, q_84 = corner.quantile(periods, [0.16, 0.5, 0.84])
            dx_down, dx_up = q_50-q_16, q_84-q_50
            outstring += '%.2f$_{-%.2f}^{+%.2f} & ' % (q_50, dx_down, dx_up)

            header += "%s\t" % parname.replace("log_omega", "omega")
            freqs = np.exp(samples[:,i]) / 2 / np.pi * days_to_seconds # a factor 2pi was missing
            q_16, q_50, q_84 = corner.quantile(freqs, [0.16, 0.5, 0.84])
            dx_down, dx_up = q_50-q_16, q_84-q_50
            outstring += '%.4f$_{-%.4f}^{+%.4f} & ' % (q_50, dx_down, dx_up)


    header += "loglikehood"
    outstring += "%.3f" % median_log_likelihood
    out_file = open("%s/parameter_medians.dat" % (outdir), "w+")
    out_file.write("%s\n%s" % (header, outstring))
    out_file.close()

    # Convert the frequency into period of the oscillator to period for plotting purposes
    #omegas = ("omega" in gp.get_parameter_names()).sum()
    Qs = np.count_nonzero(["log_Q" in param for param in gp.get_parameter_names()])
    omegas = np.count_nonzero(["omega" in param for param in gp.get_parameter_names()])

    # if there is oscillator component convert it to Period (days)
    if Qs>=1:
        omega_index = 0
        for i in range(Qs):
            samples[:, omega_index] = two_pi / (np.exp(samples[:, omega_index]) * days_to_seconds)
            omega_index += 3

    #ranges = [(median - range_span * median, median + range_span * median) for median in medians]
    #figsize=(20, 16))
    # for the levels, see this https://corner.readthedocs.io/en/latest/pages/sigmas.html we plot 1,2,3 sigma contours
    print("Generating corner plot...")
    ranges = np.ones(ndim) * 0.9
    corner_fig = corner.corner(samples, labels=labels, title_fmt='.1f', #range=ranges,
                               quantiles=[0.16, 0.5, 0.84], show_titles=True,
                               title_kwargs={"fontsize": 18}, max_n_ticks=3, labelpad=0.08,
                               levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2 ** 2))) # plots 1 and 2 sigma levels
    corner_fig.savefig("%s/corner_fig.pdf" % outdir)
    plt.close(corner_fig)

    # finally plot final PSD and Model
    # MODEL
    model_figure, model_ax = plt.subplots()
    model_ax.set_xlabel("Time (days)")
    model_ax.set_ylabel("Count-rate ct/s [0.3 - 10 keV]")
    model_ax.errorbar(time / 3600 / 24, y, yerr=yerr, fmt=".k", capsize=0)
    model_figure.savefig("%s/lightcurve.png" % outdir)
    # PSD
    psd_figure, psd_ax = plt.subplots()
    psd_ax.set_xlabel("Frequency (days$^{-1}$)")
    psd_ax.set_ylabel("Power")
    psd_ax.set_yscale("log")
    #psd_ax.axhline(psd_noise_level, ls="--", color="black")

    color = "black"
    # draw 1000 samples from the final distributions and create plots
    n_samples = 1500
    psds = np.empty((n_samples, len(frequencies)))
    psd_components = np.empty((len(gp.kernel.terms), n_samples, len(frequencies)))

    models = np.ones((n_samples, t_samples))
    mdels_st_res = np.ones((n_samples, len(time)))
    print("Generating %d samples for model and PSD plots" % n_samples)
    for index, sample in enumerate(final_samples[np.random.randint(len(samples), size=n_samples)]):
        gp.set_parameter_vector(sample)
        psd = gp.kernel.get_psd(w_frequencies)
        model = gp.predict(y, time_model, return_cov=False)
        mdels_st_res = gp.predict(y, time, return_cov=False)
        model_ax.plot(time_model / days_to_seconds, model, color="orange", alpha=0.3)
        psd_ax.plot(frequencies * days_to_seconds, psd, color=color, alpha=0.3)
        models[index] = model
        psds[index] = psd
        for term_i, term in enumerate(gp.kernel.terms):
            psd_components[term_i, index] = term.get_psd(w_frequencies)

    model_figure.savefig("%s/model_mcmc_samples.png" % outdir)
    plt.close(model_figure)

    #psd_ax.set_xlim(min_f * days_to_seconds, max_f * days_to_seconds)
    psd_ax.margins(x=0.0)
    # second axis
    ax2 = psd_ax.twiny()
    xticks = psd_ax.get_xticks()
    x2labels = ['{:01.0f}'.format(w) for w in 1 / xticks]
    psd_ax.set_xscale("log")
    ax2.set_xlabel('Period (days)')
    ax2.set_xscale("log")
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(x2labels)
    ax2.set_xlim(psd_ax.get_xlim())
    psd_figure.savefig("%s/psd_samples.png" % outdir, bbox_inches="tight")

    # Median and standard deviation figures
    # model
    model_figure, model_ax = plt.subplots()
    model_ax.set_xlabel("Time (days)")
    model_ax.set_ylabel("Count-rate ct/s [0.3 - 10 keV]")
    model_ax.errorbar(time / days_to_seconds, y, yerr=yerr, fmt=".k", capsize=0)
    m = np.nanpercentile(models, [16, 50, 84], axis=0)
    model_ax.plot(time_model / days_to_seconds, m[1], color="orange")
    model_ax.fill_between(time_model / days_to_seconds, m[0], m[2], alpha=0.3, color="orange")
    model_ax.errorbar(time / days_to_seconds, y, yerr=yerr, fmt=".k", capsize=0)
    model_figure.savefig("%s/model_mcmc_median.png" % outdir, bbox_inches="tight")
    plt.close(model_figure)
    outputs = np.array([time_model / days_to_seconds, m[1], m[0], m[2]])
    np.savetxt("%s/model_median.dat" % outdir, outputs.T, header="time(d)\tmodel\tlower\tupper", fmt="%.6f")

    standarized_residuals(y, np.median(mdels_st_res), np.std(mdels_st_res), "mcmc_median")

    # PSD median
    psd_median_figure, psd_median_ax = plt.subplots()
    psd_median_ax.set_xlabel("Frequency (days$^{-1}$)")
    psd_median_ax.set_ylabel("Power")
    psd_median_ax.set_yscale("log")

    p = np.percentile(psds, [16, 50, 84], axis=0)
    psd_output = np.array([frequencies, p[0], p[1], p[2], psd_noise_level * np.ones(len(frequencies))])
    np.savetxt("%s/psds.dat" % outdir, psd_output.T, delimiter="\t",
              fmt="%.8f", header="f\t16%\t50%\t84%\tnoise")

    psd_median_ax.plot(frequencies * days_to_seconds, p[1], color="orange")
    psd_median_ax.fill_between(frequencies * days_to_seconds, p[0], p[2], color=color, alpha=0.3)
    psd_median_ax.margins(x=0)
    #psd_median_ax.set_xlim(min_f * days_to_seconds, max_f * days_to_seconds)
    # if we are using a noise component with a break
    if omegas > Qs:
        break_period = two_pi / (np.exp(median_parameters[-1]) * days_to_seconds)
        psd_median_ax.axvline(1 / break_period, label="Break ($P$ = %.3f $d$)" % break_period, ls="--", color="black")
        psd_median_ax.legend()
    # second axis
    ax2 = psd_median_ax.twiny()
    psd_median_ax.set_xscale("log")
    xticks = psd_median_ax.get_xticks()
    x2labels = ['{:01.1f}'.format(w) for w in 1/xticks]
    ax2.set_xscale("log")
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(x2labels)
    ax2.set_xlabel('Period (days)')
    ax2.set_xlim(psd_median_ax.get_xlim())

    psd_median_figure.savefig("%s/psd_median.png" % outdir, bbox_inches="tight")

    # add components
    colors = ["green", "red", "indigo"]
    for term_i, term in enumerate(gp.kernel.terms):
        p = np.percentile(psd_components[term_i], [16, 50, 84], axis=0)
        psd_median_ax.plot(frequencies * days_to_seconds, p[1], color=colors[term_i], ls="--")
        psd_median_ax.fill_between(frequencies * days_to_seconds, p[0], p[2], alpha=0.3, color=colors[term_i])
        np.savetxt("%s/psds_comp%d.dat" % (outdir, term_i), np.array([frequencies, p[0], p[1], p[2], psd_noise_level * np.ones(len(frequencies))]).T,
                    delimiter="\t", fmt="%.8f", header="f\t16%\t50%\t84%\tnoise")
    xticks = psd_median_ax.get_xticks()
    x2labels = ['{:01.1f}'.format(w) for w in 1/xticks]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(x2labels)
    # add noise level
    psd_median_ax.axhline(psd_noise_level_median, ls="solid", color="black", zorder=-10)
    psd_median_ax.axhline(psd_noise_level, ls="--", color="black")
    psd_median_figure.savefig("%s/psd_median_comps.png" % outdir, bbox_inches="tight")


    config_file_name = os.path.basename(args.config_file)
    copyfile(args.config_file, "%s/%s" % (outdir, config_file_name))

    with open("%s/samples.info" % outdir, "w+") as file:
        file.write("#samples\tdiscard\tthin\ttau\n")
        file.write("%d\t%d\t%d\t%.2f\n" % (sampler.iteration, discard, thin, mean_tau))

    python_command = "python %s %s --tmin %.2f --tmax %.2f -c %s -m %s -o %s" % (__file__, args.input_file[0],
                        args.tmin, args.tmax, args.config_file, args.meanmodel, args.outdir)
    if args.fit:
        python_command += " --fit"

    with open("%s/python_command.txt" % outdir, "w+") as file:
        file.write(python_command)
    print("Results stored to %s" % outdir)
