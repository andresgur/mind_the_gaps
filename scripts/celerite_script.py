# @Author: Andrés Gúrpide <agurpide>
# @Date:   01-09-2020
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 15-02-2023
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from shutil import copyfile
import corner
import astropy.units as u
from mind_the_gaps.stats import bic, aicc
from mind_the_gaps.configs import read_config_file
from mind_the_gaps.lightcurves import FermiLightcurve, SwiftLightcurve, SimpleLightcurve
from mind_the_gaps.gpmodelling import GPModelling
import warnings
from scipy.stats import norm, ks_1samp
from astropy.stats import sigma_clip

os.environ["OMP_NUM_THREADS"] = "1" # https://emcee.readthedocs.io/en/stable/tutorials/parallel/

def standarized_residuals(data, model, uncer,  ouput):
    """Computes the standarized residuals and performs a KS test, testing for Gaussianity (mu=0, sigma=1)

    Parameters
    ----------
    data: array_like,
        The array of data (count rates, magnitudes, etc)
    model: array_like,
        The model prediction at the same timestamps as for the data
    uncer: array_like,
        The uncertainty on the data
    output:str,
        A string to append to the output figures

    Returns the p value of rejecting the standarized residuals as being drawn from a Gaussian (mu=0, sigma=1)
    """

    std_res = (data - model) / uncer

    counts, bins = np.histogram(std_res, bins='auto')
    bin_widths = np.diff(bins)
    normalized_res = counts / (len(std_res) * bin_widths)

    # dividing by the model
    mu, std = norm.fit(std_res, loc=0, scale=1)
    fig, ax = plt.subplots()
    plt.bar(bins[:-1], normalized_res, width=bin_widths, edgecolor="black",
            facecolor="None", label="(D - M) /$\\sigma_M$ \n (%.4f$\pm$%.4f)" % (mu, std))

    # dividing by the data
    std_res_data = (data - model) / yerr
    counts, bins = np.histogram(std_res_data, bins='auto')
    bin_widths = np.diff(bins)
    normalized_res_data = counts / (len(std_res_data) * bin_widths)
    mu_data, std_data = norm.fit(std_res_data, loc=0, scale=1)
    plt.bar(bins[:-1], normalized_res_data, width=bin_widths, edgecolor="blue",
            facecolor="None", label="(D - M) /$\\sigma_D$ \n (%.4f$\pm$%.4f)" % (mu_data, std_data))

    # ks test
    gauss = norm(loc=0, scale=1)
    xrange = np.arange(np.min(std_res), np.max(std_res), 0.05)
    plt.plot(xrange, gauss.pdf(xrange), color="black")
    kstest_res = ks_1samp(std_res[~np.isinf(std_res)], gauss.cdf)
    plt.text(0.1, 0.7, "p-value = %.3f %%" % (kstest_res.pvalue * 100),
             transform=ax.transAxes, fontsize=24)
    plt.legend()
    plt.savefig("%s/standarized_res_%s.png" % (outdir, ouput), dpi=100)
    plt.close(fig)

    # plot aucorr
    fig = plt.figure()
    lags, acf, _, __ = plt.acorr(std_res, maxlags=None)
    # sigma
    plt.axhspan(-1 / np.sqrt(len(std_res)), 1 / np.sqrt(len(std_res)), alpha=0.3, color="black")
    plt.xlim(left=0)
    plt.ylim(top=np.max(acf[acf < 0.99]) + 0.05) # the max after the 1.0 at the 0 lag
    plt.xlabel("Time lag")
    plt.ylabel("Residuals ACF")
    plt.savefig("%s/res_acf_%s.png" % (outdir, ouput))
    np.savetxt("%s/res_acf_%s.dat" % (outdir, ouput), np.array([lags, acf]).T, header="lags\tacf", fmt="%.5f")
    plt.close(fig)

    print("Mu and std of the std resduals (%.3f+-%.3f)" % (mu, std))
    print("P-value (Values <0.05 indicate the fit is bad) for the std residuals: %.5f" % kstest_res.pvalue)
    np.savetxt("%s/std_res_%s.dat" % (outdir, ouput), std_res, header="res", fmt="%.5f")
    return kstest_res.pvalue


def check_gaussianity(measurements):
    """Checks whether the input array follows a Gaussian distribution

    Parameters
    ----------
    measurements:array_like
        Array to be checked
    """
    fig, ax = plt.subplots()
    bins, n, p = plt.hist(measurements, density=True)
    mu, std = norm.fit(measurements, loc=np.mean(measurements),
                       scale=np.std(measurements))
    rv = norm(loc=mu, scale=std)
    kstest_res = ks_1samp(measurements, rv.cdf)
    plt.text(0.02, 0.8, "p-value = %.3f" % (kstest_res.pvalue),
             transform=ax.transAxes,fontsize=24)
    x = np.arange(min(measurements), max(measurements), 0.01)
    plt.plot(x, rv.pdf(x), label="Gaussian (%.3f$\pm$%.3f)" % (mu, std))
    plt.xlabel("Count-rate (ct/s)")
    plt.ylabel("PDF")
    plt.legend()
    plt.savefig("%s/pdf.png" % outdir)
    plt.close(fig)


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
                    choices=["Constant", "Linear", "Gaussian"], default=None, nargs="?")
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

    try:
        lc = SimpleLightcurve(count_rate_file, skip_header=0)
    except:
        try:
            # Swift xrt
            lc = SwiftLightcurve(count_rate_file, minCts=0)
        except:
            # fermi
            lc = FermiLightcurve(count_rate_file)

    lc = lc.truncate(tmin, tmax)

    time, y, yerr = lc.times, lc.y, lc.dy

    # get only positive measurements
    if np.count_nonzero(y<0):
        warnings.warn("Lightcurve has some negative bins!")

    if args.log:
        # this follows from error propagation
        yerr = yerr / y
        y = np.log(y)

    duration = lc.duration # seconds
    print("Duration: %.2fd" % (duration / days_to_seconds))
    # filter data according to input parameters
    time_range = "%.3f-%.3f" % (time[0], time[-1])
    dt = np.mean(np.diff(time))
    print("Mean sampling: %.2ed" % (dt / days_to_seconds))
    print("Number of datapoints: %d" % lc.n)
    # see https://dx.doi.org/10.3847/1538-4357/acbe37 see Vaughan 2003 Eq 9 and Appendix
    normalization_factor =  2 / np.sqrt(2 * np.pi) # this factor accounts for the fact that we only integrate positive frequencies and the 1 / sqrt(2pi) from the Fourier transform
    psd_noise_level = 2 * dt * np.mean(yerr**2) / (2 * np.pi * normalization_factor)
    # timestamps to plot the model
    t_samples = 18 * lc.n if lc.n < 5000 else 7 * lc.n
    time_model = np.linspace(np.min(time), np.max(time), t_samples) # seconds
    # for plotting the PSD (if less than 500d extend a bit)
    if duration / days_to_seconds < 800 and lc.n < 5000:
        df = 0.5 / duration
        frequencies = np.arange(1 / (5 * duration), 1 / (0.1 * dt), df) #* two_pi 0.1 / duration)
    elif lc.n < 5000:
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
        if args.meanmodel.lower()=="constant":
            cols.extend(["mean:value"])
            meanlabels = ["$\mu$"]
        elif args.meanmodel.lower()=="linear":
            cols.extend(["mean:slope", "mean:intercept"])
            meanlabels = ["$m$", "$b$"]
        elif args.meanmodel.lower()=="gaussian":
            cols.extend(["mean:mean", "mean:sigma", "mean:amplitude"])
            meanlabels = ["$\mu$", "$\sigma$", "$A$"]
        labels.extend(meanlabels)

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    # check for Gaussianity in the distribution of fluxes
    check_gaussianity(y)

    time_range_file = open("%s/time_range.txt" % (outdir), "w+")
    time_range_file.write("%.5f-%.5f s" % (time[0], time[-1]))
    time_range_file.close()
    # gp compute is called inside this class so no need to call it
    gpmodel = GPModelling(lc, kernel, args.meanmodel)

    print("parameter_dict:\n{0}\n".format(gpmodel.gp.get_parameter_dict()))
    print("parameter_names:\n{0}\n".format(gpmodel.gp.get_parameter_names()))
    print("parameter_vector:\n{0}\n".format(gpmodel.gp.get_parameter_vector()))
    print("parameter_bounds:\n{0}\n".format(gpmodel.gp.get_parameter_bounds()))
    print("Initial log likelihood: {0}".format(gpmodel.gp.log_likelihood(y)))
    par_names = list(gpmodel.gp.get_parameter_names())
    bounds = np.array(gpmodel.gp.get_parameter_bounds())

    # create init model figure
    model_fig, ax = plt.subplots()
    plt.xlabel("Frequency (days$^{-1}$)")
    plt.ylabel("Power")
    # plot model components
    for term, model_name in zip(gpmodel.gp.kernel.terms, outmodels):
        ax.plot(days_freqs, term.get_psd(w_frequencies), ls="--",
                             label="%s" % model_name)
    # plot total model
    plt.plot(days_freqs, gpmodel.gp.kernel.get_psd(w_frequencies), ls="--", label="Total")
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

    if args.fit:
        # solution contains the information about the fit. .x is the best fit parameters
        solution = gpmodel.fit()

        print("Final log-likelihood (it is maximized so higher better): {0}".format(-solution.fun))
        print(solution)
        if not solution.success:
            raise Exception("The solver did not converge!\n %s" % solution.message)
        # bic information (equation 54 from Foreman et al 2017,
        # see also https://github.com/dfm/celerite/blob/ad3f471f06b18d233f3dab71bb1c20a316173cae/paper/figures/simulated/wrong-qpo.ipynb)
        BIC =  bic(-solution.fun, lc.n, gpmodel.k)
        AICC = aicc(-solution.fun, lc.n, gpmodel.k)
        print("BIC (the smaller the better): %.2f" % BIC)
        gpmodel.gp.set_parameter_vector(solution.x)
        best_params = gpmodel.gp.get_parameter_dict()
        print("Best-fit params\n -------")
        print(best_params)

        for term in gpmodel.gp.kernel.terms:
            if "kernel:log_omega0" in term.get_parameter_names():
                omega_par = "kernel:log_omega0"
                period = (two_pi / (np.exp(best_params[omega_par]) * days_to_seconds))
                print("Best fit period\n----------\n %.3f days\n -----" % period)
            elif "kernel:log_rho" in term.get_parameter_names():
                omega_par = "kernel:log_rho"
                period = np.exp(best_params[omega_par] * days_to_seconds)
                print("Best fit period\n----------\n %.3f days\n -----" % period)

        celerite_figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0, 'wspace': 0})
        # get model prediction
        ####gp.compute(time, yerr) --> check the tutorial compute is only called once
        pred_mean, pred_var = gpmodel.gp.predict(y, time_model, return_var=True)
        pred_std = np.sqrt(pred_var)

        color = "orange"
        ax1.errorbar(time / days_to_seconds, y, yerr=yerr, fmt=".k", capsize=0)
        if args.log:
            ax1.set_ylabel(" log(Count-rate (ct/s))")
        else:
            ax1.set_ylabel("Count-rate (ct/s)")
        ax1.plot(time_model / days_to_seconds, pred_mean, color="orange", zorder=100)
        # plot mean model if given

        ax1.plot(time_model / days_to_seconds, gpmodel.gp.mean.get_value(time_model), color="orange",
                 ls="--", zorder=100)
        ax1.fill_between(time_model / days_to_seconds, pred_mean + pred_std, pred_mean - pred_std, color="orange", alpha=0.3,
                         edgecolor="none", zorder=101)

        outputs = np.array([time_model / days_to_seconds, pred_mean, pred_std])
        np.savetxt("%s/model_fit.dat" % outdir, outputs.T, header="time(d)\tmodel\tstd", fmt="%.6f")

        # no need to pass time here as if this is omitted the coordinates will be assumed to be x from GP.compute() and an efficient method will be used to compute the prediction (https://celerite.readthedocs.io/en/stable/python/gp/)
        #gp.compute(time, yerr) no need as we already call compute above
        pred_mean, pred_var = gpmodel.gp.predict(y, return_var=True, return_cov=False)
        try:
            pvalue = standarized_residuals(y, pred_mean, np.sqrt(pred_var), "best_fit")
        except ValueError as e:
            print("Error when computing best_fit standarized residuals")
            print(e)
            pvalue = 0

        ax2.errorbar(time / days_to_seconds, (y - pred_mean) / yerr, yerr=1, fmt=".k", capsize=0)
        ax2.axhline(y=0, ls="--", color="#002FA7")
        ax2.set_ylabel("Residuals")
        ax2.set_xlabel("Time (days)")
        celerite_figure.savefig("%s/best_fit.png" % (outdir))
        plt.close(celerite_figure)

        # store best fit parameters and fit statistics
        header = "#file\tloglikelihood\tBIC\tAICc\tpvalue\tparameters\tdatapoints"
        outstring = "%s\t%.3f\t%.2f\t%.2f\t%.3f\t%d\t%d" % (os.path.basename(count_rate_file), -solution.fun,
                                                            BIC, AICC, pvalue, gpmodel.k, lc.n)
        for parname, best_par in zip(par_names, best_params):
            header += "\t%s" % parname
            outstring += '\t%.3f' % best_params[best_par]

        out_file = open("%s/parameters_fit.dat" % (outdir), "w+")
        out_file.write("%s\n%s" % (header, outstring))
        out_file.close()

        # plot PSD from the best fit
        psd_best_fit_figure, psd_best_fit_ax = plt.subplots()
        psd_best_fit_ax.set_xlabel("Frequency [days$^{-1}$]")
        psd_best_fit_ax.set_ylabel("Power")
        psd_best_fit_ax.set_yscale("log")
        psd = gpmodel.gp.kernel.get_psd(w_frequencies)
        psd_best_fit_ax.plot(days_freqs, psd, color=color, label="Total model")
        psd_best_fit_ax.axhline(psd_noise_level, ls="--", color="black", zorder=-10)
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

        for term in gpmodel.gp.kernel.terms:
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
        initial_samples = gpmodel.spread_walkers(nwalkers, solution.x, bounds, 10.0)

    max_steps = 300000
    print("Running chain for a maximum of %d samples with %d walkers until the chain has a length 100xtau using %d cores" % (max_steps, nwalkers, cores))
    print("Initial samples\n----------")
    print(initial_samples)
    every_samples = 500
    gpmodel.derive_posteriors(initial_samples, fit=False, walkers=nwalkers, max_steps=max_steps, cores=cores)

    acceptance_fraction = gpmodel.sampler.acceptance_fraction
    autocorr = gpmodel.autocorr
    print("Acceptance fraction: (%)")
    print(acceptance_fraction)
    print("Correlation parameters:")
    tau = gpmodel.tau
    print(tau)
    mean_tau = np.mean(tau)
    print("Mean correlation time:")
    print(mean_tau)

    if not gpmodel.converged:
        # tau will be very large here, so let's reduce the numbers
        thin = int(mean_tau / 4)
        discard = int(mean_tau) * 10 # avoid blowing up if discard is larger than the number of samples, this happens if the fit has not converged

    else:
        discard = int(mean_tau * 40)
        if discard > max_steps:
            discard = int(mean_tau * 10)
        thin = int(mean_tau / 2)

    fig = plt.figure()
    index = len(autocorr)
    n = every_samples * np.arange(1, index + 1)
    plt.plot(n, autocorr, "-o")
    plt.ylabel("Mean $\\tau$")
    plt.xlabel("Number of steps")
    plt.savefig("%s/autocorr.png" % outdir, dpi=100)
    plt.close(fig)

    # plot the entire chain
    chain = gpmodel.sampler.get_chain(flat=True)
    median_values = np.median(chain, axis=0)
    chain_fig, axes = plt.subplots(gpmodel.k, sharex=True, gridspec_kw={'hspace': 0.05, 'wspace': 0})
    if len(np.atleast_1d(axes))==1:
        axes = [axes]
    for param, parname, ax, median in zip(chain.T, par_names, axes, median_values):
        ax.plot(param, linestyle="None", marker="+", color="black")
        ax.set_ylabel(parname.replace("kernel:", "").replace("log_", ""))
        ax.axhline(y=median)
        ax.axvline(discard * nwalkers, ls="--", color="red")

    axes[-1].set_xlabel("Step Number")
    chain_fig.savefig("%s/chain_samples.png" % outdir, bbox_inches="tight",
                      dpi=100)
    plt.close(chain_fig)
    # calculate R stat
    rstat = gpmodel.get_rstat(discard)

    print("R-stat (values close to 1 indicate convergence)")
    print(rstat) # https://stackoverflow.com/questions/7140738/numpy-divide-along-axis

    final_samples = gpmodel.mcmc_samples
    loglikes = gpmodel.loglikehoods

    # save samples
    outputs = np.vstack((final_samples.T, loglikes))

    header_samples = "\t".join(par_names) + "\tloglikehood"

    np.savetxt("%s/samples.dat" % outdir, outputs.T, delimiter="\t", fmt="%.5f",
              header=header_samples)

    best_loglikehood = np.argmax(loglikes)

    # save max params and standarized residuals
    best_params = gpmodel.max_parameters

    gpmodel.gp.set_parameter_vector(best_params)
    ###gp.compute(time, yerr) --> check the tutorial compute is only called once
    # (No need to pass time as it'll be assumed the same datapoints as GP compute (https://celerite.readthedocs.io/en/stable/python/gp/)
    best_model, var = gpmodel.gp.predict(y, return_var=True) # omit time for faster computing time
    try:
        pvalue = standarized_residuals(y, best_model, np.sqrt(var), "max")
    except ValueError as e:
        print("Error when computing max standarized residuals")
        print(e)
        pvalue = 0

    np.savetxt("%s/model_max.dat" % outdir, np.array([time / days_to_seconds,
               best_model, np.sqrt(var)]).T,
               header="time(d)\tmodel\tstd",
               fmt="%.6f")

    # best parameter stats
    BIC = bic(gpmodel.max_loglikehood, lc.n, gpmodel.k)
    AICC = aicc(gpmodel.max_loglikehood, lc.n, gpmodel.k)

    header = "#file\tloglikelihood\tBIC\tAICc\tpvalue\tparameters\tdatapoints"
    outstring = "%s\t%.3f\t%.2f\t%.2f\t%.3f\t%d\t%d" % (os.path.basename(count_rate_file), gpmodel.max_loglikehood,
                                                        BIC, AICC, pvalue, gpmodel.k, lc.n)
    for parname, best_par in zip(par_names, best_params):
        header += "\t%s" % parname
        decimals = int(np.abs(np.floor(np.log10(np.abs(best_par))))) + 1
        outstring += '\t%.*f' % (decimals, best_par)

    out_file = open("%s/parameters_max.dat" % (outdir), "w+")
    out_file.write("%s\n%s" % (header, outstring))
    out_file.close()

    median_parameters = gpmodel.median_parameters
    distances = np.linalg.norm(final_samples - median_parameters, axis=1)
    closest_index = np.argmin(distances)
    median_log_likelihood = loglikes[closest_index]

    header = ""
    outstring = ''

    for i, parname in enumerate(par_names):
        q_16, q_50, q_84 = corner.quantile(final_samples[:,i], [0.16, 0.5, 0.84]) # your x is q_50
        header += "%s\t" % parname
        dx_down, dx_up = q_50-q_16, q_84-q_50
        decimals = int(np.abs(np.floor(np.log10(np.abs(dx_down))))) + 1
        #if "intercept" in parname:
        #    outstring += '%.*e$_{-%.*e}^{+%.*e}$\t' % (q_50, dx_down, dx_up)
        #else:
        outstring += '%.*f$_{-%.*f}^{+%.*f}$\t' % (decimals, q_50, decimals, dx_down, decimals, dx_up)
        if "Q" in parname:
            header += "%s\t" % parname.replace("log_Q", "Q")
            q_16, q_50, q_84 = np.exp(q_16), np.exp(q_50), np.exp(q_84)
            dx_down, dx_up = q_50-q_16, q_84-q_50
            outstring += '%.1f$^{+%.1f}_{-%.1f}$\t' % (q_50, dx_up, dx_down)

        # store also in days
        if "omega" in parname:
            header += "%s\t" % parname.replace("log_omega", "P")
            periods = two_pi / (np.exp(final_samples[:,i]) * days_to_seconds)
            q_16, q_50, q_84 = corner.quantile(periods, [0.16, 0.5, 0.84])
            dx_down, dx_up = q_50-q_16, q_84-q_50
            outstring += '%.2f$_{-%.2f}^{+%.2f}\t' % (q_50, dx_down, dx_up)

            header += "%s\t" % parname.replace("log_omega", "omega")
            freqs = np.exp(final_samples[:,i]) / 2 / np.pi * days_to_seconds # a factor 2pi was missing
            q_16, q_50, q_84 = corner.quantile(freqs, [0.16, 0.5, 0.84])
            dx_down, dx_up = q_50-q_16, q_84-q_50
            outstring += '%.4f$_{-%.4f}^{+%.4f}\t' % (q_50, dx_down, dx_up)

        # save contours for each param indicating the maximum
        fig = plt.figure()
        par_vals = final_samples[:,i]
        plt.scatter(par_vals, loglikes)
        plt.scatter(par_vals[best_loglikehood], loglikes[best_loglikehood],
                    label="%.2f, L = %.2f" % (par_vals[best_loglikehood], loglikes[best_loglikehood]))
        plt.legend()
        plt.xlabel("%s" % parname)
        plt.ylabel("$L$")
        plt.savefig("%s/%s.png" % (outdir, parname), dpi=100)
        plt.close(fig)


    header += "loglikehood"
    outstring += "%.3f" % median_log_likelihood
    out_file = open("%s/parameters_median.dat" % (outdir), "w+")
    out_file.write("%s\n%s" % (header, outstring))
    out_file.close()

    # Convert the frequency into period of the oscillator to period for plotting purposes
    #omegas = ("omega" in gp.get_parameter_names()).sum()
    Qs = np.count_nonzero(["log_Q" in param for param in gpmodel.gp.get_parameter_names()])
    omegas = np.count_nonzero(["omega" in param for param in gpmodel.gp.get_parameter_names()])

    # this just puts the period first
    inds = [par_names.index("%s" % c) for c in cols]
    samples = final_samples[:, inds]
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
    ranges = np.ones(gpmodel.k) * 0.95
    corner_fig = corner.corner(samples, labels=labels, title_fmt='.1f', range=ranges,
                               quantiles=[0.16, 0.5, 0.84], show_titles=True,
                               title_kwargs={"fontsize": 18}, max_n_ticks=3, labelpad=0.08,
                               levels=(1 - np.exp(-0.5), 1 - np.exp(-0.5 * 2 ** 2))) # plots 1 and 2 sigma levels
    corner_fig.savefig("%s/corner_fig.png" % outdir)
    plt.close(corner_fig)

    # finally plot final PSD and Model
    # MODEL
    model_figure, model_ax = plt.subplots()
    model_ax.set_xlabel("Time (days)")
    model_ax.set_ylabel("Count-rate ct/s")
    model_ax.errorbar(time / 3600 / 24, y, yerr=yerr, fmt=".k", capsize=0)
    model_figure.savefig("%s/lightcurve.png" % outdir)
    # PSD
    psd_figure, psd_ax = plt.subplots()
    psd_ax.set_xlabel("Frequency (days$^{-1}$)")
    psd_ax.set_ylabel("Power")
    psd_ax.set_yscale("log")
    psd_ax.axhline(psd_noise_level, ls="--", color="black")

    color = "black"
    # draw 1000 samples from the final distributions and create plots
    n_samples = 1500
    psds = np.empty((n_samples, len(frequencies)))
    psd_components = np.empty((len(gpmodel.gp.kernel.terms), n_samples, len(frequencies)))

    models = np.ones((n_samples, t_samples))
    means = np.ones((n_samples, t_samples))
    print("Generating %d samples for model and PSD plots" % n_samples)
    for index, sample in enumerate(final_samples[np.random.randint(len(samples), size=n_samples)]):
        gpmodel.gp.set_parameter_vector(sample)
        psd = gpmodel.gp.kernel.get_psd(w_frequencies)
        # omit time for faster computing time
        model = gpmodel.gp.predict(y, time_model, return_cov=False)
        model_ax.plot(time_model / days_to_seconds, model, color="orange", alpha=0.25)
        psd_ax.plot(frequencies * days_to_seconds, psd, color=color, alpha=0.25)
        means[index] = gpmodel.gp.mean.get_value(time_model)
        models[index] = model
        psds[index] = psd
        for term_i, term in enumerate(gpmodel.gp.kernel.terms):
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
    model_ax.set_ylabel("Count-rate ct/s")
    model_ax.errorbar(time / days_to_seconds, y, yerr=yerr, fmt=".k", capsize=0)
    m = np.nanpercentile(models, [16, 50, 84], axis=0)
    model_ax.plot(time_model / days_to_seconds, m[1], color="orange")
    model_ax.fill_between(time_model / days_to_seconds, m[0], m[2], alpha=0.3, color="orange")
    model_ax.plot(time_model / days_to_seconds, np.mean(means, axis=0), ls="--", color="orange")
    model_figure.savefig("%s/model_mcmc_median.png" % outdir, bbox_inches="tight")
    plt.close(model_figure)
    outputs = np.array([time_model / days_to_seconds, m[1], m[0], m[2]])
    np.savetxt("%s/model_median.dat" % outdir, outputs.T, header="time(d)\tmodel\tlower\tupper", fmt="%.6f")

    # PSD median
    p = np.percentile(psds, [16, 50, 84], axis=0)
    psd_output = np.array([frequencies, p[0], p[1], p[2], psd_noise_level * np.ones(len(frequencies))])
    np.savetxt("%s/psds.dat" % outdir, psd_output.T, delimiter="\t",
              fmt="%.8f", header="f\t16%\t50%\t84%\tnoise")
    # figure
    psd_median_figure, psd_median_ax = plt.subplots()
    psd_median_ax.set_xlabel("Frequency (days$^{-1}$)")
    psd_median_ax.set_ylabel("Power")
    psd_median_ax.set_yscale("log")

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
    colors = ["green", "red", "indigo", "brown"]
    for term_i, term in enumerate(gpmodel.gp.kernel.terms):
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
    psd_median_ax.axhline(psd_noise_level, ls="--", color="black")
    psd_median_figure.savefig("%s/psd_median_comps.png" % outdir, bbox_inches="tight")


    config_file_name = os.path.basename(args.config_file)
    copyfile(args.config_file, "%s/%s" % (outdir, config_file_name))

    with open("%s/samples.info" % outdir, "w+") as file:
        file.write("#samples\tdiscard\tthin\ttau\n")
        file.write("%d\t%d\t%d\t%.2f\n" % (gpmodel.sampler.iteration, discard, thin, mean_tau))

    python_command = "python %s %s --tmin %.2f --tmax %.2f -c %s -m %s -o %s" % (__file__, args.input_file[0],
                        args.tmin, args.tmax, args.config_file, args.meanmodel, args.outdir)
    if args.fit:
        python_command += " --fit"

    with open("%s/python_command.txt" % outdir, "w+") as file:
        file.write(python_command)
    print("Results stored to %s" % outdir)
