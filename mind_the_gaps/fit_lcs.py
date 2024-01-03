# @Author: Andrés Gúrpide <agurpide>
# @Date:   01-09-2020
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 15-02-2023
import numpy as np
import os
import argparse
from scipy.optimize import minimize
from shutil import copyfile
import celerite
import emcee
import astropy.units as u
from multiprocessing import Pool
import warnings
import matplotlib.pyplot as plt
import time
from celerite_script import read_config_file, neg_log_like

os.environ["OMP_NUM_THREADS"] = "1" # https://emcee.readthedocs.io/en/stable/tutorials/parallel/


def fit_lightcurve(input_file):

    def log_probability(params):
        """https://celerite.readthedocs.io/en/stable/tutorials/modeling/"""
        gp.set_parameter_vector(params)

        lp = gp.log_prior() # this is 0 or -inf
        if not np.isfinite(lp):
            return -np.inf
        return lp + gp.log_likelihood(y)

    data = np.genfromtxt(input_file, names=True, usecols=(0, 1 ,2))
    y = data["rate"]
    times = data["t"]
    yerr = data["error"]
    gp = celerite.GP(kernel, mean=np.mean(y))
    gp.compute(times, yerr + 1.123e-12)  # You always need to call compute once.
    print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))
    # solution contains the information about the fit. .x is the best fit parameters
    solution = minimize(neg_log_like, initial_params, method="L-BFGS-B",
                        bounds=gp.get_parameter_bounds(), args=(y, gp))
    gp.set_parameter_vector(solution.x)
    #print(solution)
    print("Best-fit log-likelihood: {0} \n".format(-solution.fun)) # here is neg_log_like

    initial_mcmc = np.empty((nwalkers, ndim))

    if solution.success:
        #initial_samples = solution.x + 1e-2 * np.random.randn(nwalkers, ndim)
        #reshaped_params = np.resize(solution.x, (nwalkers, ndim))
        #initial_samples = solution.x + 1e-1 * np.random.randn(nwalkers, ndim)
        # Gaussian centered around the best params and 1+-sigma of 10%

        for i in range(nwalkers):
            accepted = False

            while not accepted:
                # Generate random values centered around the best-fit parameters (normally 10% works fine, but increase a bit just in case to make sure walkers
                #are independent)
                perturbed_params = np.random.normal(solution.x, np.abs(solution.x) / 10.0)

                # Check if the perturbed parameters are within the bounds
                if np.all(np.logical_and(bounds[:, 0] <= perturbed_params, perturbed_params <= bounds[:, 1])):
                    initial_mcmc[i] = perturbed_params
                    accepted = True
    else:
        warnings.warn("The solver did not converge!\n %s" % solution.message)
        initial_mcmc = initial_samples

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    try:
        sampler.run_mcmc(initial_mcmc, max_n, progress=False)
    except ValueError:
        warnings.warn("Large conditional problem, assuming uniform priors")
        sampler.run_mcmc(initial_samples, max_n, progress=False)

    log_probs = sampler.get_log_prob(flat=True)
    best_loglikehood_index = np.argmax(log_probs)
    best_mcmc_pars = sampler.get_chain(flat=True)[best_loglikehood_index]

    best_log = log_probs[best_loglikehood_index]

    print("MCMC chains maximum log-likelihood:\n------------------------------------------- \n {0}\n".format(best_log))

    gp.set_parameter_vector(best_mcmc_pars)

    best_params = gp.get_parameter_dict()

    for key in best_params.keys():
        best_pars[key].append(best_params[key])

    print("Best-fit params\n -------")
    print(best_params)

    return best_params, best_log


ap = argparse.ArgumentParser(description='Perform fit of an input lightcurve (no MCMC) (see Foreman-Mackey et al 2017. 10.3847/1538-3881/aa9332)')
ap.add_argument("-c", "--config_file", nargs='?', help="Config file with initial parameter constraints", type=str, default="/home/andresgur/scripts/swift_scripts/celerite_config/parameters.config")
ap.add_argument("-o", "--outdir", nargs='?', help="Output dir name", type=str, default="fit_lcs")
ap.add_argument("-n", "--nmax", nargs='?', help="Maximum number of samples. Default 2000", type=int, default=2000)
ap.add_argument("--cores", default=12, help="Number of cores. Default 12", nargs="?", type=int)
ap.add_argument("input_files", nargs='+', help="Input lightcurves", type=str)
args = ap.parse_args()

if "SLURM_CPUS_PER_TASK" in os.environ:
    cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    warnings.warn("The numbe of cores is being reset to SLURM_CPUS_PER_TASK = %d " % cores )
else:
    cores = args.cores

nwalkers = 32 if args.cores < 64 else args.cores * 2 # https://stackoverflow.com/questions/69234421/multiprocessing-the-python-module-emcee-but-not-all-available-cores-on-the-ma

max_n = args.nmax

days_to_seconds = 24 * 3600

python_command = "python %s -c %s -o %s" % (__file__, args.config_file,
                                            args.outdir)
input_files = args.input_files

prefix = args.outdir if args.outdir=="fit_lcs" else "fit_lcs_%s" % (args.outdir)

kernel, initial_samples, labels, cols, outmodels = read_config_file(args.config_file, walkers=nwalkers)

outmodelsstr = "_" + "_".join(outmodels)

outdir = "%s_m%s" % (prefix, outmodelsstr)

if not os.path.isdir(outdir):
    os.mkdir(outdir)

initial_samples = initial_samples.T
# dummy gp to get values
dummy_gp = celerite.GP(kernel, mean=0)
print("parameter_dict:\n{0}\n".format(dummy_gp.get_parameter_dict()))
print("parameter_names:\n{0}\n".format(dummy_gp.get_parameter_names()))
print("parameter_vector:\n{0}\n".format(dummy_gp.get_parameter_vector()))
print("parameter_bounds:\n{0}\n".format(dummy_gp.get_parameter_bounds()))
initial_params = dummy_gp.get_parameter_vector()
bounds = np.array(dummy_gp.get_parameter_bounds())
par_names = list(dummy_gp.get_parameter_names())


best_pars = {key: [ ] for key in par_names} #  dictionary to store all best fit pars

ndim = len(initial_params)

start = time.time()

with Pool(processes=cores, initializer=np.random.seed) as pool:
    results = pool.map(fit_lightcurve, input_files)

best_params = [result[0] for result in results]

# unnecessary but let's keep it like this for now
for key in best_pars.keys():
    for pars in best_params:
        best_pars[key].append(pars[key])

log_likehoods = [result[1] for result in results]

end = time.time()

print("Time taken: %.2fs" % (end-start))

with open("%s/python_command.txt" % outdir, "w+") as file:
    file.write(python_command)

# get Y
data = np.genfromtxt(input_files[0], names=True, usecols=(0, 1 ,2))
y = data["rate"]

# bic information (equation 54 from Foreman et al 2017, see also https://github.com/dfm/celerite/blob/ad3f471f06b18d233f3dab71bb1c20a316173cae/paper/figures/simulated/wrong-qpo.ipynb)
bics = -2 * np.array(log_likehoods) + len(dummy_gp.get_parameter_dict()) * np.log(len(y)) # we have now defined L as positive!

outputs = np.vstack((input_files, log_likehoods, bics, list(best_pars.values())))

header = "sim\tloglikehood\tbic\t" + "\t".join(par_names)

np.savetxt("%s/fits_results.dat" % outdir,
           outputs.T, header=header, fmt="%s") # store all strings as it is hard to do otherwise


omega_par= [name for name in par_names if "omega" in name]

# there is some omega parameter in the list store the periods too
if not omega_par:

    for i, omega in omega_par:

        periods = [(2 * np.pi / (np.exp(best_pars[omega]) * days_to_seconds)) for best_pars in best_params]

        np.savetxt("%s/periods_%d.dat" % (outdir, i), periods, header="P", fmt="%.3f")
else:
    periods = None


config_file_name = os.path.basename(args.config_file)
copyfile(args.config_file, "%s/%s" % (outdir, config_file_name))


if len(args.input_files) >1:

    plt.figure()
    plt.hist(bics)
    plt.xlabel("BIC")
    plt.savefig("%s/bics.png" % outdir)

    plt.figure()
    plt.hist(log_likehoods)
    plt.xlabel("$L$")
    plt.savefig("%s/likehoods.png"% outdir)

    for par in best_pars.keys():
        plt.figure()
        plt.hist(best_pars[par])
        plt.xlabel("%s" %par)
        plt.savefig("%s/%s.png"% (outdir, par))

    if periods is not None:
        plt.figure()
        plt.hist(periods)
        plt.xlabel("$P$ (days)")
        plt.savefig("%s/periods.png"% outdir)

print("Results stored to %s" % outdir)
