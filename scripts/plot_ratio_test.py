import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from scipy.stats import percentileofscore, lognorm, chi

ap = argparse.ArgumentParser(description='Plot LRT test')
ap.add_argument("-n", "--null_hyp", nargs="?", help="Null hypothesis file. Default fit_lcs_sims_m_Powerlaw/fit_results.dat", type=str, default=None)
ap.add_argument("-a", "--alt_hyp", nargs="?", help="Alternative hypothesis file. Default fit_lcs_sims_m_Powerlaw_Lorentzian/fit_results.dat", type=None)
ap.add_argument("-dn", "--data_null_hyp", nargs=1, help="Alternative hypothesis file for observed data", type=str)
ap.add_argument("-da", "--data_alt_hyp", nargs=1, help="Alternative hypothesis file for observed data", type=str)
args = ap.parse_args()

home = os.getenv("HOME")

if os.path.isfile('%s/.config/matplotlib/stylelib/paper.mplstyle' % home):
    plt.style.use('%s/.config/matplotlib/stylelib/paper.mplstyle' % home)
else:
    print("Style file not found!")

outfile = "fits_results.dat"

if args.null_hyp is not None:
    null_data = np.genfromtxt("%s" % args.null_hyp, names=True, dtype=("U34, f8, f8"), usecols=(0,1,2))
else:
    null_data = np.genfromtxt("fit_lcs_sims_m_Powerlaw/%s" % outfile, names=True, dtype=("U34, f8, f8"), usecols=(0,1,2))

if args.alt_hyp is not None:
    alt_data = np.genfromtxt("%s" % args.alt_hyp, names=True, dtype=("U34, f8, f8"), usecols=(0,1,2))
else:
    alt_data = np.genfromtxt("fit_lcs_sims_m_Lorentzian_Powerlaw/%s" % (outfile), names=True, dtype=("U34, f8, f8"), usecols=(0,1,2))

null_obs_data = np.genfromtxt("%s" % args.data_null_hyp[0], names=True, dtype=("U34, f8, f8"), usecols=(0,1,2))

alt_obs_data = np.genfromtxt("%s" % args.data_alt_hyp[0], names=True, dtype=("U34, f8, f8"), usecols=(0,1,2))
print(null_obs_data.dtype.names)

L_alt = alt_data["loglikelihood"]
L_hyp = null_data["loglikelihood"]

# see Protassov 2002 548
T = -2 * (L_hyp - L_alt)

negative = T<0

N_sims = len(T)

print("%d out of %d are negative" % (np.count_nonzero(T<0), N_sims))

inifinite = np.isinf(T)

print("%d out of %d are Inf:\n------------" % (np.count_nonzero(inifinite), N_sims))

if np.count_nonzero(np.isinf(T)) > 0:
    print(np.where(np.isinf(T))[0])

print("Five highest LRT:\n------------")

indices_of_highest_values = np.argsort(-T)[:5]
print(indices_of_highest_values)
print(np.sort(-T)[:5])

L_hyp = null_obs_data["loglikelihood"]
L_alt_obs = alt_obs_data["loglikelihood"]
T_obs = -2 * (L_hyp - L_alt_obs)

if T_obs < 0:
    print("Warning: observed LRT is negative!")

print("Observed LRT stat: %.3f" % T_obs)

plt.hist(T[~((negative) | (inifinite))], bins=100, edgecolor='black', density=True)
p = percentileofscore(T[~(negative)], T_obs)
#plt.axvline(T_obs, label="%.2f%%" % p, ls="--", color="black")
plt.xlabel("LRT statistic")
#plt.xlabel("$\Delta$L")
plt.ylabel("PDF")
plt.legend(frameon=False, fontsize=24)
#if N_sims < 4999
if N_sims < 900:
    outfile = "T_ratio_%d_deltaL.png" % N_sims
else:
    outfile = "T_ratio_%d.pdf" % N_sims

fit_dist = lognorm
#s, loc, scale
params = fit_dist.fit(T[~((negative) | (inifinite))]) #,
                 #(len(T), np.mean(T[~((negative) | (inifinite))]), np.std([~((negative) | (inifinite))])))
print("Distribution parameters:", params)
sortedT = np.sort(T[~((negative) | (inifinite))])

fap_dist = fit_dist(params[0], params[1], params[2])
#chist = chi(df=params[0], loc=params[1], scale=params[2])
plt.plot(sortedT, fap_dist.pdf(sortedT), label="Best-fit (Chi)")
plt.axvline(T_obs, label="%.2f%%" % (fap_dist.cdf(T_obs) * 100), ls="--", color="black")
plt.legend()
plt.savefig("%s" % outfile)


print("Numerical significance: %.2f" % (p))
print("Significance from fit", (fap_dist.cdf(T_obs) * 100))

outputs = np.array([null_data["sim"], alt_data["sim"], T])
outdatfile = "T_ratio.dat"

np.savetxt(outdatfile, outputs.T, delimiter="\t",
            fmt="%s", header="sim_null\tsim_alt\tLRT")
with open(outdatfile, "a") as file:
    file.write("# p-value\t%.2f%%\n" % p)


print("Stored to %s and %s" % (outfile, outdatfile))
