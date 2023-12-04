# @Author: Andrés Gúrpide <agurpide>
# @Date:   01-09-2023
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 01-09-2023

import numpy as np
import os
import argparse


ap = argparse.ArgumentParser(description='Generate observing strategy')
ap.add_argument('-N', "--npoints", nargs="?", help='Number of datapoints', default=1000, type=int)
ap.add_argument('-c', "--cadence", nargs="?", help='Cadence of the lightcurve', default=1, type=float)
ap.add_argument('-e', "--exposure", nargs="?", help='Exposure time of the observations (in seconds). Default 2000', default=2000, type=float)
ap.add_argument('-s', "--sigma", nargs="?", help='Sigma to vary the exposure times', default=0.2)
ap.add_argument('-r', "--root", nargs="?", help='End root output file', default=0, type=int)
ap.add_argument('-o', "--outdir", nargs="?", help='Output dir', default="strategy")
args = ap.parse_args()

sampling = np.random.normal(args.cadence, args.sigma, size=args.npoints)

times = np.cumsum(sampling)

print("Duration: %.1f" % (times[-1] - times[0]))

meanrate = 0.1 # typical of a ULX

rates = np.ones(len(times)) * meanrate
total_rate = rates / 0.9
errors = np.zeros(len(times))
bkg_counts = total_rate * args.exposure * 0.01
bkg_rate_err = np.sqrt(bkg_counts) / args.exposure
exposures = args.exposure * np.ones(len(rates))
outputs = np.array([times, rates, errors, exposures, bkg_counts, bkg_rate_err])

if not os.path.isdir(args.outdir):
    os.mkdir(args.outdir)

np.savetxt("%s/lc_data_N%d_c%d_%d.dat" % (args.outdir, args.npoints, args.cadence, args.root), outputs.T,
           fmt="%.6f", header="mjd\trate\terror\texposure\tbkg_counts\tbkg_counts_err", delimiter="\t")


print("Results stored to %s" % args.outdir)
