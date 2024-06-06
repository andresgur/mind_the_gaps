# @Author: Andrés Gúrpide <agurpide>
# @Date:   01-09-2023
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 26-04-2024

import numpy as np
import os
import argparse

ap = argparse.ArgumentParser(description='Generate an observing strategy')
ap.add_argument('-N', "--npoints", nargs="?", help='Number of datapoints (prior to any removing)', default=1000, type=int)
ap.add_argument('-c', "--cadence", nargs="?", help='Cadence of the lightcurve', default=1, type=float)
ap.add_argument('-e', "--exposure", nargs="?", help='Exposure time of the observations (in seconds). Default 2000',
                default=2000, type=float)
ap.add_argument('-s', "--sigma", nargs="?", help='Sigma to vary the cadence times. Default 0.2', default=0.2)
ap.add_argument('-r', "--remove", nargs="?", help='Remove one point every N points to add gaps. Default 0',
                default=0, type=int)
ap.add_argument('-g', "--gaps", nargs="*", help='Add random gaps (more than one possible) of length X.XX X.XX days somewhere in the lightcurve',
                default=0, type=float)
ap.add_argument('-o', "--outdir", nargs="?", help='Output dir', default="strategy")
args = ap.parse_args()

sampling = np.random.normal(args.cadence, args.sigma, size=args.npoints)

times = np.cumsum(sampling)



if args.remove > 0:
    print("Removing each datapoint every %d datapoints" % args.remove)
    idx_delete = np.arange(args.remove, times.size, args.remove)
    times = np.delete(times, idx_delete)

if args.gaps is not None:
    for gap in args.gaps:
        start_time = np.random.uniform(times[0], times[-2] - gap) # always keep the last datapoint
        print("Adding %d day gap at %.1f days"  % (gap, start_time))
        mask = (times < start_time) | (times > (start_time + gap))
        times = times[mask]

    gap_string = "_".join([str(gap) for gap in args.gaps])

Npoints  = len(times)

print("Number of datapoints: %d" % Npoints)
print("Duration: %.1f" % (times[-1] - times[0]))

meanrate = 35# 0.1 typical of a ULX

rates = np.ones(Npoints) * meanrate
total_rate = rates / 0.99
errors = np.zeros(Npoints)
bkg_counts = total_rate * args.exposure * 0.01
bkg_rate_err = np.sqrt(bkg_counts) / args.exposure
exposures = args.exposure * np.ones(Npoints)
outputs = np.array([times, rates, errors, exposures, bkg_counts, bkg_rate_err])

if not os.path.isdir(args.outdir):
    os.mkdir(args.outdir)

np.savetxt("%s/lc_data_N%d_c%d_r%d_g%s.dat" % (args.outdir, args.npoints, args.cadence, args.remove, gap_string), outputs.T,
           fmt="%.6f", header="mjd\trate\terror\texposure\tbkg_counts\tbkg_counts_err", delimiter="\t")


print("Results stored to %s" % args.outdir)
