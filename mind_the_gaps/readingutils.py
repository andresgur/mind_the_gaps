# @Author: Andrés Gúrpide <agurpide>
# @Date:   02-09-2020
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 01-12-2021
import numpy as np
from astropy.time import Time
import os
import astropy.units as u
import warnings


def read_data(input_file, tmin=0, tmax=np.inf):
    """Modify this function to read your data and filter it by time

    Time and exposure is always returned in seconds
    """
    try:
        data = readPCCURVE("%s" % input_file, minSNR=0, minSigma=0, minCts=0)
    except ValueError:
        data = readsimplePCCURVE("%s" % input_file, minSigma=0)

    time_column = data.dtype.names[0]
    rate_column = data.dtype.names[3]
    bkg_rate_err = data.dtype.names[8]
    corr_factor_column = data.dtype.names[9]
    bkg_counts_column = data.dtype.names[11]
    exposure_column = data.dtype.names[12]
    filtered_data = data[np.where((data["%s" % time_column] >= tmin) & (data["%s" % time_column] <=tmax))]
    if time_column == 'MJD':
        print("Time columnd in MJD")
        time = filtered_data[time_column] << u.d
    else:
        print("Time columnd in seconds")
        time = filtered_data[time_column] << u.s

    y = filtered_data[rate_column]
    yerr = (-filtered_data["%sneg" % rate_column] + filtered_data["%spos" % rate_column]) / 2
    exposures = filtered_data[exposure_column]
    bkg_rate_err = filtered_data[bkg_rate_err]
    bkg_counts = filtered_data[bkg_counts_column]
    corr_factor = filtered_data[corr_factor_column]
    return time, y, yerr, exposures / corr_factor, bkg_counts, bkg_rate_err * corr_factor


def read_data2(input_file, tmin=0, tmax=np.inf):
    """Modify this function to read your data and filter it by time

    Time and exposure is always returned in seconds
    """
    data = np.genfromtxt("%s" % input_file, names=True, delimiter="\t")
    time_column = data.dtype.names[0]
    rate_column = data.dtype.names[1]
    err_column = data.dtype.names[2]
    exposure_column = data.dtype.names[3]


    filtered_data = data[((data["%s" % time_column] >= tmin) & (data["%s" % time_column] <=tmax))]
    if time_column.lower() in ["mjd", "jd", "day"]:
        print("Time column in days")
        # convert data to seconds
        time = filtered_data[time_column] << u.d
    else:
        print("Time column in seconds")
        time = filtered_data[time_column] << u.s

    y = filtered_data[rate_column]
    yerr = filtered_data[err_column]
    exposures = filtered_data[exposure_column]

    if len(data.dtype.names) > 4:
        bkg_rate_column = data.dtype.names[4]
        bkg_rate_err_column = data.dtype.names[5]
        bkg_counts = filtered_data[bkg_rate_column] * exposures
        bkg_rate_err = filtered_data[bkg_rate_err_column]
    else:
        warnings.warn("No background count-rates found! Assuming 0s everywhere")
        bkg_counts = np.zeros(len(filtered_data))
        bkg_rate_err = np.zeros(len(filtered_data))

    return time, y, yerr, exposures, bkg_counts, bkg_rate_err


def read_zero_point(file="t0.date"):
    """Read zero point value from the input file in MJD

        If the argument `file` isn't passed in, the default is t0.date file from the Swift pipeline is used.

        Parameters
        ----------
        file : str, optional
            The file where to look the zero point (default is t0.date)
        """
    f = open(file)
    lines = f.readlines()
    f.close()
    zero_point = float(lines[2])
    # first of january of 2001 in mjd https://swift.gsfc.nasa.gov/analysis/suppl_uguide/time_guide.html'+ zero_poin to days
    # from this file we have seconds since swift reference time, starting date, modified julian days and julian days (http://www.csgnetwork.com/julianmodifdateconv.html)
    # swift_ref_time = Time(51910, format="mjd", scale="tt")
    swift_zero_point = Time(zero_point, format="mjd", scale="tt")
    return swift_zero_point


def readPCCURVE(file="PCCURVE.qdp", minExposure=0, minSigma=0, minSNR=0, minCts=0):
    """Read PCCURVE from Swift data pipeline.

        Parameters
        ----------
        file: str, optional
            The file to be read. Default is PCCURVE.qdp
        minExposure : float, optional
            Minimum exposure to consider
        minSigma : float, optional
            Minimum Sigma to consider.
        minSNR: float, optional
            Minimum SNR to consider.
        minCts: float, optional
            Minimum number of source counts to consider
        """
    print("Reading data from %s" % file)
    try:
        # file with Obsids
        data = np.genfromtxt("%s" % file, names=True, delimiter="\t", skip_header=2, comments="!",
                             dtype=("f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8, f8, f8, f8, f8, U30"))
    except ValueError:
        # Snapshot file without obsids
        data = np.genfromtxt("%s" % file, names=True, delimiter="\t", skip_header=2, comments="!",
                             dtype=("f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8, f8, f8, f8, f8"))
    filtered_data = data[(data["Exposure"] > minExposure) & (data["SNR"] > minSNR) & (data["Sigma"] > minSigma) & (data["CtsInSrc"] >= minCts)]
    filtered_obs = len(data) - len(filtered_data)
    print("Filtered %d datapoints by minSNR = %d and minSigma = %.2f, minCts = %d and minExp=%.1f" % (filtered_obs, minSNR, minSigma, minCts, minExposure))
    return filtered_data


def readPCUL(file="PCUL.qdp", minExposure=0):
    """Read PCUL from Swift data pipeline.

        Parameters
        ----------
        file: str, optional
            The file to be read. Default is PCUL.qdp
        minExposure : float, optional
            Minimum exposure to consider in seconds
        minSigma : float, optional
            Minimum Sigma to consider.
        minSNR: float, optional
            Minimum SNR to consider.
        """
    print("Reading upper limits from %s" % file)
    try:
        data = np.genfromtxt("%s" % file, names=True, delimiter="\t", skip_header=2, comments="!", dtype=("f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8, f8, f8, f8, f8, U30"))
    except ValueError:
        data = np.genfromtxt("%s" % file, names=True, delimiter="\t", skip_header=2, comments="!", dtype=("f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8, f8, f8, f8, f8"))
    # fix when there is only one upper limit
    if len(np.atleast_1d(data)) == 1:
        data = np.array([data])
    print('Found %d upper limits' % len(data))
    filtered_data = data[(data["Exposure"] > minExposure)]
    filtered_obs = len(data) - len(filtered_data)
    print("Filtered %d upper limits by minExp = %.2f (s)" % (filtered_obs, minExposure))

    return filtered_data


def readsimplePCCURVE(file="PCCURVE.qdp", minExposure=0, minSigma=0, minSNR=0):
    """Read simple Swift-XRT data from proposals.

        Parameters
        ----------
        file: str, optional
            The file to be read. Default is PCCURVE.qdp
        minExposure : float, optional
            Minimum exposure to consider
        """
    print("Reading %s data" % file)
    data = np.genfromtxt("%s" % file, names=True, delimiter="\t", comments="!", dtype=("f8, f8, f8, f8, f8, f8"), deletechars="~", usecols=(0, 1, 2, 3, 4, 5))
    filtered_data = data[(((data["T_+ve"] - data["T_-ve"]) > minExposure) & (data["Sigma"] > minSigma))]
    filtered_obs = len(data) - len(filtered_data)
    print("Filtered %d datapoints by minSNR = %d and minSigma = %.2f" % (filtered_obs, minSNR, minSigma))
    return filtered_data


def readPCHR(file="PCHR.qdp", minSoftSig=0, minHardSig=0, reject_errors=True, minExposure=0):
    """Read PCHR from Swift data pipeline.

        Parameters
        ----------
        file: str, optional
            The file to be read. Default is PCHR.qdp
        minSoftSig : float, optional
            Minimum soft signal to filer. Default is 0.
        minHardSig : float, optional
            Minimum hard signal to filer. Default is 0.
        reject_errors : boolean, optional
            Whether to reject data points with errors higher than the data point value. Default is True.
        """
    print("Reading %s data" % file)
    try:
        data = np.genfromtxt("%s" % file, names=True, delimiter="\t", skip_header=2, comments="!", dtype=("f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, U30"))
    except ValueError:
        data = np.genfromtxt("%s" % file, names=True, delimiter="\t", skip_header=2, comments="!", dtype=("f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8"))
    if reject_errors is True:
        filtered_data = data[(~np.isnan(data["HR"])) & (data["HR"] > 0) & (data["SoftSig"] > minSoftSig) & (data["HardSig"] > minHardSig) & (data["HRerr"] < data["HR"]) & (data["Exposure"] > minExposure)]
    else:
        filtered_data = data[(~np.isnan(data["HR"])) & (data["HR"] > 0) & (data["SoftSig"] > minSoftSig) & (data["HardSig"] > minHardSig) & (data["Exposure"] > minExposure)]
    print("Filtered %d datapoints" % (len(data) - len(filtered_data)))
    return filtered_data



def readPC_catalog(file="PC_catalog.qdp", minExposure=0):
    """Read PC from the Swift LSXP catalog (https://www.swift.ac.uk/LSXPS/LSXPS%20J133007.8%2B471105).

    Parameters
    ----------
    file: str, optional
        The file to be read. Default is PC_catalog.qdp
    minExposure : float, optional
        Whether to reject data points when the exposure is below a certain threshold.
    """
    print("Reading %s data" % file)
    data = np.genfromtxt("%s" % file, names=["Time", "T_ve", "T_ve_1", "Rate", "Ratepos", "Rateneg"],
                         delimiter="\t", skip_header=2, comments="!", dtype=("f8, f8, f8, f8, f8, f8"))
    exposure = data["T_ve"] - data["T_ve_1"]
    filtered_data = data[exposure > minExposure]
    print("Filtered %d datapoints" % (len(data) - len(filtered_data)))
    return filtered_data


def read_best_fit(file):
    """Read best fit file from the LS periodogram.

        Parameters
        ----------
        file: str, optional
            Path to the file to be read. Default is swift_countrates.dat
        """
    print("Reading %s data" % file)
    return np.genfromtxt("%s" % file, names=True, delimiter="\t", comments="#",
                         dtype=("f8, f8"))


def read_standard_lightcurve(input_file):
    """Read a column separated file containing the lightcurve timestamps, measurements, etc
    The columns can be given any name but need to be in the following order: time, y, yerr, exposures, bkg_counts and err on the background rate (all 0s if none)

    input_file:str
        The input file to be read
    """
    lightcurve =  np.genfromtxt(input_file, names=True)
    timestamps = lightcurve["t"] << u.s
    rates = lightcurve["rate"]
    errors = lightcurve["error"]
    exposures = lightcurve["exposure"]
    bkg_counts = lightcurve["bkg_counts"]
    bkg_rate_err = lightcurve["bkg_rate_err"]
    return timestamps, rates, errors, exposures, bkg_counts, bkg_rate_err
