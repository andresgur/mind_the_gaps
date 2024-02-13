# @Author: Andrés Gúrpide <agurpide>
# @Date:   05-02-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-02-2022
import numpy as np
import astropy.units as u
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve

class SwiftLightcurve(GappyLightcurve):
    """
    A class to create a GappyLightcurve based on a Swift-XRT file
    """
    def __init__(self, swift_xrt_file="PCCURVE.qdp", minSNR=0, minSigma=0, minCts=0):
        """

        Parameters
        ----------
        swift_xrt_file:str,
            Path to a swift XRT file (QDP format)
        minSNR: float
            Minimum signal to noise ratio to filter out datapoints
        minSigma: float
            Minimum significance to filter out datapoints
        minCts: float
            Minimum counts to filter out datapoints

        """
        try:
            data = self.readPCCURVE(swift_xrt_file, minSNR=minSNR, minSigma=minSigma, minCts=minCts)
        except ValueError:
            data = self.readsimplePCCURVE(swift_xrt_file, minSigma=minSigma)

        time_column = data.dtype.names[0]
        rate_column = data.dtype.names[3]
        bkg_rate = data.dtype.names[7]
        bkg_rate_err = data.dtype.names[8]
        corr_factor_column = data.dtype.names[9]
        bkg_counts_column = data.dtype.names[11]
        exposure_column = data.dtype.names[12]

        if time_column == 'MJD':
            print("Time columnd in MJD")
            time = data[time_column] << u.d
        else:
            print("Time columnd in seconds")
            time = data[time_column] << u.s

        y = data[rate_column]
        yerr = (-data["%sneg" % rate_column] + data["%spos" % rate_column]) / 2

        corr_factor = data[corr_factor_column]
        # apply correction factor to exposures
        exposures = data[exposure_column] / corr_factor
        # NOTE: the background rate and err are divided by the corr factor so that bkg* (exposure / corrfactor)
        # when added to the source counts gives the correct bkg contribution
        super().__init__(time.to(u.s).value, y, yerr, exposures, data[bkg_rate] * corr_factor, data[bkg_rate_err] * corr_factor)


    def readPCCURVE(self, file="PCCURVE.qdp", minExposure=0, minSigma=0, minSNR=0, minCts=0):
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
        print("Trying to read data from %s" % file)
        try:
            # file with Obsids
            data = np.genfromtxt("%s" % file, names=True, delimiter="\t", skip_header=2, comments="!",
                                 dtype=("f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8, f8, f8, f8, f8, U30"))
            print("File with obsids")
        except ValueError as e:
            # Snapshot file without obsids
            data = np.genfromtxt("%s" % file, names=True, delimiter="\t", skip_header=2, comments="!",
                                 dtype=("f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, i8, f8, f8, f8, f8"))
            print("File with snapshots")
        filtered_data = data[(data["Exposure"] > minExposure) & (data["SNR"] > minSNR) & (data["Sigma"] > minSigma) & (data["CtsInSrc"] >= minCts)]
        filtered_obs = len(data) - len(filtered_data)
        print("Filtered %d datapoints by minSNR = %d and minSigma = %.2f, minCts = %d and minExp=%.1f" % (filtered_obs, minSNR, minSigma, minCts, minExposure))
        return filtered_data


    def readsimplePCCURVE(self, file="PCCURVE.qdp", minExposure=0, minSigma=0, minSNR=0):
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
