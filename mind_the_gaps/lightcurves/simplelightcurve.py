# @Author: Andrés Gúrpide <agurpide>
# @Date:   05-02-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-02-2022
import numpy as np
import astropy.units as u
import warnings
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve


class SimpleLightcurve(GappyLightcurve):
    """
    A class to create a GappyLightcurve based on a csv file containing only times, rates and errors.
    """
    def __init__(self, input_file, skip_header=0, delimiter=None):
        """

        Parameters
        ----------
        input_file:str,
            Path to a csv file containing time, rates and errors
        skip_header: int
            How many rows to skip
        """
        time, y, yerr, exposures, bkg_rate, bkg_err = self.readdata(input_file, skip_header, delimiter)

        super().__init__(time.to(u.s).value, y, yerr, exposures, bkg_rate, bkg_err)


    def readdata(self, input_file, skip_header, delimiter):
        """Read a csv file containing three columns."""
        data = np.genfromtxt("%s" % input_file, names=True,
                            skip_header=skip_header, delimiter=delimiter)
        time_column = data.dtype.names[0]
        rate_column = data.dtype.names[1]
        err_column = data.dtype.names[2]

        if time_column in ["mjd", "jd", "day"]:
            print("Time in days")
            time = data[time_column] << u.d
        else:
            time = data[time_column] << u.s

        if len(data.dtype) > 3:
            print("Reading exposures")
            exposures = data[data.dtype.names[3]]
            if len(data.dtype) >= 6:
                print("Reading background rates")
                bkg_rate = data[data.dtype.names[4]]
                bkg_err = data[data.dtype.names[5]]
            else:
                bkg_rate = np.zeros_like(time)
                bkg_err = np.zeros_like(time)
        else:
            warnings.warn("Lightcurve has no exposures!")
            exposures = np.zeros_like(time)
            bkg_rate = np.zeros_like(time)
            bkg_err = np.zeros_like(time)
        return time, data[rate_column], data[err_column], exposures, bkg_rate, bkg_err
