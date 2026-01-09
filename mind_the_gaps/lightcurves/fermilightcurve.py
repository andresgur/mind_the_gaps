# @Author: Andrés Gúrpide <agurpide>
# @Date:   05-02-2022
# @Email:  agl1f22@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 08-01-2026
import numpy as np
import astropy.units as u
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve


class FermiLightcurve(GappyLightcurve):
    """
    A class to create a GappyLightcurve based on a csv file containing only times, rates and errors.
    """

    def __init__(self, input_file):
        """

        Parameters
        ----------
        input_file:str,
            Path to a csv file containing time, rates and errors
        """
        time, y, yerr = self.readdata(input_file)

        super().__init__(time.to(u.s).value, y, yerr)

    def readdata(self, input_file):

        data = np.genfromtxt("%s" % input_file, names=True, delimiter=",")
        time_column = data.dtype.names[0]
        rate_column = data.dtype.names[1]

        if "MJD" in time_column.upper():
            time = data[time_column] << u.d
        else:
            time = data[time_column] << u.s

        y = data[rate_column]
        yerr = (
            np.abs(data["%s_err_neg" % rate_column]) + data["%s_err_pos" % rate_column]
        ) / 2
        return time, y, yerr
