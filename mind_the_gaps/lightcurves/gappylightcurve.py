# @Author: Andrés Gúrpide <agurpide>
# @Date:   05-02-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-02-2022
import numpy as np
from random import sample
##from mind_the_gaps import Simulator

class GappyLightcurve:
    """
    A class to store parameters of a irregularly-sampled lightcurve
    """
    def __init__(self, times, y, dy, exposure=None, bkg_rate=None, bkg_rate_err=None):
        """

        Parameters
        ----------
        times:array_like,
            Timestamps of the lightcurve (i.e. times at the "center" of the sampling). Always in seconds
        y: array_like
            Observed flux or count rate
        dy: array-like
            1 sigma uncertainty on y
        exposures:
            Exposure time of each datapoint in seconds
        bkg_rate: array-like
            Associated background rate (or flux) with each datapoint
        bkg_rate_err:array-like
            uncertainty on background rate

        """
        self._times = times
        self._y = y
        self._dy = dy
        self._exposures = exposure if exposure is not None else np.zeros(len(times))
        self._bkg_rate = bkg_rate if bkg_rate is not None else np.zeros(len(times))
        self._bkg_rate_err = bkg_rate_err if bkg_rate_err is not None else np.zeros(len(times))

    @property
    def times(self):
        """
        Timestamps of the lightcurve.

        Returns
        -------
        array-like
            Timestamps of the lightcurve.
        """
        return self._times

    @property
    def n(self):
        """
        Number of datapoints

        Returns
        -------
        int
            Number of datapoints
        """
        return len(self._times)

    @property
    def y(self):
        """
        Observed flux or count rate.

        Returns
        -------
        array-like
            Observed flux or count rate.
        """
        return self._y

    @property
    def dy(self):
        """
        1 sigma uncertainty on y.

        Returns
        -------
        array-like
            1 sigma uncertainty on y.
        """
        return self._dy

    @property
    def exposures(self):
        """
        Exposure time of each datapoint.

        Returns
        -------
        array-like
            Exposure time of each datapoint.
        """
        return self._exposures

    @property
    def bkg_rate(self):
        """
        Associated background rate (or flux) with each datapoint.

        Returns
        -------
        array-like
            Associated background rate (or flux) with each datapoint.
        """
        return self._bkg_rate

    @property
    def bkg_rate_err(self):
        """
        Uncertainty on background rate.

        Returns
        -------
        array-like
            Uncertainty on background rate.
        """
        return self._bkg_rate_err

    @property
    def duration(self):
        """
        Duration of the lightcurve (times[-1] - times[0]).

        Returns
        -------
        float
            Duration of the lightcurve.
        """
        return self._times[-1] - self._times[0]

    @property
    def mean(self):
        """Mean count rate"""
        return np.mean(self._y)


    def truncate(self, tmin=-np.inf, tmax=np.inf):
        """
        Create a new GappyLightcurve instance by cutting the data between tmin and tmax.

        Parameters
        ----------
        tmin : float
            Minimum timestamp for the cut.
        tmax : float
            Maximum timestamp for the cut.

        Returns
        -------
        GappyLightcurve
            New instance representing the cut data.
        """
        if tmin >= tmax:
            raise ValueError("Minimum truncation time (%.2es) is greater than or equal to maximum truncation time (%.3es)!" %(tmin, tmax))
        if tmax < self._times[0]:
            raise ValueError("Maximum truncation time (%.2f) is lower than initial lightcurve time (%.2f)" % (tmax, self._times[0]))
        mask = (self._times >= tmin) & (self._times <= tmax)

        return GappyLightcurve(
            self._times[mask],
            self._y[mask],
            self._dy[mask],
            self._exposures[mask],
            self._bkg_rate[mask],
            self._bkg_rate_err[mask]
        )

    def rand_remove(self, points_remove):
        """Randomly remove a given number of points from the lightcurve"""
        if points_remove > self.n:
            return ValueError("Number of points to remove (%d) is greater than number of lightcurve datapoints (%d)"  % (points_remove, self.n))
        ints = sample(range(len(self._times)), points_remove)
        mask = np.ones(len(self._times), dtype=bool)
        mask[ints] = False
        return GappyLightcurve(
            self._times[mask],
            self._y[mask],
            self._dy[mask],
            self._exposures[mask],
            self._bkg_rate[mask],
            self._bkg_rate_err[mask]
        )

    def to_csv(self, outname: str):
        """Save lightcurve properties to csv file"""
        outputs = np.array([self._times, self._y, self._dy, self._exposures, self._bkg_rate, self._bkg_rate_err])
        np.savetxt(outname, outputs.T, fmt="%.6f", header="t\trate\terror\texposure\tbkg_rate\tbkg_rate_err")


    def get_simulator(self, psd_model, pdf, mean=None):
        """Creates an instance of mind_the_gaps.Simulator based on the lightcurve
            properties (timestamps, exposures, etc)

        Parameters
        ----------
        psd_model: astropy.modeling.Model,
            The model for the PSD
        pdf: str,
            The probability distribution (Gaussian, Lognormal or Uniform)
        """
        if pdf.lower() not in ["gaussian", "lognormal", "uniform"]:
            raise ValueError("%s not implemented! Currently implemented: Gaussian, Uniform or Lognormal")

        if mean is None:

            mean = self._mean

        ##return Simulator(psd_model, pdf, self_.times, self_.bkg_rate, self_.bkg_rate_err, mean)