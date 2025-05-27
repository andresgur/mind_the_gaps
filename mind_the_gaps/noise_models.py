# @Author: Andrés Gúrpide <agurpide>
# @Date:   12-12-2024
# @Email:  a.gurpide-lasheras@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 12-12-2024
from typing import Tuple, Union
from numpy import floating
from numpy.typing import ArrayLike
from mind_the_gaps.stats import kraft_pdf
import numpy as np
from astropy.stats import poisson_conf_interval


class BaseNoise:
    def __init__(
            self,
            name: str
    ):
        self.name = name

    def add_noise(
            self,
            rates: ArrayLike,
    ):
        """Randomize the input rates and calculate uncertainties"""
        raise NotImplementedError("This method should be implemented by subclasses")


class PoissonNoise(BaseNoise):
    def __init__(
            self,
            exposures: ArrayLike,
            background_counts: Union[float, floating, None] = None,
            bkg_rate_err: Union[ArrayLike, None] = None
    ):
        super().__init__(name="Poisson")

        self.exposures = exposures
        if background_counts is None:
            self.background_counts = np.zeros(len(exposures), dtype=int)
        else:
            self.background_counts = background_counts
        if bkg_rate_err is None:
            self.bkg_rate_err = np.zeros(len(exposures))
        else:
            self.bkg_rate_err = bkg_rate_err


    def add_noise(
            self,
            rates: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Add Poisson noise and estimate uncertainties

        Parameters
        ----------
        rates
            Array of count rates
        exposures
            Exposure time at each bin (or a single value if same everywhere)

        Returns
        -------
        ArrayLike
            New Poissonian modified rates
        ArrayLike
            Uncertainties on the rates
        """
        total_counts = rates * self.exposures + self.background_counts

        total_counts_poiss = np.random.poisson(total_counts)

        net_counts = total_counts_poiss - self.background_counts #  frequentists

        dy = np.sqrt((np.sqrt(total_counts_poiss) / self.exposures)**2 + self.bkg_rate_err**2)

        return net_counts  / self.exposures, dy


class KraftNoise(PoissonNoise):
    def __init__(
            self,
            exposures: ArrayLike,
            background_counts: Union[float, None] = None,
            bkg_rate_err: Union[ArrayLike, None] = None,
            kraft_counts: float = 15
    ):
        """
        Add Poisson/Kraft noise to a given count rates rates based on a real lightcurve and estimate the uncertainty

        Parameters
        ----------
        bkg_counts: float or np.ndarray
            The number of background counts. None will assume 0s
        exposures: array
            In seconds
        bkg_rate_err: array
            Error on the background rate. None will assume 0s
        kraft_counts: float
            Threshold counts below which to use Kraft+91 posterior probability distribution
        """
        super().__init__(exposures, background_counts, bkg_rate_err)
        self.name = "Kraft"
        self.kraft_counts = kraft_counts

    def add_noise(
            self,
            rates: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Add Poisson/Kraft noise to a given count rates rates based on a real lightcurve and estimate the uncertainty

        This method was tested for speed against a single for loop (instead of three list comprehension) and it was found to be faster using the lists (around 10% reduction in time from the for loop approach))

        Parameters
        ----------
        rates
            The count rates per second

        Returns
        -------
        ArrayLike
            New Poissonian modified rates
        ArrayLike
            Uncertainties on the rates
        """
        net_rates, dy = super().add_noise(rates)
        total_counts = net_rates * self.exposures + self.background_counts

        #  frequentists bins
        net_counts = total_counts - self.background_counts #  frequentists
        upper_limits = net_rates / self.bkg_rate_err < 1 # frequentists upper limit

        # bayesian bins
        expression = total_counts < self.kraft_counts
        if np.any(expression):
            # calculate the medians
            pdf = kraft_pdf(a=0, b=35)
            net_counts[expression] = [pdf(counts, bkg_counts_).median() for counts, bkg_counts_ in zip(np.round(total_counts[expression]).astype(int), self.background_counts[expression])]
            net_rates[expression] = net_counts[expression] / self.exposures[expression]

            # uncertainties (round to nearest integer number of counts)
            lower_confs, upper_confs = poisson_conf_interval(total_counts[expression].astype(int), "kraft-burrows-nousek",
                                                background=self.background_counts[expression], confidence_level=0.68)
            dy[expression] = (upper_confs - lower_confs) / 2 / self.exposures[expression] # bayesian

            upper_limits[expression] = lower_confs==0 # bayesian upper limit

        return net_rates, dy
    
class GaussianNoise(BaseNoise):
    def __init__(
            self,
            exposures: ArrayLike,
            sigma_noise: float
    ):
        super().__init__(name="Gaussian")

        self.sigma_noise = sigma_noise

    def add_noise(
            self,
            rates: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Add Gaussian noise to a given count rates rates based on a real lightcurve and estimate the uncertainty

        Parameters
        ----------
        rates
            The count rates per second

        Returns
        -------
        ArrayLike
            New Poissonian modified rates
        ArrayLike
            Uncertainties on the rates
        """
        noisy_rates = rates + np.random.normal(scale=self.sigma_noise, size=len(rates))
        #dy = np.sqrt(rates * self._exposures) / self._exposures
        dy = self.sigma_noise * np.ones(len(rates))
        return noisy_rates, dy
