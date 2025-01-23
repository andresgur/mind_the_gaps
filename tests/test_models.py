# @Author: Andrés Gúrpide <agurpide>
# @Date:   28-03-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-03-2022
import unittest

import numpy as np
from celerite.terms import Matern32Term, SHOTerm

from mind_the_gaps.models import DampedRandomWalk as DRW_celerite
from mind_the_gaps.models import Lorentzian as celerite_Lorentzian
from mind_the_gaps.models.psd_models import (
    SHO,
    BendingPowerlaw,
    Lorentzian,
    Matern,
    Matern32,
    Matern52,
)


class TestSimulator(unittest.TestCase):

    def test_DRW(self):
        Q = 1 / 2
        S_0 = 10
        w_0 = 5

        cel_BPL = DRW_celerite(log_S0=np.log(S_0), log_omega0=np.log(w_0))
        astropy_BPL = BendingPowerlaw(S0=S_0, omega0=w_0, Q=Q)
        frequencies = np.arange(1, 1000)
        np.testing.assert_array_almost_equal(
            astropy_BPL(frequencies), cel_BPL.get_psd(frequencies)
        )

    def test_SHO(self):
        Qs = [10, 1, 1 / np.sqrt(2), 0.1]
        S_0 = 10
        w_0 = 5
        frequencies = np.arange(1, 1000)
        for Q in Qs:
            cel_SHO = SHOTerm(
                log_S0=np.log(S_0), log_Q=np.log(Q), log_omega0=np.log(w_0)
            )
            astropy_SHO = SHO(S0=S_0, omega0=w_0, Q=Q)
            np.testing.assert_array_almost_equal(
                astropy_SHO(frequencies), cel_SHO.get_psd(frequencies)
            )

    def test_materns(self):
        sigma = 10
        rhos = [1, 10, 20]
        frequencies = np.arange(1, 1000)
        for rho in rhos:
            cel_mattern = Matern32Term(
                log_sigma=np.log(sigma), log_rho=np.log(rho), eps=1e-15
            )
            astropy_mattern = Matern32(sigma=sigma, rho=rho)
            np.testing.assert_array_almost_equal(
                astropy_mattern(frequencies), cel_mattern.get_psd(frequencies)
            )

            matern52 = Matern52(sigma=sigma, rho=rho)
            matern522 = Matern(frequencies, sigma=sigma, rho=rho, n=1, nu=5 / 2)
            np.testing.assert_array_almost_equal(matern52(frequencies), matern522)

            matern32 = Matern32(sigma=sigma, rho=rho)
            matern322 = Matern(frequencies, sigma=sigma, rho=rho, n=1, nu=3 / 2)
            np.testing.assert_array_almost_equal(matern32(frequencies), matern322)

    def test_Lorentzian(self):
        Qs = [10, 1, 1 / np.sqrt(2), 0.1]
        S_0 = [10, 5, 1]
        w_0 = 5
        frequencies = np.arange(1, 1000)
        for Q in Qs:
            for S in S_0:
                cel_Lorentzian = celerite_Lorentzian(
                    log_S0=np.log(S), log_Q=np.log(Q), log_omega0=np.log(w_0)
                )
                astropy_Lorentzian = Lorentzian(S0=S, omega0=w_0, Q=Q)
                np.testing.assert_array_almost_equal(
                    astropy_Lorentzian(frequencies), cel_Lorentzian.get_psd(frequencies)
                )


if __name__ == "__main__":
    unittest.main()
