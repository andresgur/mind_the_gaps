# @Author: Andrés Gúrpide <agurpide>
# @Date:   28-03-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-03-2022
import unittest
from mind_the_gaps.psd_models import BendingPowerlaw, SHO, Matern32, Lorentzian
from mind_the_gaps.celerite_models import DampedRandomWalk as DRW_celerite
from mind_the_gaps.celerite_models import Lorentzian as celerite_Lorentzian
from celerite.terms import SHOTerm, Matern32Term
import numpy as np
import matplotlib.pyplot as plt


class TestSimulator(unittest.TestCase):

    def test_DRW(self):
        Q = 1 /2
        S_0 = 10
        w_0 = 5

        cel_BPL = DRW_celerite(log_S0=np.log(S_0), log_omega0=np.log(w_0))
        astropy_BPL = BendingPowerlaw(S_0=S_0, w_0=w_0,Q=Q)
        frequencies = np.arange(1, 1000)
        np.testing.assert_array_almost_equal(astropy_BPL(frequencies), cel_BPL.get_psd(frequencies))

    def test_SHO(self):
        Qs = [10, 1, 1 / np.sqrt(2), 0.1]
        S_0 = 10
        w_0 = 5
        frequencies = np.arange(1, 1000)
        for Q in Qs:
            cel_SHO = SHOTerm(log_S0=np.log(S_0), log_Q=np.log(Q), log_omega0=np.log(w_0))
            astropy_SHO = SHO(S_0=S_0, w_0=w_0, Q=Q)
            np.testing.assert_array_almost_equal(astropy_SHO(frequencies), cel_SHO.get_psd(frequencies))

    def test_mattern(self):
        sigma = 10
        rhos = [1, 10, 20]
        frequencies = np.arange(1, 1000)
        for rho in rhos:
            cel_mattern = Matern32Term(log_sigma=np.log(sigma), log_rho=np.log(rho), eps=1e-15)
            astropy_mattern = Matern32(sigma=sigma, rho=rho)
            np.testing.assert_array_almost_equal(astropy_mattern(frequencies), cel_mattern.get_psd(frequencies))

    def test_Lorentzian(self):
        Qs = [10, 1, 1 / np.sqrt(2), 0.1]
        S_0 = 10
        w_0 = 5
        frequencies = np.arange(1, 1000)
        for Q in Qs:
            cel_Lorentzian = celerite_Lorentzian(log_S0=np.log(S_0), log_Q=np.log(Q), log_omega0=np.log(w_0))
            astropy_Lorentzian = Lorentzian(S_0=S_0, w_0=w_0, Q=Q)
            np.testing.assert_array_almost_equal(astropy_Lorentzian(frequencies), cel_Lorentzian.get_psd(frequencies))


if __name__ == '__main__':
    unittest.main()
