import unittest
from .stats import *
from scipy.stats import norm, lognorm, rv_continuous
import numpy as np
from scipy import special

class TestSimulator(unittest.TestCase):

    def test_create_lognormal(self):

        var = 5 # desired variance
        mean = 12 # desired mean
        samples = create_log_normal(mean, var).rvs(size=2000000)
        self.assertAlmostEqual(np.mean(samples), mean, 2, "Log normal probability distribution has wrong mean")
        self.assertAlmostEqual(np.var(samples), var, 2, "Log normal probability distribution has wrong variance")


    def test_create_uniform(self):

        var = 5 # desired variance
        means = [1, 10, 12] # desired mean
        for mean in means:
            samples = create_uniform_distribution(mean, var).rvs(size=2000000)
            self.assertAlmostEqual(np.mean(samples), mean, 2, "Uniform probability distribution has wrong mean")
            self.assertAlmostEqual(np.var(samples), var, 2, "Uniform probability distribution has wrong variance")

    def test_lognormal(self):
        log_1 = lognorm(1)
        samples = log_1.rvs(size=100000)
        # center 0, sigma = 1
        log_2 = lognormal(a=0)(0, 1)
        samples_2 = log_2.rvs(size=100000)
        self.assertAlmostEqual(np.mean(samples), np.mean(samples_2), 3, "Log normal probability distribution gives unexpected results")

    def test_chi_loglikelihood(self):
        data_powers = np.array([0, 1, 2])
        model_powers = np.array([0.5, 1.5, 2.5])

        log_like = 2. * (np.log(model_powers[0]) + data_powers[0]/model_powers[0] + np.log(model_powers[1]) + data_powers[1]/model_powers[1] +
                       np.log(model_powers[2]) + data_powers[2]/model_powers[2])
        self.assertEqual(log_like, chi_log_likehood(data_powers, model_powers), "Log-likehood for odd values gives unexpected results")

        data_powers = np.array([0, 1, 2, 3])
        model_powers = np.array([0.5, 1.5, 2.5, 3.5])

        log_like = 2 * (np.log(model_powers[0]) + data_powers[0]/model_powers[0] +np.log(model_powers[1]) + data_powers[1]/model_powers[1] +np.log(model_powers[2]) + data_powers[2]/model_powers[2])
        log_like += np.log(np.pi * data_powers[-1] * model_powers[-1]) + 2 * data_powers[-1]/model_powers[-1]
        self.assertEqual(log_like, chi_log_likehood(data_powers, model_powers), "Log-likehood for even values gives unexpected results")

    def test_chi_cov(self):
        N = 500
        input_cov = np.array([[1, 1],[1 , 1]])
        samples = np.random.multivariate_normal([0, 0], input_cov, size=N)
        cov = np.cov(samples, rowvar=False, bias=True)

        np.testing.assert_array_almost_equal(input_cov, cov, 4)
        print(np.linalg.diag(cov))
        model = np.arange(N)
        invcov =  np.linalg.inv(cov)

        input_cov = np.array([[1.5, 0, 0],[0 , 1.5, 0], [0, 0,  1.5]])
        model = np.array([5, 2, 6])
        data = np.array([4, 1, 5])
        self.assertAlmostEqual(chi_square(data, model, np.sqrt(np.diag(input_cov))),
                               chi_cov(data, model, np.linalg.inv(input_cov)), 2)


if __name__ == '__main__':
    unittest.main()
