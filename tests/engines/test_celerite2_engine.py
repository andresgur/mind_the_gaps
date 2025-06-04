import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from mind_the_gaps.engines.celerite2_engine import Celerite2GPEngine


class TestCelerite2GPEngine(unittest.TestCase):
    def setUp(self):
        self.kernel_spec = MagicMock()
        self.kernel_spec.get_param_array.return_value = jnp.array([1.0, 2.0])
        self.kernel_spec.get_bounds_array.return_value = [(0.0, 3.0), (1.0, 5.0)]
        self.kernel_spec.terms = [
            MagicMock(
                parameters={
                    "param1": MagicMock(fixed=False, value=1.0, bounds=(0.5, 1.5))
                }
            )
        ]

        self.lightcurve = MagicMock()
        self.lightcurve.times = jnp.linspace(0, 10, 100)
        self.lightcurve.y = jnp.ones(100)
        self.lightcurve.get_simulator.return_value = MagicMock(
            generate_lightcurve=lambda: jnp.ones(100),
            add_noise=lambda x: (x, jnp.ones(100)),
        )

        patcher = patch("mind_the_gaps.engines.celerite2_engine.Celerite2GP")
        self.MockGP = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_gp_instance = self.MockGP.return_value
        self.mock_gp_instance.meanmodel.sampled_parameters = 0
        self.mock_gp_instance.meanmodel.sampled_mean = False
        self.mock_gp_instance.negative_log_likelihood = MagicMock(return_value=1.0)
        self.mock_gp_instance.compute = MagicMock()
        self.mock_gp_instance.get_psd.return_value = MagicMock()
        self.mock_gp_instance.numpyro_dist.return_value = MagicMock()
        self.mock_gp_instance.log_likelihood.return_value = 1.0

        self.engine = Celerite2GPEngine(
            kernel_spec=self.kernel_spec,
            lightcurve=self.lightcurve,
            meanmodel=None,
            mean_params=None,
            seed=0,
            fit_mean=False,
        )

    def test_minimize_calls_compute(self):
        with patch("jaxopt.ScipyBoundedMinimize") as mock_solver:
            mock_instance = mock_solver.return_value
            mock_instance.run.return_value = (jnp.array([1.5, 2.1]), MagicMock())
            result = self.engine.minimize()

            self.assertTrue(self.mock_gp_instance.compute.called)
            np.testing.assert_array_equal(result, jnp.array([1.5, 2.1]))

    def test_initialize_params_generates_values_within_bounds(self):
        params = self.engine.initialize_params(num_chains=10, std_dev=1.0)
        self.assertIn("term0_param1", params)
        values = params["term0_param1"]
        self.assertEqual(values.shape[0], 10)

        param_spec = self.kernel_spec.terms[0].parameters["param1"]
        lower, upper = param_spec.bounds
        self.assertEqual(params["term0_param1"].shape[0], 10)
        self.assertTrue(jnp.all(values >= lower), f"Values below lower bound {lower}")
        self.assertTrue(jnp.all(values <= upper), f"Values above upper bound {upper}")

    def test_generate_from_posteriors_raises_if_samples_not_derived(self):
        self.engine._mcmc_samples = None
        with self.assertRaises(RuntimeError):
            self.engine.generate_from_posteriors(nsims=10)

    def test_max_loglikelihood(self):
        self.engine._loglikelihoods = jnp.array([-2.3, -1.0, -5.5])
        self.assertEqual(self.engine.max_loglikelihood, -1.0)

    def test_max_parameters(self):
        self.engine._loglikelihoods = jnp.array([[1.0, 2.0], [1.5, 1.8]])
        self.engine._mcmc_samples = {
            "log_likelihood": jnp.array([[1.0, 2.0], [1.5, 1.8]]),
            "Param1": jnp.array([[1.0, 1.5], [1.3, 1.4]]),
            "Param2": jnp.array([[2.0, 2.5], [2.3, 2.4]]),
        }
        max_params = self.engine.max_parameters
        np.testing.assert_array_equal(max_params, jnp.array([1.5, 2.5]))

        self.engine._loglikelihoods = jnp.array([[1.0], [2.0]])
        self.engine._mcmc_samples = {
            "log_likelihood": jnp.array([[1.0], [2.0]]),
            "Param1": jnp.array([[1.0], [1.5]]),
            "Param2": jnp.array([[2.0], [2.5]]),
        }
        max_params = self.engine.max_parameters
        np.testing.assert_array_equal(max_params, jnp.array([1.5, 2.5]))

    def test_median_parameters(self):
        self.engine._mcmc_samples = {
            "log_likelihood": jnp.array([[1.0, 2.0], [1.5, 1.8]]),
            "Param1": jnp.array([[1.0, 1.5], [1.3, 1.4]]),
            "Param2": jnp.array([[2.0, 2.5], [2.3, 2.4]]),
        }

        median_params = self.engine.median_parameters
        np.testing.assert_allclose(median_params, jnp.array([1.35, 2.35]), rtol=1e-6)
        self.engine._mcmc_samples = {
            "log_likelihood": jnp.array([[1.0], [2.0]]),
            "Param1": jnp.array([[1.0], [1.5]]),
            "Param2": jnp.array([[2.0], [2.5]]),
        }

        median_params = self.engine.median_parameters
        np.testing.assert_allclose(median_params, jnp.array([1.25, 2.25]), rtol=1e-6)

    def test_derive_posteriors(self):

        true_param1 = jnp.log(1.0)
        true_param2 = jnp.log(3.0)
        true_param3 = jnp.log(5.0)

        def fake_numpyro_model(t, params=None, fit=False):
            numpyro.deterministic("log_likelihood", jnp.array(0.0))
            numpyro.sample("param1", dist.Normal(true_param1, 0.01))
            numpyro.sample("param2", dist.Normal(true_param2, 0.01))
            numpyro.sample("param3", dist.Normal(true_param3, 0.01))

        self.engine.numpyro_model = fake_numpyro_model

        self.engine.derive_posteriors(
            num_warmup=100,
            num_chains=1,
            max_steps=1000,
            converge_steps=200,
            fit=True,
            progress=False,
        )

        mean_1 = jnp.mean(self.engine._mcmc_samples["param1"])
        mean_2 = jnp.mean(self.engine._mcmc_samples["param2"])
        mean_3 = jnp.mean(self.engine._mcmc_samples["param3"])

        self.assertAlmostEqual(
            mean_1, true_param1, delta=0.1, msg="log_S0 not converged"
        )
        self.assertAlmostEqual(
            mean_2, true_param2, delta=0.1, msg="log_Q not converged"
        )
        self.assertAlmostEqual(
            mean_3, true_param3, delta=0.1, msg="log_omega0 not converged"
        )


if __name__ == "__main__":
    unittest.main()
