import unittest
from typing import Any, List, Mapping

import jax.numpy as jnp
import numpy as np

from mind_the_gaps.engines.gp_engine import BaseGPEngine
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve


class DummyGPEngine(BaseGPEngine):
    posterior_params = {"param1", "param2"}

    def __init__(self):
        self._ndim = 3
        self._loglikelihoods = jnp.array([1.0, 2.0])
        self._mcmc_samples = np.array([[1.0, 2.0], [3.0, 4.0]])

    def derive_posteriors(
        self, **engine_kwargs: Mapping[str, Any]
    ) -> List[GappyLightcurve]:
        return []

    def generate_from_posteriors(
        self, **engine_kwargs: Mapping[str, Any]
    ) -> List[GappyLightcurve]:
        return []

    @property
    def autocorr(self) -> List[float]:
        return [0.5, 0.7]

    @property
    def max_loglikelihood(self):
        return 123.4

    @property
    def max_parameters(self):
        return np.array([1.0, 2.0, 3.0])

    @property
    def median_parameters(self):
        return np.array([1.5, 2.5, 3.5])

    @property
    def parameter_names(self):
        return ["param1", "param2", "param3"]

    @property
    def tau(self):
        return [20.0, 30.0]


class TestBaseGPEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DummyGPEngine()

    def test_validate_kwargs(self):
        self.engine.validate_posterior_kwargs({"param1": 1, "param2": "test"})
        with self.assertRaises(ValueError):
            self.engine.validate_posterior_kwargs({"bad_param": 123})

    def test_ndim_property(self):
        self.assertEqual(self.engine.k, 3)

    def test_loglikelihoods_property(self):
        np.testing.assert_array_equal(self.engine.loglikelihoods, jnp.array([1.0, 2.0]))

    def test_mcmc_samples_property(self):
        np.testing.assert_array_equal(
            self.engine.mcmc_samples, np.array([[1.0, 2.0], [3.0, 4.0]])
        )

    def test_autocorr_property(self):
        self.assertEqual(self.engine.autocorr, [0.5, 0.7])

    def test_max_loglikelihood_property(self):
        self.assertEqual(self.engine.max_loglikelihood, 123.4)

    def test_max_parameters_property(self):
        np.testing.assert_array_equal(self.engine.max_parameters, [1.0, 2.0, 3.0])

    def test_median_parameters_property(self):
        np.testing.assert_array_equal(self.engine.median_parameters, [1.5, 2.5, 3.5])

    def test_parameter_names_property(self):
        self.assertEqual(self.engine.parameter_names, ["param1", "param2", "param3"])

    def test_tau_property(self):
        self.assertEqual(self.engine.tau, [20.0, 30.0])

    def test_loglikelihoods_missing(self):
        engine = DummyGPEngine()
        engine._loglikelihoods = None
        with self.assertRaises(AttributeError):
            _ = engine.loglikelihoods

    def test_mcmc_samples_missing(self):
        engine = DummyGPEngine()
        engine._mcmc_samples = None
        with self.assertRaises(AttributeError):
            _ = engine.mcmc_samples


class TestSubclassValidation(unittest.TestCase):
    def test_missing_posterior_params_raises(self):
        with self.assertRaises(NotImplementedError):

            class IncompleteGPEngine(BaseGPEngine):
                posterior_params = None

                def derive_posteriors(self, **engine_kwargs):
                    pass

                def generate_from_posteriors(self, **engine_kwargs):
                    pass

                @property
                def autocorr(self):
                    return []

                @property
                def max_loglikelihood(self):
                    return 0

                @property
                def max_parameters(self):
                    return []

                @property
                def median_parameters(self):
                    return []

                @property
                def parameter_names(self):
                    return []

                @property
                def tau(self):
                    return []
