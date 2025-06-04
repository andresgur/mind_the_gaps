import unittest
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.handlers

from mind_the_gaps.models.celerite2.mean_terms import (
    ConstantMean,
    FixedMean,
    GaussianMean,
    LinearMean,
)


class DummyLightcurve:
    def __init__(self, y):
        self.y = jnp.array(y)
        self.mean = jnp.mean(self.y)
        self.t = jnp.linspace(0, 1, len(y))


class TestFixedMean(unittest.TestCase):
    def test_compute_mean_returns_fixed_value(self):
        y = jnp.array([1.0, 2.0, 3.0])
        lc = DummyLightcurve(y)
        mean_func = FixedMean(lc)
        self.assertAlmostEqual(float(mean_func.compute_mean()), float(jnp.mean(y)))


class TestConstantMean(unittest.TestCase):
    def setUp(self):
        self.y = jnp.array([1.0, 2.0, 3.0])
        self.lc = DummyLightcurve(self.y)
        self.mean_func = ConstantMean(self.lc)

    def test_compute_mean_fit_mode(self):
        param = jnp.array([2.5])
        result = self.mean_func.compute_mean(params=param, fit=True)
        self.assertTrue(jnp.allclose(result, param))

    def test_compute_mean_sample_mode(self):
        rng_key = jax.random.PRNGKey(0)
        with numpyro.handlers.seed(rng_seed=0):
            val = self.mean_func.compute_mean(fit=False, rng_key=rng_key)
            self.assertTrue(
                self.mean_func.bounds[0][0] <= val <= self.mean_func.bounds[1][0]
            )


class TestLinearMean(unittest.TestCase):
    def setUp(self):
        self.y = jnp.array([1.0, 2.0, 3.0])
        self.lc = DummyLightcurve(self.y)
        self.lc.t = jnp.linspace(0, 1, len(self.y))
        self.mean_func = LinearMean(self.lc)

    def test_compute_mean_with_params(self):
        m, b = 2.0, 1.0
        t = self.lc.t
        result = self.mean_func.compute_mean(params=jnp.array([m, b]), rng_key=None)
        expected = m * t + b
        self.assertTrue(jnp.allclose(result, expected))

    def test_compute_mean_with_sampling(self):
        key = random.PRNGKey(0)

        bounds = {
            "mean_params": jnp.array(
                [
                    [0.0, 1.0],
                    [0.0, 2.0],
                ]
            )
        }

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                result = self.mean_func.compute_mean(
                    params=None, rng_key=key, bounds=bounds
                )

        m = tr["m"]["value"]
        b = tr["b"]["value"]

        expected = m * self.lc.t + b

        self.assertTrue(jnp.allclose(result, expected))
        self.assertEqual(result.shape, self.lc.t.shape)


class TestGaussianMean(unittest.TestCase):
    def setUp(self):
        self.y = jnp.array([1.0, 2.0, 3.0])
        self.lc = DummyLightcurve(self.y)
        self.lc.t = jnp.linspace(0, 1, len(self.y))
        self.mean_func = GaussianMean(self.lc)

    def test_compute_mean_with_params(self):
        t = jnp.linspace(0, 1, 10)
        A, mu, sigma = 1.0, 0.5, 0.1
        result = self.mean_func.compute_mean(
            t=t,
            params=jnp.array([A, mu, sigma]),
            rng_key=None,
        )
        expected = A * jnp.exp(-((t - mu) ** 2) / (2 * sigma**2))
        self.assertTrue(jnp.allclose(result, expected))

    def test_compute_mean_with_sampling(self):
        t = jnp.linspace(0, 1, 10)
        key = random.PRNGKey(0)

        bounds = {
            "mean_params": jnp.array(
                [
                    [0.5, 1.5],  # A in [0.5, 1.5]
                    [0.2, 0.8],  # mu in [0.2, 0.8]
                    [0.05, 0.2],  # sigma in [0.05, 0.2]
                ]
            )
        }

        with numpyro.handlers.seed(rng_seed=0):
            with numpyro.handlers.trace() as tr:
                result = self.mean_func.compute_mean(
                    t=t, params=None, rng_key=key, bounds=bounds
                )

        # Extract actual sampled values
        A = tr["amplitude"]["value"]
        mu = tr["mu"]["value"]
        sigma = tr["sigma"]["value"]

        expected = A * jnp.exp(-((t - mu) ** 2) / (2 * sigma**2))

        self.assertTrue(jnp.allclose(result, expected))
        self.assertEqual(result.shape, t.shape)


if __name__ == "__main__":
    unittest.main()
