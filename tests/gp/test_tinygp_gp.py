import unittest
from unittest.mock import MagicMock, patch

import numpyro.distributions as dist
import tinygp
from jax import numpy as jnp
from numpyro.handlers import seed, trace
from tinygp.numpyro_support import TinyDistribution

from mind_the_gaps.gp.celerite2_gaussian_process import Celerite2GP
from mind_the_gaps.gp.tinygp_gaussian_process import TinyGP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve
from mind_the_gaps.models.kernel_spec import (
    KernelParameterSpec,
    KernelSpec,
    KernelTermSpec,
)


class TestCelerite2GP(unittest.TestCase):

    def setUp(self):

        self.mock_lc = MagicMock(spec=GappyLightcurve)
        self.mock_lc.times = jnp.linspace(0, 1, 10)
        self.mock_lc.y = jnp.sin(self.mock_lc.times)
        self.mock_lc.dy = 0.1 * jnp.ones_like(self.mock_lc.times)
        self.mock_lc.mean.return_value = jnp.mean(self.mock_lc.y)

        self.param1 = KernelParameterSpec(
            value=1.0, prior=dist.Uniform, fixed=False, bounds=(0.1, 5.0)
        )
        self.param2 = KernelParameterSpec(value=1e-6, fixed=True)
        self.param3 = KernelParameterSpec(
            value=3.0, prior=dist.Uniform, fixed=False, bounds=(0.1, 5.0)
        )
        self.param4 = KernelParameterSpec(
            value=4.0, prior=dist.Uniform, fixed=False, bounds=(0.1, 5.0)
        )
        self.term_spec = KernelTermSpec(
            term_class=tinygp.kernels.quasisep.Celerite,
            parameters={
                "a": self.param1,
                "b": self.param2,
                "c": self.param3,
                "d": self.param4,
            },
        )
        self.kernel_spec = KernelSpec(terms=[self.term_spec])

        self.rng_key = jnp.array([0, 0], dtype=jnp.uint32)

        self.mean_params = jnp.array([1.0])
        self.gp_model = TinyGP(
            kernel_spec=self.kernel_spec,
            lightcurve=self.mock_lc,
            meanmodel="constant",
            mean_params=self.mean_params,
        )

    def test_initialization(self):
        gp = TinyGP(
            kernel_spec=self.kernel_spec, lightcurve=self.mock_lc, meanmodel="fixed"
        )
        self.assertIsInstance(gp, TinyGP)
        self.assertEqual(gp._lightcurve, self.mock_lc)

    def test_get_kernel_fit(self):
        kernel = self.gp_model.kernel_spec.get_kernel(fit=True)
        self.assertIsInstance(kernel, tinygp.kernels.quasisep.Celerite)
        self.assertEqual(kernel.a, jnp.exp(self.param1.value))
        self.assertEqual(kernel.b, jnp.exp(self.param2.value))
        self.assertEqual(kernel.c, jnp.exp(self.param3.value))
        self.assertEqual(kernel.d, jnp.exp(self.param4.value))

    def test_get_kernel_sampling(self):

        with seed(rng_seed=0), trace() as tr:
            kernel = self.gp_model.kernel_spec.get_kernel(fit=False)

        # Verify sample sites were created
        expected_sample_sites = [
            "terms[0]:log_a",
            "terms[0]:log_c",
            "terms[0]:log_d",
        ]
        for site in expected_sample_sites:
            self.assertIn(site, tr)

        # Validate that the sampled and exponentiated values were passed to the kernel
        sampled_a = jnp.exp(tr["terms[0]:log_a"]["value"])
        sampled_c = jnp.exp(tr["terms[0]:log_c"]["value"])
        sampled_d = jnp.exp(tr["terms[0]:log_d"]["value"])

        self.assertIsInstance(kernel, tinygp.kernels.quasisep.Celerite)
        self.assertAlmostEqual(kernel.a, sampled_a, places=5)
        self.assertAlmostEqual(kernel.b, jnp.exp(self.param2.value), places=5)
        self.assertAlmostEqual(kernel.c, sampled_c, places=5)
        self.assertAlmostEqual(kernel.d, sampled_d, places=5)

    def test_get_kernel_raises_if_no_prior(self):
        self.param1.fixed = False
        self.param1.prior = None
        with self.assertRaises(ValueError):
            self.gp_model.kernel_spec.get_kernel(fit=False)

    def test_numpyro_dist(self):
        mock_dist = MagicMock(spec=TinyDistribution)
        self.gp_model.gp = MagicMock()
        self.gp_model.gp.numpyro_dist.return_value = mock_dist

        result = self.gp_model.numpyro_dist()

        self.gp_model.gp.numpyro_dist.assert_called_once()
        self.assertIs(result, mock_dist)


if __name__ == "__main__":
    unittest.main()
