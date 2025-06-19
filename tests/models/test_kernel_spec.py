import unittest
from collections import OrderedDict
from unittest import skip
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from celerite import terms as cel_terms
from celerite2.jax import terms as cel2_terms

from mind_the_gaps.models.kernel_spec import (
    KernelParameterSpec,
    KernelSpec,
    KernelTermSpec,
)


class KernelSpecTestCase(unittest.TestCase):
    def setUp(self):
        self.param1 = KernelParameterSpec(value=1.0, fixed=False, bounds=(0.1, 10.0))
        self.param2 = KernelParameterSpec(value=2.0, fixed=True)

        self.term_spec1 = KernelTermSpec(
            term_class=cel_terms.RealTerm,
            parameters={"log_a": self.param1, "log_c": self.param2},
        )

        self.param3 = KernelParameterSpec(value=3.0, fixed=False, bounds=(0.5, 5.0))
        self.param4 = KernelParameterSpec(value=4.0, fixed=False, bounds=(0.1, 5.0))
        self.param5 = KernelParameterSpec(value=5.0, fixed=True)
        self.term_spec2 = KernelTermSpec(
            term_class=cel_terms.ComplexTerm,
            parameters={
                "log_a": self.param3,
                "log_c": self.param4,
                "log_d": self.param5,
            },
        )

        self.kernel_spec = KernelSpec(terms=[self.term_spec1, self.term_spec2])

    def test_add_kernel_spec(self):
        combined = self.kernel_spec + self.kernel_spec
        self.assertIsInstance(combined, KernelSpec)
        self.assertEqual(len(combined.terms), 4)

    def test_add_invalid_type_raises(self):
        with self.assertRaises(TypeError):
            _ = self.kernel_spec + 123

    def test_update_params_from_array(self):
        array = np.array([1.5, 2.5, 3.5])

        self.kernel_spec.update_params_from_array(array)
        self.assertEqual(self.param1.value, 1.5)
        self.assertEqual(self.param2.value, 2.0)
        self.assertEqual(self.param3.value, 2.5)
        self.assertEqual(self.param4.value, 3.5)
        self.assertEqual(self.param5.value, 5.0)

        with self.assertRaises(ValueError):
            self.kernel_spec.update_params_from_array(np.array([1.0, 2.0]))

    def test_get_param_array(self):

        param_array = self.kernel_spec.get_param_array()
        self.assertTrue(np.allclose(param_array, [1.0, 3.0, 4.0]))

    def test_get_param_names(self):
        names = self.kernel_spec.get_param_names()
        self.assertEqual(
            names,
            ["term0.log_a", "term1.log_a", "term1.log_c"],
        )

    def test_get_bounds_array(self):
        bounds = self.kernel_spec.get_bounds_array()
        np.testing.assert_array_equal(
            bounds, np.array([[0.1, 10.0], [0.5, 5.0], [0.1, 5.0]])
        )

    def test_get_bounds_array_raises_if_missing(self):
        self.param3.bounds = None
        with self.assertRaises(ValueError):
            _ = self.kernel_spec.get_bounds_array()

    def test_get_kernel(self):
        kernel = self.kernel_spec.get_kernel(fit=False)
        self.assertIsInstance(kernel, cel_terms.Term)


class KernelSpecCelerite2TestCase(unittest.TestCase):
    def setUp(self):
        self.param1 = KernelParameterSpec(
            value=1.0, fixed=False, prior=dist.Uniform, bounds=(0.1, 5.0)
        )
        self.param2 = KernelParameterSpec(value=1e-6, fixed=True)
        self.param3 = KernelParameterSpec(
            value=3.0, fixed=False, prior=dist.Uniform, bounds=(0.1, 5.0)
        )
        self.param4 = KernelParameterSpec(
            value=4.0, fixed=False, prior=dist.Uniform, bounds=(0.1, 5.0)
        )
        self.term_spec = KernelTermSpec(
            term_class=cel2_terms.ComplexTerm,
            parameters={
                "a": self.param1,
                "b": self.param2,
                "c": self.param3,
                "d": self.param4,
            },
        )
        self.kernel_spec = KernelSpec(terms=[self.term_spec])

    def test_add_kernel_spec(self):
        combined = self.kernel_spec + self.kernel_spec
        self.assertIsInstance(combined, KernelSpec)
        self.assertEqual(len(combined.terms), 2)

    def test_get_jax_kernel(self):

        kernel = self.kernel_spec._get_jax_kernel()
        self.assertIsInstance(kernel, cel2_terms.ComplexTerm)


if __name__ == "__main__":
    unittest.main()
