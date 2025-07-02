import unittest
from unittest.mock import MagicMock, patch

import celerite2.jax.terms as jax_terms
import celerite.terms as cel_terms
import jax
import numpy as np
import numpyro.distributions as dist
import tinygp
from celerite2.jax.distribution import CeleriteNormal
from jax import numpy as jnp

from mind_the_gaps.gp.celerite2_gaussian_process import Celerite2GP
from mind_the_gaps.gp.celerite_gaussian_process import CeleriteGP
from mind_the_gaps.gp.tinygp_gaussian_process import TinyGP
from mind_the_gaps.lightcurves.gappylightcurve import GappyLightcurve
from mind_the_gaps.models.kernel_spec import (KernelParameterSpec, KernelSpec,
                                              KernelTermSpec)


class GPRegressionTest(unittest.TestCase):

    def setUp(self):

        mean_value = 100
        noise_std = 5
        N = 1000
        times = np.linspace(0, 10, N)
        synthetic_countrates = mean_value + np.random.normal(0, noise_std, size=N)
        errors = np.full_like(synthetic_countrates, noise_std)

        t_jax = jnp.array(times)
        y_jax = jnp.array(synthetic_countrates)
        dy_jax = jnp.array(errors)

        flux_centered = y_jax

        kernel_params = jnp.array([4.60517019, -1.15785521])
        a = jnp.exp(kernel_params[0])
        c = jnp.exp(kernel_params[1])
        b = 3.0
        d = 4.0

        cel2_kernel_spec = KernelSpec(
            terms=[
                KernelTermSpec(
                    term_class=jax_terms.ComplexTerm,
                    parameters={
                        "a": KernelParameterSpec(
                            value=np.log(a),
                            fixed=False,
                            prior=dist.Uniform,
                            bounds=(0.1, 20.0),
                        ),
                        "b": KernelParameterSpec(
                            value=np.log(b),
                            fixed=False,
                            prior=dist.Uniform,
                            bounds=(0.1, 20.0),
                        ),
                        "c": KernelParameterSpec(
                            value=np.log(c),
                            fixed=False,
                            prior=dist.Uniform,
                            bounds=(0.01, 10.0),
                        ),
                        "d": KernelParameterSpec(
                            value=np.log(d),
                            fixed=False,
                            prior=dist.Uniform,
                            bounds=(0.01, 10.0),
                        ),
                    },
                )
            ]
        )

        cel_kernel_spec = KernelSpec(
            terms=[
                KernelTermSpec(
                    term_class=cel_terms.ComplexTerm,
                    parameters={
                        "log_a": KernelParameterSpec(
                            value=np.log(a),
                            fixed=False,
                            bounds=(np.log(1e-2), np.log(1e3)),
                        ),
                        "log_b": KernelParameterSpec(
                            value=np.log(b),
                            fixed=False,
                            bounds=(np.log(1e-2), np.log(1e3)),
                        ),
                        "log_c": KernelParameterSpec(
                            value=np.log(c),
                            fixed=False,
                            bounds=(np.log(1e-3), np.log(1e2)),
                        ),
                        "log_d": KernelParameterSpec(
                            value=np.log(d),
                            fixed=False,
                            bounds=(np.log(1e-3), np.log(1e2)),
                        ),
                    },
                )
            ]
        )

        tinygp_spec = KernelSpec(
            terms=[
                KernelTermSpec(
                    term_class=tinygp.kernels.quasisep.Celerite,
                    parameters={
                        "a": KernelParameterSpec(
                            value=np.log(a),
                            fixed=False,
                            prior=dist.Uniform,
                            bounds=(0.1, 20.0),
                        ),
                        "b": KernelParameterSpec(
                            value=np.log(b),
                            fixed=False,
                            prior=dist.Uniform,
                            bounds=(0.1, 20.0),
                        ),
                        "c": KernelParameterSpec(
                            value=np.log(c),
                            fixed=False,
                            prior=dist.Uniform,
                            bounds=(0.01, 10.0),
                        ),
                        "d": KernelParameterSpec(
                            value=np.log(d),
                            fixed=False,
                            prior=dist.Uniform,
                            bounds=(0.01, 10.0),
                        ),
                    },
                )
            ],
        )

        self.lc = GappyLightcurve(times=times, y=flux_centered, dy=errors)
        self.cel2_gp = Celerite2GP(
            kernel_spec=cel2_kernel_spec,
            lightcurve=self.lc,
            meanmodel="fixed",
            mean_params=jnp.array([self.lc.mean]),
        )
        self.cel_gp = CeleriteGP(
            kernel_spec=cel_kernel_spec,
            lightcurve=self.lc,
            fit_mean=True,
        )

        self.tinygp_gp = TinyGP(
            kernel_spec=tinygp_spec,
            lightcurve=self.lc,
            meanmodel="fixed",
            mean_params=jnp.array([self.lc.mean]),
        )

    def test_gp_regression(self):
        self.assertAlmostEqual(
            self.cel2_gp.negative_log_likelihood(
                self.cel2_gp.kernel_spec.get_param_array()
            ),
            -self.cel_gp.log_likelihood(self.lc.y),
        )
        self.assertAlmostEqual(
            self.tinygp_gp.negative_log_likelihood(
                self.tinygp_gp.kernel_spec.get_param_array()
            ),
            -self.cel_gp.log_likelihood(self.lc.y),
        )

    def test_gp_regression_real(self):
        mean_value = 100
        noise_std = 5
        N = 1000
        times = np.linspace(0, 10, N)
        synthetic_countrates = mean_value + np.random.normal(0, noise_std, size=N)
        errors = np.full_like(synthetic_countrates, noise_std)

        t_jax = jnp.array(times)
        y_jax = jnp.array(synthetic_countrates)
        dy_jax = jnp.array(errors)

        flux_centered = y_jax

        kernel_params = jnp.array([4.60517019, -1.15785521])
        a = jnp.exp(kernel_params[0])
        c = jnp.exp(kernel_params[1])
        b = 0.0
        d = 0.0

        cel2_kernel_spec = KernelSpec(
            terms=[
                KernelTermSpec(
                    term_class=jax_terms.RealTerm,
                    parameters={
                        "a": KernelParameterSpec(
                            value=np.log(a),
                            fixed=False,
                            prior=dist.Uniform,
                            bounds=(0.1, 20.0),
                        ),
                        "c": KernelParameterSpec(
                            value=np.log(c),
                            fixed=False,
                            prior=dist.Uniform,
                            bounds=(0.01, 10.0),
                        ),
                    },
                )
            ]
        )

        cel_kernel_spec = KernelSpec(
            terms=[
                KernelTermSpec(
                    term_class=cel_terms.RealTerm,
                    parameters={
                        "log_a": KernelParameterSpec(
                            value=np.log(a),
                            fixed=False,
                            bounds=(np.log(1e-2), np.log(1e3)),
                        ),
                        "log_c": KernelParameterSpec(
                            value=np.log(c),
                            fixed=False,
                            bounds=(np.log(1e-3), np.log(1e2)),
                        ),
                    },
                )
            ]
        )

        tinygp_spec = KernelSpec(
            terms=[
                KernelTermSpec(
                    term_class=tinygp.kernels.quasisep.Celerite,
                    parameters={
                        "a": KernelParameterSpec(
                            value=np.log(a),
                            fixed=False,
                            prior=dist.Uniform,
                            bounds=(0.1, 20.0),
                        ),
                        "b": KernelParameterSpec(
                            value=-1e-20,
                            zeroed=True,
                        ),
                        "c": KernelParameterSpec(
                            value=np.log(c),
                            fixed=False,
                            prior=dist.Uniform,
                            bounds=(0.01, 10.0),
                        ),
                        "d": KernelParameterSpec(value=-1e-20, zeroed=True),
                    },
                )
            ],
        )

        self.lc = GappyLightcurve(times=times, y=flux_centered, dy=errors)
        self.cel2_gp = Celerite2GP(
            kernel_spec=cel2_kernel_spec,
            lightcurve=self.lc,
            meanmodel="fixed",
            mean_params=jnp.array([self.lc.mean]),
        )
        self.cel_gp = CeleriteGP(
            kernel_spec=cel_kernel_spec,
            lightcurve=self.lc,
            fit_mean=True,
        )

        self.tinygp_gp = TinyGP(
            kernel_spec=tinygp_spec,
            lightcurve=self.lc,
            meanmodel="fixed",
            mean_params=jnp.array([self.lc.mean]),
        )
        self.assertAlmostEqual(
            self.cel2_gp.negative_log_likelihood(
                self.cel2_gp.kernel_spec.get_param_array()
            ),
            -self.cel_gp.log_likelihood(self.lc.y),
        )
        self.assertAlmostEqual(
            self.tinygp_gp.negative_log_likelihood(
                self.tinygp_gp.kernel_spec.get_param_array()
            ),
            -self.cel_gp.log_likelihood(self.lc.y),
        )


if __name__ == "__main__":
    unittest.main()
