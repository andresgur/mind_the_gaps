import unittest

import numpy as np

from mind_the_gaps.gpmodelling import GPModelling
from mind_the_gaps.lightcurves import GappyLightcurve
from mind_the_gaps.models import DampedRandomWalk, Lorentzian
from mind_the_gaps.models.kernel_spec import (
    KernelParameterSpec,
    KernelSpec,
    KernelTermSpec,
)


class TestGPModelling(unittest.TestCase):

    def test_parameters_within_bounds(self):
        walkers = 100
        lor_params = [10, 5, -5]
        drw_params = [5.0, 10.0]
        parameters = drw_params + lor_params
        bounds_drw = [(4.0, 6.0), (8.0, 12.0)]
        bounds_lor = [(5, 15), (1, 6), (-7, -1)]  # Bounds for the parameters

        kernel_spec = KernelSpec(
            terms=[
                KernelTermSpec(
                    term_class=DampedRandomWalk,
                    parameters={
                        "log_S0": KernelParameterSpec(
                            value=drw_params[0], bounds=bounds_drw[0]
                        ),
                        "log_omega0": KernelParameterSpec(
                            value=drw_params[1], bounds=bounds_drw[1]
                        ),
                    },
                ),
                KernelTermSpec(
                    term_class=Lorentzian,
                    parameters={
                        "log_S0": KernelParameterSpec(
                            value=lor_params[0], bounds=bounds_lor[0]
                        ),
                        "log_Q": KernelParameterSpec(
                            value=lor_params[1], bounds=bounds_lor[1]
                        ),
                        "log_omega0": KernelParameterSpec(
                            value=lor_params[2], bounds=bounds_lor[2]
                        ),
                    },
                ),
            ]
        )

        lc = GappyLightcurve(np.arange(100), np.arange(100), np.arange(100))

        gpmodel = GPModelling(lightcurve=lc, kernel_spec=kernel_spec)

        bounds = bounds_drw + bounds_lor
        samples = gpmodel.modelling_engine._spread_walkers(
            walkers, parameters, bounds, percent=0.1, max_attempts=100
        )

        # Ensure all walkers have values within the bounds
        for i, sample in enumerate(samples.T):
            self.assertTrue(
                np.all(np.logical_and(bounds[i][0] <= sample, sample <= bounds[i][1]))
            )

        samples = gpmodel.modelling_engine._spread_walkers(
            walkers, parameters, bounds, percent=0.9, max_attempts=2
        )

        # Ensure all walkers have values within the bounds
        for i, sample in enumerate(samples.T):
            self.assertTrue(
                np.all(np.logical_and(bounds[i][0] <= sample, sample <= bounds[i][1]))
            )

    def test_infinite_bounds(self):
        walkers = 100
        lor_params = [10, 5, -5]
        drw_params = [5.0, 10.0]
        parameters = drw_params + lor_params
        bounds_drw = [(None, None), (8.0, 12.0)]
        bounds_lor = [(5, 15), (1, 6), (-7, -1)]  # Bounds for the parameters

        kernel_spec = KernelSpec(
            terms=[
                KernelTermSpec(
                    term_class=DampedRandomWalk,
                    parameters={
                        "log_S0": KernelParameterSpec(
                            value=drw_params[0], bounds=bounds_drw[0]
                        ),
                        "log_omega0": KernelParameterSpec(
                            value=drw_params[1], bounds=bounds_drw[1]
                        ),
                    },
                ),
                KernelTermSpec(
                    term_class=Lorentzian,
                    parameters={
                        "log_S0": KernelParameterSpec(
                            value=lor_params[0], bounds=bounds_lor[0]
                        ),
                        "log_Q": KernelParameterSpec(
                            value=lor_params[1], bounds=bounds_lor[1]
                        ),
                        "log_omega0": KernelParameterSpec(
                            value=lor_params[2], bounds=bounds_lor[2]
                        ),
                    },
                ),
            ]
        )

        lc = GappyLightcurve(np.arange(100), np.arange(100), np.arange(100))

        gpmodel = GPModelling(lightcurve=lc, kernel_spec=kernel_spec)
        bounds = bounds_drw + bounds_lor
        samples = gpmodel.modelling_engine._spread_walkers(
            walkers, parameters, bounds, percent=0.1, max_attempts=50
        )
        # first parameter can be any value
        self.assertTrue(np.all(np.isfinite(samples[:, 0])))
        # Ensure all walkers have values within the bounds
        for bounds_i, sample in zip(bounds[1:], samples.T[1:]):
            self.assertTrue(
                np.all(np.logical_and(bounds_i[0] <= sample, sample <= bounds_i[1]))
            )

        samples = gpmodel.modelling_engine._spread_walkers(
            walkers, parameters, bounds, percent=0.99, max_attempts=5
        )
        # first parameter can be any value
        self.assertTrue(np.all(np.isfinite(samples[:, 0])))

    def test_zero_percent(self):
        walkers = 100
        lor_params = [10, 5, -5]
        drw_params = [5.0, 10.0]
        parameters = drw_params + lor_params
        bounds_drw = [(None, None), (8.0, 12.0)]
        bounds_lor = [(5, 15), (1, 6), (-7, -1)]  # Bounds for the parameters
        kernel_spec = KernelSpec(
            terms=[
                KernelTermSpec(
                    term_class=DampedRandomWalk,
                    parameters={
                        "log_S0": KernelParameterSpec(
                            value=drw_params[0], bounds=bounds_drw[0]
                        ),
                        "log_omega0": KernelParameterSpec(
                            value=drw_params[1], bounds=bounds_drw[1]
                        ),
                    },
                ),
                KernelTermSpec(
                    term_class=Lorentzian,
                    parameters={
                        "log_S0": KernelParameterSpec(
                            value=lor_params[0], bounds=bounds_lor[0]
                        ),
                        "log_Q": KernelParameterSpec(
                            value=lor_params[1], bounds=bounds_lor[1]
                        ),
                        "log_omega0": KernelParameterSpec(
                            value=lor_params[2], bounds=bounds_lor[2]
                        ),
                    },
                ),
            ]
        )

        lc = GappyLightcurve(np.arange(100), np.arange(100), np.arange(100))

        gpmodel = GPModelling(lightcurve=lc, kernel_spec=kernel_spec)

        bounds = bounds_drw + bounds_lor
        samples = gpmodel.modelling_engine._spread_walkers(
            walkers, parameters, bounds, percent=0, max_attempts=50
        )

        # All samples should be exactly at the parameters
        np.testing.assert_array_equal(samples, np.array([parameters] * walkers))

    def test_max_attempts(self):
        walkers = 100
        lor_params = [10, 5, -5]
        drw_params = [5.0, 10.0]
        parameters = drw_params + lor_params
        bounds_drw = [
            (drw_params[0] - 0.01, drw_params[0] + 0.01),
            (drw_params[1] - 0.01, drw_params[1] + 0.01),
        ]
        bounds_lor = [
            (lor_params[0] - 0.01, lor_params[0] + 0.01),
            (lor_params[1] - 0.01, lor_params[1] + 0.01),
            (lor_params[2] - 0.01, lor_params[2] + 0.01),
        ]  # Bounds for the parameters

        kernel_spec = KernelSpec(
            terms=[
                KernelTermSpec(
                    term_class=DampedRandomWalk,
                    parameters={
                        "log_S0": KernelParameterSpec(
                            value=drw_params[0], bounds=bounds_drw[0]
                        ),
                        "log_omega0": KernelParameterSpec(
                            value=drw_params[1], bounds=bounds_drw[1]
                        ),
                    },
                ),
                KernelTermSpec(
                    term_class=Lorentzian,
                    parameters={
                        "log_S0": KernelParameterSpec(
                            value=lor_params[0], bounds=bounds_lor[0]
                        ),
                        "log_Q": KernelParameterSpec(
                            value=lor_params[1], bounds=bounds_lor[1]
                        ),
                        "log_omega0": KernelParameterSpec(
                            value=lor_params[2], bounds=bounds_lor[2]
                        ),
                    },
                ),
            ]
        )

        lc = GappyLightcurve(np.arange(100), np.arange(100), np.arange(100))

        gpmodel = GPModelling(lightcurve=lc, kernel_spec=kernel_spec)

        bounds = bounds_drw + bounds_lor
        samples = gpmodel.modelling_engine._spread_walkers(
            walkers, parameters, bounds, percent=0, max_attempts=50
        )

        # Since the bounds are impossible, all samples should be equal to the parameter
        for i, sample in enumerate(samples.T):
            self.assertTrue(np.all(sample == parameters[i]))


if __name__ == "__main__":
    unittest.main()
