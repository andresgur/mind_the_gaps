"""
This is a fictional script depicting how the code might be used after refactoring into an
object-oriented model
"""
# Code imports
import numpy as np
from multiprocessing import Pool
# These are the imaginary new classes the code would be refactored into
from mind_the_gaps.models import Model
from mind_the_gaps.models.lorentzian import Lorentzian
from mind_the_gaps.models.damped_random_walk import DampedRandomWalk
from mind_the_gaps.lightcurves import SwiftLightcurve
from mind_the_gaps.simulation.methods import TimmerKoenig95
from mind_the_gaps.stats import TestSignificance

# Purely imported for typing
from typing import List
# More imaginary new classes; here I import classes that are returned from functions,
# so we can use them in typing annotations for clarity
from mind_the_gaps.models import ModelFit
from mind_the_gaps.lightcurves import SyntheticLightcurve


# Setup the models and lightcurve
# This replaces having the config settings spread around multiple different files
# and command-line calls - the full pipeline is documented and easily reproducibile
null_hypothesis = Model(
    DampedRandomWalk(
        S_0=[1E-11, 1E-8, 1],
        period=[1, 61, 900]
    )
)
alternative_hypothesis = Model(
    [
        DampedRandomWalk(
            S_0=[1E-15, 1E-8, 1],
            period=[1, 61, 900]
        ),
        Lorentzian(
            S_0=[1E-15, 1E-8, 1],
            period=[2, 100, 500],
            Q=[0.5, 20, 3000000]
        )
    ]
)

lightcurve = SwiftLightcurve("input.dat")
num_processes: int = 16

# Fit the real lightcurve
null_real_fit: ModelFit = null_hypothesis.fit_lightcurve(
    lightcurve, limit=None
)
alternative_real_fit: ModelFit = alternative_hypothesis.fit_lightcurve(
    lightcurve, limit=None
)

# Generate synthetic lightcurves
synthetic_lightcurves: List[SyntheticLightcurve] = null_real_fit.generate_lightcurves(
    pdf_model="Gaussian", processes=num_processes
)

# Fit synthetic lightcurves
# This could be moved to a method on the model itself,
# and probably should so people don't end up having to write parallel code
# e.g.
# null_synthetic_fits: List[ModelFit] = null_hypothesis.fit_lightcurve(synthetic_lightcurves, num_processes)

with Pool(processes=num_processes, initializer=np.random.seed) as pool:
    null_synthetic_fits: List[ModelFit] = pool.map(
        null_hypothesis.fit_lightcurve, 
        synthetic_lightcurves
    )

with Pool(processes=num_processes, initializer=np.random.seed) as pool:
    alternative_synthetic_fits: List[ModelFit] = pool.map(
        alternative_hypothesis.fit_lightcurve, 
        synthetic_lightcurves
    )

# Test hypothesis. 
# This could be done as a function, or as a class,
# but a class makes it easier to 
comparison = TestSignificance(
    null_real_fit=null_real_fit, 
    alternative_real_fit=alternative_real_fit,
    null_synthetic_fits=null_synthetic_fits, 
    alternative_synthetic_fits=alternative_synthetic_fits
)
# Then you can get the statistics from the comparison and feed them into
# plotting routines you've already written, or people can write their own.
stats = TestSignificance.get_statistics()  

# Splitting plotting out from the main code into a submodule that takes 
# input from the models makes it much easier to create a variety of different
# plots
from mind_the_gaps.plotting import plot_model_fits_on_lightcurve
plot_model_fits_on_lightcurve(
    lightcurve=lightcurve,
    model_fits={
        "Damped Random Walk": null_real_fit,
        "DRW + Lorentzian": alternative_real_fit,
    },
    x_label="Day (MJD)",
    y_label="Count rate (s^{-1})"
)
