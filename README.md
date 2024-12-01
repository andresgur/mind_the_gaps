[![DOI](https://zenodo.org/badge/727285474.svg)](https://doi.org/10.5281/zenodo.14253753)

Gaussian Processes time series modelling with focus on period detection on irregularly-sampled stochastically-varying astronomical time series

![Mind The Gaps](https://github.com/andresgur/mind_the_gaps/blob/main/docs/mind_the_gaps.jpg)


# Installation
To install the repository, first clone to your compute:
`git clone https://github.com/andresgur/mind_the_gaps`
then enter the `mind_the_gaps` main directory and pip install it by running:
 `pip install .`
# Warning
Users are welcome to use/test the code and provide feedback but beware, the code is still in alpha phase, so expect bugs, errors and crashes! If you are intending to use the code, my recommendation would be to get in touch first!
# Usage
See the jupyter-notebook tutorial in the notebooks folder

## Workflow

An example workflow is included in [docs/workflow.md](docs/workflow.md).

Usually the user will a stochastic model and a stochastic + periodic component model (let's call them null_hypothesis.config and alternative_model.config). The workflow to determine whether the periodic component is required (i.e. whether there is periodic variability in the lightcurve) would look like this (see [Protassov et al. 2002](https://pages.github.com/](https://ui.adsabs.harvard.edu/abs/2002ApJ...571..545P/abstract))):
1. Fit the observed data using the two config files and `celerite_script.py' (it can be parallelized)
2. Generate lightcurves from the posteriors of the stochastic model fit using `generate_lcs_significance.py' (it can be parallelized)
3. Fit the generated lightcurves using `fit_lcs.py' (it can be parallelized) using both models (i.e. one run for each null_hypothesis and alternative_model config files).
4. Build a histogram of the fit-improvements (delta likelihoods) found for the simulated lightcurves and check where the value observed in the data falls in the histogram. An outlier (say p<0.05) suggest the fit-improvement observed in the data is highly unlikely. High p-values indicate the fit-improvement is simply due to noise


# Tests
Tests are included in the `tests/` directory and can be run as:
```
python setup.py test
```
**Warning:** Tests currently are not passing. Please check the [issues](https://github.com/andresgur/mind_the_gaps/issues/13) on the respository to see the current status.
