[![DOI](https://zenodo.org/badge/727285474.svg)](https://doi.org/10.5281/zenodo.14600069)
![Tests](https://github.com/andresgur/mind_the_gaps/actions/workflows/test.yml/badge.svg)

Gaussian Processes time series modelling with focus on period detection on irregularly-sampled stochastically-varying astronomical time series

![Mind The Gaps](https://github.com/andresgur/mind_the_gaps/blob/main/docs/mind_the_gaps.jpg)

The method effectively combines Gaussian Process modelling with the likelihood ratio test outline in [Protassov et al. 2002](https://ui.adsabs.harvard.edu/abs/2002ApJ...571..545P/abstract). At present, the code uses the **celerite** kernels proposed by [Foreman-Mackey et al. 2017](https://iopscience.iop.org/article/10.3847/1538-3881/aa9332)


# Installation
To install the repository, first clone to your compute:
`git clone https://github.com/andresgur/mind_the_gaps`
then enter the `mind_the_gaps` main directory and pip install it by running:
 `pip install .`
# Warning
Users are welcome to use/test the code and provide feedback but beware, the code is still in alpha phase, so expect bugs, errors and crashes! If you are intending to use the code, my recommendation would be to get in touch first!

## Usage
The code can be used as a standalone to generate lightcurves or as it is desineg for, to test for a periodicity in a lightcurve. At a minimum your lightcurve should have timestamps, rates, uncertainties and exposures, and it can also contain background rates and uncertainties on the background.

Usually the first task is to identify the null hypothesis (a stochastic-only model) and the alternative model, which contains the periodic component (stochastic model + periodic component). To see how to go about model selection, see the notebook in [tutorials](https://github.com/andresgur/mind_the_gaps/tree/main/notebooks). Once you've established these two (note you can also have several sets of null hypothesis and alternative model) we follow the method outlined by [Protassov et al. 2002](https://ui.adsabs.harvard.edu/abs/2002ApJ...571..545P/abstract)
1. Fit the observed data using the two null and alternative models
2. Generate lightcurves from the posteriors of the stochastic model
3. Fit the generated lightcurves using both models
4. Build a histogram of the fit-improvements (LRT) found for the simulated lightcurves and check where the value observed in the data falls in the histogram. An outlier (say p<0.05) suggest the fit-improvement observed in the data is highly unlikely. High p-values indicate the fit-improvement is simply due to noise.

Another notebook in [tutorials](https://github.com/andresgur/mind_the_gaps/tree/main/notebooks) shows how to implement this process using functions and objects from the package.


# Tests
Tests are included in the `tests/` directory and can be run as:
```
python setup.py test
```
**Warning:** Tests currently are not passing. Please check the [issues](https://github.com/andresgur/mind_the_gaps/issues/13) on the respository to see the current status.

# Citation
```
@software{andres_gurpide_lasheras_2024_14253754,
  author       = {Andrés Gúrpide Lasheras and
                  Sam Mangham},
  title        = {{andresgur/mind\_the\_gaps: Release for paper}},
  month        = jan,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.2.0-beta},
  doi          = {10.5281/zenodo.14600069},
  url          = {https://doi.org/10.5281/zenodo.14600069}
}
```
