GP time series analysis with focus on period detection on irregularly-sampled stochastically-varying astronomical time series

![Mind The Gaps](https://github.com/andresgur/mind_the_gaps/blob/main/docs/mind_the_gaps.jpg)


# Installation
To install the repository, first clone to your compute:
`git clone https://github.com/andresgur/mind_the_gaps`
then enter the `mind_the_gaps` main directory and pip install it by running:
 `pip install .`
# Warning
Users are welcome to use/test the code and provide feedback but beware, the code is still in alpha phase, so expect bugs, errors and crashes! If you are intending to use the code, but recommendation would be to get in touch first!

# Scripts
Several scripts exists that allow to perform the main tasks. Each of them has a set of input parameters, for help running a given script type
`python <script> -h` from the terminal and you'll be presented with the input parameters and their definition.

celerite_script.py --> Main script to fit an observed lightcurve and derive posteriors using celerite. Covariance functions are input using a config file (see examples in the config_file folder). The config sets the parameter ranges (i.e. Uniform priors) are starting values for the fit. The mean function can be (for now) a Constant, Linear or Gaussian function. The MCMC chain convergence is evaluated on the fly. 

fit_lcs.py --> This is a lighter and faster version of the above and is meant to be used to fit the simulated lightcurves, when deriving the full posteriors is not necessary (as we are only looking for the maximum of the loglikehood). 

generate_lcs.py --> Generates lightcurves based on an existing observing window (-f file option) using the [TK95](https://ui.adsabs.harvard.edu/abs/1995A&A...300..707T) method (when PDF is Gaussian) or the [E13](https://academic.oup.com/mnras/article/433/2/907/1746942) method when the PDF is not Gaussian. The PDF is specified through the --pdf command (only Gaussian, Uniform or Lognormal are accepted so far). The lightcurve input file needs to have at least 4 columns: timestamps, rates, errors, exposures, and optionally, bkg count rates and uncertainties on the background rates. The model can be specified through the command line or through a config file. 

generate_lcs_significance.py --> This takes the posteriors from a celerite_script.py run (from the input folder) and generates N lightcurves sampling from them. As generate_lcs.py, the pdf and observing strategy can be specified.

## Workflow

An example workflow is included in [docs/workflow.md](docs/workflow.md).

# Tests
Tests are included in the `tests/` directory and can be run as:
```
python setup.py test
```
**Warning:** Tests currently are not passing. Please check the [issues](https://github.com/andresgur/mind_the_gaps/issues/13) on the respository to see the current status.
