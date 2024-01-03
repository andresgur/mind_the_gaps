GP time series analysis with focus on period detection on irregularly-sampled stochastically-varying astronomical time series

# Installation
To install the repository, first clone to your compute:
`git clone https://github.com/andresgur/mind_the_gaps`
then enter the `mind_the_gaps` main directory and pip install it by running:
 `pip install .`


# Scripts
Several scripts exists that allow to perform the main tasks. Each of them has a set of input parameters, for help run
`python <script> -h` from the terminal and you'll be presented with the input parameters and their definition.

celerite_script.py --> Main script to fit a lightcurve and derive posteriors using celerite. Covariance functions are input using a config file. The mean function can be (for now) a constant function or a Linear function. The MCMC chain convergence is evaluated on the fly. A config file is needed with the parameter ranges (see config_file folder for examples).

fit_lcs.py --> This is a lighter and faster version of the above and is meant to be used to fit the simulated lightcurves, when deriving the full posteriors is not necessary (as we are only looking for the maximum of the loglikehood). 

generate_lc.py --> Generates a lightcurve based on an existing observing window (-f file option) using the  [TK95](https://ui.adsabs.harvard.edu/abs/1995A&A...300..707T) method (when specified or PDF is Gaussian) or the [E13](https://academic.oup.com/mnras/article/433/2/907/1746942) method. The PDF is specified through the --pdf command (only Gaussian, Uniform or Lognormal are accepted so far). The lightcurve input file needs to have 6 columns: timestamps, rates, errors, exposures, bkg count rates and uncertainties on the background rates. The model can be specified through the command line or through a config file. 
