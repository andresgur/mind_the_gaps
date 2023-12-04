GP time series analysis with focus on period detection on irregularly-sampled stochastically-varying astronomical time series

# Installation
move the setup.py outside the mind_the_gaps folder and run `pip install .`


# Scripts
A bunch of scripts exists that allow to perform the main tasks. Each of them has a set of input parameters, for help run
`python <script> -h` from the terminal and you'll be presented with the input parameters and their definition.

celerite_script.py --> Main script to fit a lightcurve and derive posteriors using celerite. Covariance functions are input using a config file. The mean function can be (for now) a constant function or a Linear function. The MCMC chain convergence is evaluated on the fly. A config file is needed with the parameter ranges (see config_file folder for examples).


generate_lc.py --> Generates a lightcurve based on an existing observing window. The model is specifiec
