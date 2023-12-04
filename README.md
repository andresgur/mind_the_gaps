GP time series analysis with focus on period detection on irregularly-sampled stochastically-varying astronomical time series

# Installation
move the setup.py outside the mind_the_gaps folder and run `pip install .`


# Scripts

celerite_script.py --> Main script to fit a lightcurve and derive posteriors using celerite. Covariance functions are input using a config file. The mean function can be (for now) a constant function or a Linear function. The MCMC chain convergence is evaluated on the fly.


