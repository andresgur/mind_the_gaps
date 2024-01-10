# Workflow

This is the typical workflow for fitting a model using **Mind the Gaps**, illustrated [in a flowchart here](workflow.drawio.png).

## Inputs

Three main input files are required, a lightcurve, and model files for the 'null hypothesis' model to describe it and the alternative model proposed.

### Lightcurve
The pipeline takes input in the form of tab-separated text files with 4 or more columns, named:

| **Field**          | **Format** | **Function**                                 |
| ------------------ | -----------| -------------------------------------------- |
| timestamps         | float      | Times of observation, in fractional days     |
| rates              | float      | Count rates, fluxes, or magnitudes           |
| uncertainties      | float      | Error on the rate value                      |
| exposure           | float      | Exposure time, in fractional days (?)        |
| bkg_counts         | int        | Background count rate                        |
| bkg_counts_err     | int        | Error on the background count rate           |


### Celerite configs
Celerite configuration files for the null hypothesis and alternative models for the lightcurve, using the examples in the [Celerite config directory](../celerite_config).

## Process

1. Use `celerite_script.py` to perform an MCMC fit to the input lightcurve using the two models:
   ```
   celerite_script.py input_lightcurve.dat -c null_model.config -o output --fit
   celerite_script.py input_lightcurve.dat -c alternative_model.config -o output --fit
   ```
   Where `-o` adds a component to the output file name, and `-fit` runs a minimisation routine before
   This will create two directories, named:
   ```
   celerite_output_t<start>-<end>_m_<null_model_components>_fit/
   celerite_output_t<start>-<end>_m_<alternative_model_components>_fit/
   ```
   Where `t<start>-<end>` is the start and end times of the fitted section of lightcurve. Subsets of the curve can be fitted using the `--tmin` and `--tmax` arguments.

   `<null_model_components>` and `<alternative_model_components>` are the components listed in the
   config files for each, separated by underscores, e.g. `Lorentzian_DampedRandomWalk`.

   Each folder includes:
   * `best_fit.dat`: The best fit parameters for that models for the observed lightcurve.
   * `samples.dat`: A set of the MCMC parameter samples used, after the burn-in period has finished. These samples are thinned, and not a full set.

2. Use `generate_lcs_significance.py` to generate a set of synthetic lightcurves based on the fits for the null hypothesis model:
   ```
   generate_lcs_significance.py --input_dir celerite_output_t<start>-<end>_m_<null_model_components>_fit -c <num_processes> -n <num_lightcurves> --pdf Lognormal -f input_lightcurve.dat
   ```
   This script is parallelised, and `-c` sets the number of processes to spawn, with `-n` setting the number of simulated lightcurves to generate, 5000+ being recommended.

   `--pdf` sets the flux distribution PDF for the synthetic lightcurves, and can be `Gaussian` or `Lognormal`.
   The distribution's mean is derived from the input lightcurve (provided with `-f`) and the standard deviation is a function of the model used.

   This will create a directory, named:
   ```
   lightcurves_significance_p<?>-<?>_t<start>-<end>s_n<num_lightcurves>_s<??>_pdf<pdf_type>/
   ```
   This folder contains:
   * `lightcurves/`: The synthetic lightcurves, in the same format as the input lightcurve.


3. Use `fit_lcs.py` to then calculate the fit of these synthetic lightcurves to the two models (the null hypothesis model, used to generate them, and the alternative model):
   ```
   fit_lcs.py lightcurves_significance_p<?>-<?>_t<start>-<end>s_n<num_lightcurves>_s<??>_pdf<pdf_type>/lightcurves/*.dat -c null_model.config -o sims --cores <num_processes>
   fit_lcs.py lightcurves_significance_p<?>-<?>_t<start>-<end>s_n<num_lightcurves>_s<??>_pdf<pdf_type>/lightcurves/*.dat -c alternative_model.config -o sims --cores <num_processes>
   ```
   Where `-o` adds a component to the output directory. As before, this script is parallelised and `--cores` sets the number of processes to spawn.

   This will create two directories, named:
   ```
   fit_lcs_sims_m_<null_model_components>/
   fit_lcs_sims_m_<alternative_model_components>/
   ```
   Each folder contains:
   * `fit_results.dat`: For each synthetic lightcurve, the log-likelihoods of the model.

4. Use `fit_lcs.py` to also calculate the fit of the two models to the real lightcurve:
   ```
   fit_lcs.py input_lightcurve.dat -c null_model.config -o data
   fit_lcs.py input_lightcurve.dat -c alternative_model.config -o data
   ```
   This has already been done in step 1 but we repeat for convenience.

   This will create two directories, named:
   ```
   fit_lcs_data_m_<null_model_components>/
   fit_lcs_data_m_<alternative_model_components>/
   ```
   As above, each folder contains:
   * `fit_results.dat`: The log-likelihoods of the model fit to that lightcurve.

   The fits should be better in the alternative model directory, as an additional component has been added to the model.

5. Use `plot_ratio_test.py` to test the hypothesis:
   ```
   plot_ratio_test.py -n fit_lcs_sims_m_<null_model_components>/fits_results.dat -a fit_lcs_sims_m_<alternative_model_components>/fits_results.dat -dn fit_lcs_data_m_<null_model_components>/fits_results.dat -da fit_results_data_m_<alternative_model_components>/fits_results.dat
   ```
   The output shows the improvement in fit caused by using the alternative hypothesis model on the synthetic lightcurves (generated purely using the null hypothesis model, so improvement is due to overfitting), as compared to the improvement in fit caused by using the alternative hypothesis model in the *observed* lightcurves. 
   
   There should be a much larger gain for applying the alternative hypothesis model to the real lightcurve, than for applying it to the synthetic models, and based on this the probability of rejecting the null hypothesis is provided.
    