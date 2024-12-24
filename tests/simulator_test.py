# @Author: Andrés Gúrpide <agurpide>
# @Date:   28-03-2022
# @Email:  agurpidelash@irap.omp.eu
# @Last modified by:   agurpide
# @Last modified time: 28-03-2022
import unittest
from mind_the_gaps.models.psd_models import BendingPowerlaw, Lorentzian
from mind_the_gaps.simulator import *
from mind_the_gaps.fitting import fit_psd_powerlaw
from astropy.modeling.powerlaws import BrokenPowerLaw1D, PowerLaw1D, SmoothlyBrokenPowerLaw1D
import time
from scipy.stats import lognorm, rv_continuous
from mind_the_gaps.lightcurves import GappyLightcurve
from mind_the_gaps.models.psd_models import BendingPowerlaw
from scipy.optimize import minimize
from mind_the_gaps.fitting import s_statistic
#import matplotlib.pyplot as plt


class TestSimulator(unittest.TestCase):

    def power_spectrum(self, timestamps, rate):
        dt = np.mean(np.diff(timestamps))
        freqs = np.fft.rfftfreq(len(timestamps), dt)
        # For even fourier transforms we need to ignore the nyquist frequency (Vaughan +2005)
        if len(freqs) % 2 == 0:
            pow_spec = (np.absolute(np.fft.rfft(rate)[1:-1])) ** 2
            frequencies = freqs[1:-1]
        else:
            pow_spec = (np.absolute(np.fft.rfft(rate)[1:])) ** 2
            frequencies = freqs[1:]
        return frequencies, pow_spec

    def model_fit(self, params, freqs, powers):
        model = BendingPowerlaw(np.exp(params[0]), np.exp(params[1]))(freqs * 2* np.pi)
        S = s_statistic(powers, model)
        return S

    def test_slope_TK95(self):
        dt = 0.5
        points = 500
        timestamps = np.arange(0, points, dt) + dt/2
        input_beta = 1
        psd_model = PowerLaw1D(amplitude=1, alpha=input_beta)
        mean = 0.5
        start = time.time()
        rate = tk95_sim(timestamps, psd_model, mean, dt, 1.05, dt)
        end = time.time()
        print("Time to simulate a %d point TK lightcurve: %.2fs" % (len(timestamps), (end - start)))
        frequencies, pow_spec = self.power_spectrum(timestamps, rate)

        psd_slope, err, psd_norm, psd_norm_err = fit_psd_powerlaw(frequencies, pow_spec)
        self.assertAlmostEqual(-input_beta, psd_slope.value, None, "Slope of the power spectrum is not the same as the input at the 3 sigma level!", 3 * err)
        Nsims = 250
        rates = [tk95_sim(timestamps, psd_model, mean, dt, 1, dt) for n in np.arange(Nsims)]
        slopes = np.empty(len(rates))
        for index, rate in enumerate(rates):
            frequencies, pow_spec = self.power_spectrum(timestamps, rate)
            psd_slope, err, psd_norm, psd_norm_err = fit_psd_powerlaw(frequencies, pow_spec)
            slopes[index] = psd_slope.value
        err = 3 * np.abs(np.std(slopes))
        self.assertAlmostEqual(-input_beta, np.mean(slopes), None, "Average slope of %d lightcurve is not the same as the input!" % Nsims, err)

        # Increasing the binning of the simulated lightcurve
        rates = [tk95_sim(timestamps, psd_model, mean, dt, 2, dt) for n in np.arange(Nsims)]
        slopes = np.empty(len(rates))
        for index, rate in enumerate(rates):
            frequencies, pow_spec = self.power_spectrum(timestamps, rate)
            psd_slope, err, psd_norm, psd_norm_err = fit_psd_powerlaw(frequencies, pow_spec)
            slopes[index] = psd_slope.value
        err = np.abs(3 * np.std(slopes))
        #self.assertAlmostEqual(-input_beta, np.mean(slopes), None, "Average slope of %d lightcurve is not the same as the input!" % Nsims, err)

    def test_powerspec_bendingpowerlaw(self):
        dt = 1 # 1 day
        n = 1000
        times = np.arange(0, dt * n, dt) * 3600 * 24
        dummyrates = np.ones(len(times))
        exposures = 2000 * np.ones(len(times))
        lc = GappyLightcurve(times, dummyrates, np.ones(len(dummyrates)), exposures)
        variance = 162754.
        bendscale = 10. * 3600 * 24 # days
        omega0 = 2 * np.pi / bendscale
        psd_model = BendingPowerlaw(S0=variance, omega0=omega0)
        simu = lc.get_simulator(psd_model, "Gaussian")
        bnds = ((0.1, 60), (-18, np.log(1e-4)))
        # do 200 lightcurves and fit them
        n_sims = 200
        omegas = []
        for i in range(n_sims):
            lc_ = simu.simulator.prepare_segment(2)
            freqs, powers = self.power_spectrum(lc_.time, lc_.countrate)
            results = minimize(self.model_fit, [12, np.log(1e-6)], args=(freqs, powers),
                              bounds=bnds, method='L-BFGS-B')
            omegas.append(results.x[1])
        omegas = np.exp(omegas)
        self.assertAlmostEqual(np.mean(omegas), omega0, None, delta=np.std(omegas),
                               msg="Bend of the Bending Powerlaw does not match simulated periodograms!")

    def test_variance(self):
        """Test that the powerspectrum has the correct area after simulation"""
        psd_model = PowerLaw1D(amplitude=15, alpha=1)
        sim_dt = 0.1 * 3600 * 24
        bin_exposures = 0.8 * 3600 * 24
        timestamps = np.arange(0, 1000 * 3600 * 24, bin_exposures)
        lc = simulate_lightcurve(timestamps, psd_model,
                    dt=sim_dt, extension_factor=5)
        freqs = np.fft.rfftfreq(lc.n, sim_dt)
        pow_spec = (np.absolute(np.fft.rfft(lc.countrate)[1:])) ** 2
        frequencies = freqs[1:]
        pow_spec *= 2 * lc.dt / lc.meanrate**2 / lc.n # apply appropiate normalization
        integral = np.median(np.diff(frequencies)) * np.sum(pow_spec)
        rms = np.var(lc.countrate) / np.mean(lc.countrate)**2
        self.assertAlmostEqual(integral, rms, 1, msg="Area of the power spectrum is not equal to the RMS!")

    def test_variance_celerite(self):
        var = 10
        freq = 2 * np.pi / 10 # 1 / 10 in rad
        psd_model = BendingPowerlaw(S0=10, omega0=freq)
        sim_dt = 0.05
        timestamps = np.arange(0, 5000, sim_dt)
        extension_factor = 10
        vars = []

        for i in range(25):
            lc = simulate_lightcurve(timestamps, psd_model,
                            dt=sim_dt,
                             extension_factor=extension_factor) # this is just to get tseg for df
            vars.append(np.var(lc.countrate))
        self.assertAlmostEqual(var, np.mean(vars), delta=0.1, msg="Variance is not the same in TK95 method!")

        var = 10
        freq = 2 * np.pi / 10 # 1 / 10 in rad
        psd_model = Lorentzian(S0=var, omega0=freq, Q=10)
        sim_dt = 0.05
        timestamps = np.arange(0, 5000, sim_dt)
        extension_factor = 10
        vars = []

        for i in range(25):
            lc = simulate_lightcurve(timestamps, psd_model,
                            dt=sim_dt,
                             extension_factor=extension_factor) # this is just to get tseg for df
            vars.append(np.var(lc.countrate))

        self.assertAlmostEqual(var, np.mean(vars), delta=0.1, msg="Variance is not the same in TK95 method!")

    def test_slope_E13(self):

        dt = 0.5
        points = 500
        timestamps = np.arange(0, points, dt) + dt/2
        input_beta = 1
        psd_model = PowerLaw1D(amplitude=1, alpha=input_beta)
        mean, std  = 0.5, 0.1
        dist = norm(scale=std, loc=mean)
        start = time.time()
        erate = E13_sim(timestamps, psd_model, [dist], [1], dt, 1, dt)
        end = time.time()
        print("Time to simulate a %d point lightcurve with the E13 method: %.2f s \n" % (len(timestamps), (end - start)))
        frequencies, pow_spec = self.power_spectrum(timestamps, erate)
        self.assertEqual(len(timestamps), len(erate))
        psd_slope, err, psd_norm, psd_norm_err = fit_psd_powerlaw(frequencies, pow_spec)
        self.assertAlmostEqual(-input_beta, psd_slope.value, None, "Slope of the power spectrum is not the same as the input at the 3 sigma level!",
                                3 * err)

        Nsims = 250
        erates = [E13_sim(timestamps, psd_model, [dist], [1], dt, 1, dt) for n in np.arange(Nsims)]
        slopes = np.empty(Nsims)
        for index, rates in enumerate(erates):
            frequencies, pow_spec = self.power_spectrum(timestamps, rates)
            psd_slope, err, psd_norm, psd_norm_err = fit_psd_powerlaw(frequencies, pow_spec)
            slopes[index] = psd_slope.value
        err = 3 * np.abs(np.std(slopes))
        self.assertAlmostEqual(-input_beta, np.mean(slopes), None, "Average slope of %d lightcurve is not the same as the input!" % Nsims, err)
        # Increasing the binning of the simulated lightcurve
        erates = [E13_sim(timestamps, psd_model, [dist], [1], sim_dt=0.1, extension_factor=2, bin_exposures=dt) for n in np.arange(Nsims)]
        slopes = np.empty(Nsims)
        for index, rates in enumerate(erates):
            frequencies, pow_spec = self.power_spectrum(timestamps, rates)
            psd_slope, err, psd_norm, psd_norm_err = fit_psd_powerlaw(frequencies, pow_spec)
            slopes[index] = psd_slope.value
        err = np.abs(3 * np.std(slopes))
        self.assertAlmostEqual(-input_beta, np.mean(slopes), None, "Slope of the power spectrum is not the same as the input!", err)
        # test other method directly from TK95
        lcs = [simulate_lightcurve(timestamps, psd_model, dt, 1) for n in np.arange(Nsims)]
        erates = [E13_sim_TK95(lc, timestamps, [dist], [1]) for lc in lcs]
        slopes = np.empty(Nsims)
        for index, rates in enumerate(erates):
            frequencies, pow_spec = self.power_spectrum(timestamps, rates)
            psd_slope, err, psd_norm, psd_norm_err = fit_psd_powerlaw(frequencies, pow_spec)
            slopes[index] = psd_slope.value
        err = np.abs(3 * np.std(slopes))
        self.assertAlmostEqual(-input_beta, np.mean(slopes), None, "Slope of the power spectrum is not the same as the input!", err)


    def test_std_mean_TK95(self):
        dt = 1
        timestamps = np.arange(0, 8500, dt)
        variance = 10
        psd_model = BendingPowerlaw(S0=variance, omega0=np.exp(-3)) # 126 seconds
        mean  = 1
        vars = []
        for i in range(30):
            rate = tk95_sim(timestamps, psd_model, mean, dt, 1.01, dt) # this is just to get tseg for df
            vars.append(np.var(rate))

        self.assertAlmostEqual(variance, np.mean(vars), delta=0.5)
        self.assertAlmostEqual(mean, np.mean(rate), delta=0.5)


    def test_std_mean_E13(self):
        dt = 0.1
        timestamps = np.arange(0, 4000, dt) + dt/2
        input_beta = 1
        psd_model = PowerLaw1D(amplitude=1, alpha=input_beta)
        mean,std  = 0.5, 0.1
        var = std**2
        dist = norm(scale=std, loc=mean)
        rate_norm = E13_sim(timestamps, psd_model, [dist], [1], dt, 1, dt)
        self.assertAlmostEqual(std, np.std(rate_norm), 2, "Error standard deviation is not preserved for norm")
        self.assertAlmostEqual(mean, np.mean(rate_norm), 2, "Error mean deviation is not preserved for norm")
        # conver to lognormal params
        mu = np.log((mean**2)/np.sqrt(var+mean**2))
        sigma = np.sqrt(np.log(var/(mean**2)+1))
        dist = lognorm(sigma, scale=np.exp(mu))
        rate_lognorm = E13_sim(timestamps, psd_model, [dist], [1], dt, 1, dt)
        nbins = int(np.sqrt(len(rate_lognorm)))
        lognorm_fit, prefixes = fit_pdf(rate_lognorm, nbins)
        prefix = prefixes[0]
        std_fit = lognorm_fit.params.get("%ssigma" % prefix).value
        mean_fit = lognorm_fit.params.get("%scenter" % prefix).value
        self.assertAlmostEqual(np.std(rate_lognorm), std, 2, "Error distribution is not preserved for lognorm")
        self.assertAlmostEqual(np.mean(rate_lognorm), mean, 2, "Error distribution is not preserved for lognorm")


    def test_evenly_lc_duration(self):

        sim_dt = 0.01
        timestamps = np.arange(0, 10, sim_dt)
        mean, std  = 0.5, 0.1
        input_beta = 1
        psd_model = PowerLaw1D(amplitude=1, alpha=input_beta)
        lc = simulate_lightcurve(timestamps, psd_model, sim_dt, extension_factor=50)
        duration = timestamps[-1] - timestamps[0]
        lc_cut = cut_random_segment(lc, duration)
        self.assertAlmostEqual(lc_cut.tseg, duration, None, "Lightcurve duration is not preserved for regular timestamps!", sim_dt)


    def test_unevenly_lc_duration(self):
        sim_dt = 0.01
        timestamps = np.cumsum(np.random.exponential(0.4, 150))
        mean,std  = 0.5, 0.1
        input_beta = 1
        psd_model = PowerLaw1D(amplitude=1, alpha=input_beta)
        lc = simulate_lightcurve(timestamps, psd_model, sim_dt, extension_factor=50)
        duration = timestamps[-1] - timestamps[0]
        lc_cut = cut_random_segment(lc, duration)
        duration = timestamps[-1] - timestamps[0]
        self.assertAlmostEqual(lc_cut.tseg, duration, None, "Lightcurve duration is not preserved for irregular timestamps!",
                              np.median(np.diff(timestamps)) + sim_dt)



    def test_lc_binning(self):
        sim_dt = 0.01
        timestamps = np.cumsum(np.random.exponential(0.4, 150))
        mean, std  = 0.5, 0.1
        input_beta = 1
        psd_model = PowerLaw1D(amplitude=1, alpha=input_beta)
        lc = simulate_lightcurve(timestamps, psd_model, sim_dt, extension_factor=50)
        duration = timestamps[-1] - timestamps[0]
        lc_cut = cut_random_segment(lc, duration)
        self.assertAlmostEqual(lc_cut.dt, sim_dt, 3, "Lightcurve binning is not correct!")

        dt = 0.1
        timestamps = np.arange(0, 10, dt)
        sim_dt = 0.05
        lc = simulate_lightcurve(timestamps, psd_model, sim_dt, extension_factor=50)
        lc_cut = cut_random_segment(lc, duration)
        self.assertAlmostEqual(lc_cut.dt, sim_dt, 3, "Lightcurve binning is not correct!")


    def test_downsampling_basic(self):
        real_dt = 0.5
        times = np.arange(0, 10, real_dt) + real_dt
        rates = np.arange(len(times))
        new_dt = 2
        timestamps = np.arange(0, 5, new_dt) + new_dt
        lc = Lightcurve(times, rates, input_counts=False, skip_checks=True, dt=new_dt)
        downsampled_rates = imprint_sampling_pattern(lc, timestamps, new_dt)
        np.testing.assert_array_almost_equal(downsampled_rates, [3, 7, 11], 2)
        np.testing.assert_equal(len(timestamps), len(downsampled_rates))

        timestamps = np.arange(0, 8, new_dt) + new_dt
        timestamps = np.delete(timestamps, 2)
        lc = Lightcurve(times, rates, input_counts=False, dt=new_dt, skip_checks=True)
        downsampled_rates = imprint_sampling_pattern(lc, timestamps, new_dt)
        #import matplotlib.pyplot as plt
        #plt.errorbar(times, rates, xerr=real_dt / 2)
        #plt.errorbar(timestamps, downsampled_rates, xerr=new_dt / 2)
        #plt.show()
        np.testing.assert_array_almost_equal(downsampled_rates, [3, 7, 15], 2)
        np.testing.assert_equal(len(timestamps), len(downsampled_rates))

    def test_downsampling(self):
        length = 500
        sim_dt = 0.1
        sampling = 1.
        tk_timestamps = np.arange(0, length, sampling)
        model = PowerLaw1D(amplitude=10, alpha=1)
        tk_lc = simulate_lightcurve(tk_timestamps, model, sim_dt, 2)
        duration_sec = (tk_timestamps[-1] + sampling - (tk_timestamps[0] - sampling))
        tk_lc_cut = get_segment(tk_lc, duration_sec, 0)
        tk_rate = downsample(tk_lc_cut, tk_timestamps, sampling)
        indexstart = 6
        bins = int(sampling / sim_dt) - 2 # the oversampling minus the neighbouring bins
        for i in range(len(tk_rate)):
            print(i)
            self.assertAlmostEqual(tk_rate[i], np.mean(tk_lc_cut.countrate[indexstart + i:indexstart + bins]), delta=0.5, msg="Downsampling is not working!")
            indexstart += bins

    def test_bending_powerlaw(self):

        bin_exposures = 0.8 / 3600 / 24 # 0.8 days
        sim_dt = 0.1 / 3600 / 24 # 0.1 days
        timestamps = np.arange(0, (2000 / 3600 / 24), bin_exposures)
        bend = 1 / 100 #days
        psd_model = SmoothlyBrokenPowerLaw1D(amplitude=1, alpha_1=0, alpha_2=2, x_break=bend, delta=1/2)
        mean, sigma = np.exp(-2.82), 0.39
        pdfs = [lognorm(sigma, scale=mean)]
        weights = [1]
        erates = E13_sim(timestamps, psd_model, pdfs, weights, sim_dt=sim_dt,
                         extension_factor=2, max_iter=1000, bin_exposures=bin_exposures)
        self.assertAlmostEqual(np.mean(erates), mean, delta=sigma, msg="E13 not working with SmoothlyBrokenPowerLaw1D (mean)")
        self.assertAlmostEqual(np.std(erates), sigma, delta=sigma, msg="E13 not working with SmoothlyBrokenPowerLaw1D (sigma)")


if __name__ == '__main__':
    unittest.main()
