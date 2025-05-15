# @Author: Andrés Gúrpide <agurpide>
# @Date:   28-03-2022
# @Email:  agl1f22@soton.ac.uk
# @Last modified by:   agurpide
# @Last modified time: 07-05-2025
import unittest
from mind_the_gaps.models.psd_models import BendingPowerlaw, Lorentzian
from mind_the_gaps.simulator import *
from mind_the_gaps.fitting import fit_psd_powerlaw
from astropy.modeling.powerlaws import BrokenPowerLaw1D, PowerLaw1D, SmoothlyBrokenPowerLaw1D
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import fit, lognorm, uniform, norm
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
        model = BendingPowerlaw(params[0], params[1])(freqs)
        S = s_statistic(powers, model)
        return S

    def test_slope_TK95(self):
        """
        Test the slope of the PSD is correct by simualting an ensemble of lightcurves
        and checking the mean slop is consistent with the input slope value.
        """
        dt = 0.5
        points = 500
        timestamps = np.arange(0, points, dt) + dt/2
        input_beta = 1
        psd_model = PowerLaw1D(amplitude=1, alpha=input_beta)
        start = time.time()
        simu = Simulator(psd_model, timestamps, dt, 0, aliasing_factor=1, extension_factor=1.05)
        rate = simu.generate_lightcurve()#tk95_sim(timestamps, psd_model, mean, dt, 1.05, dt)
        end = time.time()
        print("Time to simulate a %d point TK lightcurve: %.2fs" % (len(timestamps), (end - start)))
        frequencies, pow_spec = self.power_spectrum(timestamps, rate)

        psd_slope, err, _, _ = fit_psd_powerlaw(frequencies, pow_spec)
        self.assertAlmostEqual(-input_beta, psd_slope.value, None, "Slope of the power spectrum is not the same as the input at the 3 sigma level!", err)
        Nsims = 250
        slopes = np.empty(Nsims)
        for index in np.arange(Nsims):
            rate = simu.generate_lightcurve()#tk95_sim(timestamps, psd_model, mean, dt, 1, dt)
            frequencies, pow_spec = self.power_spectrum(timestamps, rate)
            psd_slope, err, _, _ = fit_psd_powerlaw(frequencies, pow_spec)
            slopes[index] = psd_slope.value
        err = np.abs(np.std(slopes))
        self.assertAlmostEqual(-input_beta, np.mean(slopes), None, "Average slope of %d lightcurve is not the same as the input!" % Nsims, err)


    def test_slope_E13(self):

        dt = 0.5
        points = 500
        timestamps = np.arange(0, points, dt) + dt/2
        input_beta = 1
        input_mean = 100
        psd_model = PowerLaw1D(amplitude=1, alpha=input_beta)
        start = time.time()
        simulator = Simulator(psd_model, timestamps, dt, input_mean, "Lognormal", extension_factor=1.05, aliasing_factor=1)
        rates = simulator.generate_lightcurve()
        end = time.time()
        print("Time to simulate a %d point lightcurve with the E13 method: %.2f s \n" % (len(timestamps), (end - start)))

        Nsims = 250
        slopes = np.empty(Nsims)
        means = np.empty(Nsims)

        for index in range(Nsims):
            rate = simulator.generate_lightcurve()
            frequencies, pow_spec = self.power_spectrum(timestamps, rate)
            psd_slope, err, psd_norm, psd_norm_err = fit_psd_powerlaw(frequencies, pow_spec)
            slopes[index] = psd_slope.value
            means[index] = np.mean(rate)
        err = 3 * np.abs(np.std(slopes))
        self.assertAlmostEqual(-input_beta, np.mean(slopes), None, "Average slope of %d lightcurve is not the same as the input!" % Nsims, err)
        err = 3 * np.abs(np.std(means))
        self.assertAlmostEqual(input_mean, np.mean(means), None, "Average mean count rate of %d lightcurve is not the same as the input!" % Nsims, err)


    def test_powerspec_bendingpowerlaw_TK95(self):
        dt = 1 # 1 day
        n = 1000
        times = np.arange(0.5, dt * n, dt)
        exposures = 0.2
        variance = 100.
        bendscale = 20 # days
        omega0 = 2 * np.pi / bendscale
        psd_model = BendingPowerlaw(S0=variance, omega0=omega0)
        simu = Simulator(psd_model, times, exposures, 10, "Gaussian", extension_factor=1.0, aliasing_factor=2)
        bnds = ((1e-5, 1e5), (omega0 / 100, omega0 * 100))
        # do 200 lightcurves and fit them
        n_sims = 200
        omegas = []
        for i in range(n_sims):
            rates = simu.generate_lightcurve()
            freqs, powers = self.power_spectrum(times, rates)
            results = minimize(self.model_fit, [variance, 1/ bendscale], args=(freqs, powers),
                              bounds=bnds, method='L-BFGS-B')
            omegas.append(results.x[1] * 2 * np.pi)
        self.assertAlmostEqual(np.mean(omegas), omega0, None, delta=np.std(omegas),
                               msg="Bend of the Bending Powerlaw does not match simulated periodograms!")
 
    def test_powerspec_bendingpowerlaw_E13(self):
        dt = 1 # 1 day
        n = 1000
        times = np.arange(0.5, dt * n, dt)
        exposures = 0.2
        variance = 100.
        bendscale = 20 # days
        omega0 = 2 * np.pi / bendscale
        psd_model = BendingPowerlaw(S0=variance, omega0=omega0)
        simu = Simulator(psd_model, times, exposures, 10, "Lognormal", extension_factor=1.0, aliasing_factor=2, 
                         max_iter=600)
        bnds = ((1e-5, 1e5), (omega0 / 100, omega0 * 100))
        # do 200 lightcurves and fit them
        n_sims = 250
        omegas = []
        for i in range(n_sims):
            rates = simu.generate_lightcurve()
            freqs, powers = self.power_spectrum(times, rates)
            results = minimize(self.model_fit, [variance, 1/ bendscale], args=(freqs, powers),
                              bounds=bnds, method='L-BFGS-B')
            omegas.append(results.x[1] * 2 * np.pi)
        self.assertAlmostEqual(np.mean(omegas), omega0, None, delta=np.std(omegas),
                               msg="Bend of the Bending Powerlaw does not match simulated periodograms!")


    def test_powerspectrum_normalization(self):
        """Test that the powerspectrum has the correct area after simulation"""
        psd_model = PowerLaw1D(amplitude=1e-10, alpha=1)
        exposures = 0.8
        times = np.arange(0, 1000, exposures)
        mean = 10000
        simu = Simulator(psd_model, times, exposures, mean, "Gaussian", extension_factor=1.05, aliasing_factor=8)
        lc = simu.simulate_regularly_sampled()
        freqs = np.fft.rfftfreq(lc.n, lc.dt)
        pow_spec = (np.absolute(np.fft.rfft(lc.countrate)[1:])) ** 2
        frequencies = freqs[1:]
        pow_spec *= 2 * lc.dt / np.mean(lc.countrate)**2 / lc.n # apply appropiate normalization
        integral = np.median(np.diff(frequencies)) * np.sum(pow_spec)
        rms = np.var(lc.countrate) / np.mean(lc.countrate)**2
        self.assertAlmostEqual(integral, rms, delta=0.1, msg="Area of the power spectrum is not equal to the RMS!")



    def test_std_mean_and_variance_TK95(self):
        dt = 1
        timestamps = np.arange(0, 8500, dt)
        variance = 10
        psd_model = BendingPowerlaw(S0=variance, omega0=np.exp(-3)) # 126 seconds
        mean  = 1
        simu = Simulator(psd_model, timestamps, dt, mean, "Gaussian", extension_factor=1.05, aliasing_factor=1)
        vars = []
        means = []
        for i in range(100):
            rate = simu.generate_lightcurve()
            means.append(np.mean(rate))
            vars.append(np.var(rate))

        self.assertAlmostEqual(variance, np.mean(vars), delta=np.std(vars))
        self.assertAlmostEqual(mean, np.mean(means), delta=np.std(means))



    def test_std_mean_and_variance_E13(self):
        dt = 1
        timestamps = np.arange(0, 8500, dt)
        variance = 10
        psd_model = BendingPowerlaw(S0=variance, omega0=np.exp(-3)) # 126 seconds
        mean  = 10
        simu = Simulator(psd_model, timestamps, dt, mean, "Lognormal", extension_factor=1.05, aliasing_factor=1, max_iter=600)
        vars = []
        means = []
        for i in range(150):
            rate = simu.generate_lightcurve()
            means.append(np.mean(rate))
            vars.append(np.var(rate))

        self.assertAlmostEqual(variance, np.mean(vars), delta=np.std(vars))
        self.assertAlmostEqual(mean, np.mean(means), delta=np.std(means))


    def test_downsampling_1(self):

        np.random.seed(20)
        timestamps = np.append(np.arange(1, 3.1, 2), np.arange(5, 7.1, 2))
        exposures = 0.5
        dt = 0.1
        times = np.arange(0.5, 10.1, dt)
        counts = np.linspace(5, 20, len(times))
        countrates = counts / exposures

        lc = Lightcurve(times, countrates, input_counts=False)
        psd_model = PowerLaw1D(amplitude=10, alpha=2)
        simu = Simulator(psd_model, timestamps, exposures, 0, extension_factor=1.0, aliasing_factor=1)
        idxstrue = [[3, 4, 5, 6, 7], [23, 24, 25, 26, 27], [43, 44, 45, 46, 47], [63, 64, 65, 66, 67]]
        truerates = [ ]
        for idxtrue in idxstrue:
            truerates.append(np.mean(countrates[idxtrue[0]:idxtrue[-1] + 1]))

        downsampled_rates = simu.downsample(lc)
        self.assertListEqual(truerates, downsampled_rates, msg="Downsampling does not work!")

    
    def test_downsampling_2(self):

        np.random.seed(20)
        timestamps = np.append(np.arange(1, 3.1, 2), np.arange(5, 7.1, 2))
        exposures = 0.6
        dt = 0.1
        times = np.arange(0.5, 10.1, dt)
        counts = np.linspace(5, 20, len(times))
        countrates = counts / exposures

        lc = Lightcurve(times, countrates, input_counts=False)
        psd_model = PowerLaw1D(amplitude=10, alpha=2)
        simu = Simulator(psd_model, timestamps, exposures, 0, extension_factor=1.0, aliasing_factor=1)
        idxstrue = [[2, 3, 4, 5, 6, 7, 8], [22, 23, 24, 25, 26, 27, 28], [42, 43, 44, 45, 46, 47, 48], [62, 63, 64, 65, 66, 67, 68]]
        truerates = [ ]
        for idxtrue in idxstrue:
            truerates.append(np.mean(countrates[idxtrue[0]:idxtrue[-1] + 1]))

        downsampled_rates = simu.downsample(lc)
        self.assertListEqual(truerates, downsampled_rates, msg="Downsampling does not work!")

    def test_downsampling_3(self):
        timestamps = np.append(np.arange(1, 3.1, 2), np.arange(5, 7.1, 2))
        exposures = 0.1
        dt = 0.1
        times = np.arange(0.5, 10.1, dt)
        counts = np.linspace(5, 20, len(times))
        countrates = counts / exposures

        lc = Lightcurve(times, countrates, input_counts=False)
        psd_model = PowerLaw1D(amplitude=10, alpha=2)
        simu = Simulator(psd_model, timestamps, exposures, 0, extension_factor=1.0, aliasing_factor=1)

        idxstrue = [[5], [25], [45], [65]]
        truerates = [ ]
        for idxtrue in idxstrue:
            truerates.append(np.mean(countrates[idxtrue[0]:idxtrue[-1] + 1]))

        downsampled_rates = simu.downsample(lc)
        self.assertListEqual(truerates, downsampled_rates, msg="Downsampling does not work!")

    def test_evenly_lc_duration(self):

        sim_dts = np.arange(0.01, 0.1, 1)
        input_beta = 1
        mean  = 0.5
        psd_model = PowerLaw1D(amplitude=1, alpha=input_beta)
        for sim_dt in sim_dts:
            timestamps = np.arange(0, 10, sim_dt)
            simu = Simulator(psd_model, timestamps, sim_dt, mean, extension_factor=50)
            lc = simu.simulate_regularly_sampled()
            duration = timestamps[-1] - timestamps[0]
            lc_cut = cut_random_segment(lc, duration)
            # Calculate duration manually as stingray does not work
            duration_cut = (lc_cut.time[-1] - lc_cut.dt /2) - (lc_cut.time[0] + lc_cut.dt / 2)
            self.assertAlmostEqual(duration_cut, duration, None, "Lightcurve duration is not preserved for regular timestamps!", sim_dt)


    def test_unevenly_lc_duration(self):
        
        exposures = 0.1
        timestamps = np.arange(1, 3.1, 1)
        mean  = 50
        input_beta = 1
        psd_model = PowerLaw1D(amplitude=1, alpha=input_beta)
        simu = Simulator(psd_model, timestamps, exposures, mean, extension_factor=50)
        lc = simu.simulate_regularly_sampled()
        duration = timestamps[-1] - timestamps[0]
        lc_cut = cut_random_segment(lc, duration)
        duration_cut = (lc_cut.time[-1] - lc_cut.dt /2) - (lc_cut.time[0] + lc_cut.dt / 2)
        self.assertAlmostEqual(lc_cut.tseg, duration_cut, None, "Lightcurve duration is not preserved for irregular timestamps!",
                            np.median(np.diff(timestamps)))

    def test_lc_sampling(self):
        """
        Check that the lightcurve sampling is correct after cutting a random segment 
        for various expoure times
        """
        input_beta = 1
        mean  = 0.5
        exposures = [0.01, 0.1, 1]
        psd_model = PowerLaw1D(amplitude=1, alpha=input_beta)

        for dt in exposures:
            timestamps = np.arange(0, 10, dt)
            simu = Simulator(psd_model, timestamps, dt, mean, extension_factor=50, 
                             aliasing_factor=1)
            lc = simu.simulate_regularly_sampled()
            duration = timestamps[-1] - timestamps[0]
            lc_cut = cut_random_segment(lc, duration)
            self.assertEqual(lc_cut.dt, dt, "Lightcurve binning is not correct!")
           
class TestRegularlySampledBendingPowerlaw(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(100)
        self.variance = 1.
        bendscale = 200
        omega0 = 2 * np.pi / bendscale
        exposures = 0.2
        times = np.arange(0, 20000, exposures)
        self.inputmean = 100
        psd_model = BendingPowerlaw(S0=self.variance, omega0=omega0)
        
        simu = Simulator(psd_model, times, exposures, self.inputmean, "Gaussian", extension_factor=1.05, aliasing_factor=1)
        means = []
        variances = []
        for i in range(100):
            lc = simu.simulate_regularly_sampled()
            means.append(np.mean(lc.countrate))
            variances.append(np.var(lc.countrate))
        self.outputmean = np.mean(means)
        self.outputvariance = np.mean(variances)

    def test_mean(self):
        """
        Test that the simulated lightcurve has the correct mean count rate.

        Over multiple simulations, the average of the mean countrates should 
        converge to the input value (here, 0), within expected statistical variation. 
        The tolerance is set to one-third of the sample standard deviation.
        """
        self.assertAlmostEqual(self.outputmean, self.inputmean, delta=0.01, msg="Mean is not right in regularly sampled lightcurve!")

    def test_variance(self):
        self.assertAlmostEqual(self.outputvariance, self.variance, delta=0.02, msg="Variance is not right in regularly sampled lightcurve!")

class TestRegularlySampledLorentzian(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        np.random.seed(100)
        self.variance = 1.
        bendscale = 200
        omega0 = 2 * np.pi / bendscale
        exposures = 0.2
        times = np.arange(0, 50000, exposures)
        self.inputmean = 100
        psd_model = Lorentzian(S0=self.variance, omega0=omega0, Q=10)
        simu = Simulator(psd_model, times, exposures, self.inputmean, "Gaussian", extension_factor=1.05, aliasing_factor=1)
        means = []
        variances = []
        for i in range(100):
            lc = simu.simulate_regularly_sampled()
            means.append(np.mean(lc.countrate))
            variances.append(np.var(lc.countrate))
        self.outputmean = np.mean(means)
        self.outputvariance = np.mean(variances)

    def test_mean(self):
        """
        Test that the simulated lightcurve has the correct mean count rate.

        Over multiple simulations, the average of the mean countrates should 
        converge to the input value (here, 0), within expected statistical variation. 
        The tolerance is set to one-third of the sample standard deviation.
        """
        self.assertAlmostEqual(self.outputmean, self.inputmean, delta=0.01, msg="Mean is not right in regularly sampled lightcurve!")

    def test_variance(self):
        self.assertAlmostEqual(self.outputvariance, self.variance, delta=0.02, msg="Variance is not right in regularly sampled lightcurve!")

class TestPDF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12)
        cls.dt = 1
        cls.timestamps = np.arange(0, 2000000, cls.dt)
        bend = 1000 #
        omega = 2 * np.pi / bend
        cls.psd_model = BendingPowerlaw(S0=10, omega0=omega)
        cls.max_iter = 600
        cls.inputmean = 10
        cls.extension_factor = 1.05

    def get_segment_var(self, simulator):
        lc = simulator.simulate_regularly_sampled()
        segment = cut_random_segment(lc, simulator.sim_duration)
        sample_std = np.std(segment.countrate)
        pdf = simulator.simulator.pdfmethod(simulator.mean, sample_std)
        adjusted_lc = simulator.simulator.adjust_lightcurve_pdf(segment, pdf, max_iter=simulator.simulator.max_iter)
        input_var = sample_std**2
        return adjusted_lc.countrate, input_var
        

    def test_pdf_lognormal(self):
        pdf_type = "Lognormal"
        simulator = Simulator(self.psd_model, self.timestamps, self.dt, self.inputmean, pdf_type, extension_factor=self.extension_factor, 
                        aliasing_factor=1, max_iter=self.max_iter)
        adjusted_counrate, inputvar = self.get_segment_var(simulator)
        inputmu = np.log((self.inputmean**2) / np.sqrt(inputvar + self.inputmean**2))
        inputsigma = np.sqrt(np.log(inputvar/(self.inputmean**2) + 1))
        fitresult = fit(lognorm, adjusted_counrate,  
                            guess={"loc":0, "s":inputsigma, "scale": np.exp(inputmu)}, 
                            bounds={"loc":[0,0], "s":[0, 1e4], "scale":[1e-3, 1e4]})
        s, loc, scale = fitresult.params
        mu = np.log(scale)
        mean = np.exp(mu + s**2/2)
        variance = (np.exp(s**2) - 1) * np.exp(2. * mu + s**2)
        self.assertAlmostEqual(mean, self.inputmean, delta=0.1, msg=f"Mean in {pdf_type} lightcurves is not correct!")
        self.assertAlmostEqual(variance, inputvar, delta=0.1, msg=f"Variance in {pdf_type} lightcurves is not correct!")


    def test_pdf_uniform(self):
        pdf_type = "Uniform"
        simu = Simulator(self.psd_model, self.timestamps, self.dt, self.inputmean, pdf_type, extension_factor=self.extension_factor, 
                        aliasing_factor=1, max_iter=self.max_iter)
        
        adjusted_counrate, inputvar = self.get_segment_var(simu)
        inputb = np.sqrt(3 * inputvar) + self.inputmean
        inputa = 2 * self.inputmean - inputb
        fitresult = fit(uniform, adjusted_counrate,  
                            guess={"loc":inputa, "scale": inputb - inputa}, 
                            bounds={"loc":[0, 1e3], "scale":[1e-3, 1e4]})
        loc, scale = fitresult.params
        a =  loc
        b = loc + scale
        mean = 0.5 * (a + b)
        variance = 1 / 12 * (b-a)**2
        self.assertAlmostEqual(mean, self.inputmean, delta=0.01, msg="Mean in {pdf_type} lightcurves is not correct!")
        self.assertAlmostEqual(variance, inputvar, delta=0.1, msg="Variance in {pdf_type} lightcurves is not correct!")

    def test_pdf_gaussian(self):
        pdf_type = "Gaussian"
        simulator = Simulator(self.psd_model, self.timestamps, self.dt, self.inputmean, pdf_type, extension_factor=self.extension_factor, 
                        aliasing_factor=1, max_iter=self.max_iter)
        
        lc = simulator.simulate_regularly_sampled()
        segment = cut_random_segment(lc, simulator.sim_duration)
        inputvar = np.var(segment.countrate)
        adjusted_countrates = simulator.simulator.adjust_pdf(segment) # this method doesn't do anything in principle
        fitresult = fit(norm, adjusted_countrates,  
                            guess={"loc":self.inputmean, "scale": np.sqrt(inputvar)}, 
                            bounds={"loc":[0, 1e3], "scale":[1e-3, 1e4]})
        loc, scale = fitresult.params
        mean = loc
        variance = scale**2
        self.assertAlmostEqual(mean, self.inputmean, delta=0.01, msg="Mean in {pdf_type} lightcurves is not correct!")
        self.assertAlmostEqual(variance, inputvar, delta=0.01, msg="Variance in {pdf_type} lightcurves is not correct!")


if __name__ == '__main__':
    unittest.main()
