"""
This file contains useful python functions meant to simplify the Jupyter notebook.
AnalysisHandle
written by Mo Chen in Oct. 2023
"""
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface, generate_qua_script,QuantumMachinesManager
from qm.octave import *
from configuration import *
from scipy import signal
from qm.octave import QmOctaveConfig
from quam import QuAM
from scipy.io import savemat
from scipy.io import loadmat
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter
#from qutip import *
from typing import Union
import datetime
import os
import time
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
import math
import lmfit

class AH_exp1D:
	"""
	Class for analysis of 1D experiments
	Attributes:

	Methods (useful ones):
		time_of_flight(self, adc1, adc2, adc1_single_run, adc2_single_run)
		rr_freq(self, res_freq_sweep, sig_amp)
	"""

	def __init__(self, ref_to_ham_param, ref_to_poly_param, ref_to_json_name):
		self.ham_param = ref_to_ham_param
		self.poly_param = ref_to_poly_param
		self.json_name = ref_to_json_name


	def time_of_flight(self, expt_dataset):
		"""
		Analyze time of flight experiment.
		
		Plot the experiment result, both single run, to check if signal level falls within +-0.5V, the range for ADC
		Does some rudimentary analysis, like the dc offsets for I and Q. TOF is subject to scrutiny.
		
		Args:
			adc1: average adc voltage value from I channel
			adc2: average adc voltage value from Q channel
			adc1_single_run: adc voltage value from I channel, single run
			adc2_single_run: adc voltage value from Q channel, single run
		
		Returns:
			dc_offset_i: to add to machine.global_parameters.con1_downconversion_offset_I
			dc_offset_q: to add to machine.global_parameters.con1_downconversion_offset_Q
			delay: to add to machine.global_parameters.time_of_flight
		"""
		
		adc1_mean = np.mean(expt_dataset.I).values
		adc2_mean = np.mean(expt_dataset.Q).values
		adc1_unbiased = expt_dataset.I.values - adc1_mean
		adc2_unbiased = expt_dataset.Q.values - adc2_mean
		signal = savgol_filter(np.abs(adc1_unbiased + 1j * adc2_unbiased), 11, 3)
		# detect arrival of readout signal
		th = (np.mean(signal[:100]) + np.mean(signal[:-100])) / 2
		delay = np.where(signal > th)[0][0]
		delay = np.round(delay / 4) * 4
		dc_offset_i = -adc1_mean
		dc_offset_q = -adc2_mean

		# Update the config
		print(f"DC offset to add to I: {dc_offset_i:.6f} V")
		print(f"DC offset to add to Q: {dc_offset_q:.6f} V")
		print(f"TOF to add: {delay} ns")

		# Plot data
		fig = plt.figure(figsize=[14, 6])
		plt.subplot(121)
		plt.title("Single run")
		plt.plot(expt_dataset.I_single_run, "b", label="I")
		plt.plot(expt_dataset.Q_single_run, "r", label="Q")
		plt.axhline(y=0.5)
		plt.axhline(y=-0.5)
		xl = plt.xlim()
		yl = plt.ylim()
		plt.plot(xl, adc1_mean * np.ones(2), "k--")
		plt.plot(xl, adc2_mean * np.ones(2), "k--")
		plt.plot(delay * np.ones(2), yl, "k--")
		plt.xlabel("Time [ns]")
		plt.ylabel("Signal amplitude [V]")
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
				   fancybox=True, shadow=True, ncol=5)

		plt.subplot(122)
		plt.title("Averaged run")
		plt.plot(expt_dataset.I, "b", label="I")
		plt.plot(expt_dataset.Q, "r", label="Q")
		plt.xlabel("Time [ns]")
		plt.ylabel("Signal amplitude [V]")
		xl = plt.xlim()
		yl = plt.ylim()
		plt.plot(xl, adc1_mean * np.ones(2), "k--")
		plt.plot(xl, adc2_mean * np.ones(2), "k--")
		plt.plot(delay * np.ones(2), yl, "k--")
		plt.xlabel("Time [ns]")
		plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
				   fancybox=True, shadow=True, ncol=5)
		plt.grid("all")
		plt.tight_layout(pad=2)
		plt.show()

		return dc_offset_i.item(), dc_offset_q.item(), delay.item()


	def rr_freq(self, expt_dataset, to_plot = True, data_process_method = 'Amplitude'):
		"""Find resonator frequency
		
		Analysis for 1D resonator spectroscopy experiment. 
		Find the resonator frequency by looking for the minima.
		Tried to fit to Lorentzian lineshape, not good. Need more accurate model.
		
		Args:
			expt_dataset ([type]): [description]
			to_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			res_freq: [description]
		"""

		y = expt_dataset[data_process_method].values

		# get the key for coordinate along x
		coord_key = list(expt_dataset.coords.keys())
		for key in coord_key:
			if len(expt_dataset.coords[key].dims) == 1:
				coord_key_x = key # although this could be 'y', if the 1D data comes from a slice of 2D data.

		argmin_freq = y.argmin()
		res_freq = expt_dataset[coord_key_x].isel(x=argmin_freq).values.item()  # .item() done here!

		print(f"resonator frequency: {res_freq / u.MHz:.3f} MHz")

		if to_plot:
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [6, 3]
			plt.cla()
			expt_dataset[data_process_method].plot(x=coord_key_x, marker = '.')
			plt.axvline(x=(res_freq))
			plt.title(expt_dataset.attrs['long_name'])
			plt.xlabel(f"{coord_key_x} [{expt_dataset.coords[coord_key_x].attrs['units']}]")

			if data_process_method is 'Phase':
				plt.ylabel("Signal Phase [rad]")
			elif data_process_method is 'Amplitude':
				plt.ylabel("Signal Amplitude [V]")
			elif data_process_method is 'I':
				plt.ylabel("Signal I Quadrature [V]")
			elif data_process_method is 'Fidelity':
				plt.ylabel("Fidelity [%]")
			elif data_process_method is 'SNR':
				plt.ylabel("SNR")

		return res_freq


	def rr_freq_ge(self, expt_dataset, to_plot = True):
		"""Find resonator frequency from the phase signal.
		
		Analysis for 1D resonator spectroscopy experiment from rr_freq_ge. 
		Find the resonator frequency by looking for the maximum phase contrast between ground state and excited state readout signal.
		Note np.angle between (-pi, pi].
		
		Args:
			expt_dataset ([type]): [description]
			to_plot (bool): [description] (default: `True`)
		
		Returns:
			res_freq: [description]
		"""

		y = expt_dataset.Phase_g - expt_dataset.Phase_e

		# get the key for coordinate along x
		coord_key = list(expt_dataset.coords.keys())
		for key in coord_key:
			if len(expt_dataset.coords[key].dims) == 1:
				coord_key_x = key # although this could be 'y', if the 1D data comes from a slice of 2D data.

		argmin_freq = y.argmin()
		res_freq = expt_dataset[coord_key_x].isel(x=argmin_freq).values.item() # .item() done here!

		print(f"resonator frequency: {res_freq / u.MHz:.3f} MHz")

		if to_plot:
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [6, 3]
			plt.cla()
			y.plot(x=coord_key_x, marker = '.')
			plt.axvline(x=(res_freq))
			plt.ylabel("Phase [rad]")

		return res_freq


	def peak_fit(self, expt_dataset, method = "Lorentzian", to_plot = True, data_process_method = 'Amplitude'):
		"""Fit data to a peak with lineshape defined by method
		
		Mainly for spectroscopy experiments. Using lmfit to fit.
		Data must show a clear peak, otherwise the initial guess would not work well.
		output int right now!! Assuming it's always frequency!
		Todo: educated guess of the initial fitting parameters.
		
		Args:
			expt_dataset ([type]): [description]
			method (str): [description] (default: `"Lorentzian"`)
			to_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			[type]: [description]
		"""

		y = expt_dataset[data_process_method].values

		# get the key for coordinate along x
		coord_key = list(expt_dataset.coords.keys())
		for key in coord_key:
			if len(expt_dataset.coords[key].dims) == 1:
				coord_key_x = key # although this could be 'y', if the 1D data comes from a slice of 2D data.

		x = expt_dataset.coords[coord_key_x].values

		if method == "Lorentzian":
			mod = lmfit.models.LorentzianModel()
			pars_peak = mod.guess(y, x = x)
		elif method == "Gaussian":
			mod = lmfit.models.GaussianModel()
			pars_peak = mod.guess(y, x = x)
		else:
			print('-'*12 + 'model name (method) not defined ...')
			return None

		mod = mod + lmfit.models.LinearModel()
		pars = mod.make_params()
		for keys in pars_peak:
			pars[keys] = pars_peak[keys]
		pars['slope'].set(value = 0, vary = False) # flat background offset

		out = mod.fit(y, pars, x = x)
		peak_pos = out.params['center'].value

		if to_plot:
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [6, 3]
			plt.cla()
			plt.plot(x, y, '.')
			plt.plot(x, out.best_fit, 'r--')

			plt.title(expt_dataset.attrs['long_name'])
			plt.xlabel(f"{coord_key_x} [{expt_dataset.coords[coord_key_x].attrs['units']}]")

			if data_process_method is 'Phase':
				plt.ylabel("Signal Phase [rad]")
			elif data_process_method is 'Amplitude':
				plt.ylabel("Signal Amplitude [V]")
			elif data_process_method is 'I':
				plt.ylabel("Signal I Quadrature [V]")
			elif data_process_method is 'Fidelity':
				plt.ylabel("Fidelity [%]")
			elif data_process_method is 'SNR':
				plt.ylabel("SNR")

			if expt_dataset.coords[coord_key_x].attrs['units'] == 'V':
				print(f"peak voltage: {peak_pos: .3f} [V]")
			elif expt_dataset.coords[coord_key_x].attrs['units'] == 'Hz':
				print(f"peak frequency: {peak_pos / u.MHz: .1f} [MHz]")
			elif expt_dataset.coords[coord_key_x].attrs['units'] == 'ns':
				print(f"peak duration: {peak_pos: .0f} [ns]")

		if expt_dataset.coords[coord_key_x].attrs['units'] == 'V':
			return peak_pos.item()
		elif expt_dataset.coords[coord_key_x].attrs['units'] == 'Hz':
			return int(peak_pos.item())
		elif expt_dataset.coords[coord_key_x].attrs['units'] == 'ns':
			return int(peak_pos.item())
		else:
			print("coordinate units not recognized.")
			return None


	def multi_peak_fit(self, expt_dataset, method = "Lorentzian", to_plot = True):
		# place holder. Useful code block below

		Gausslist = []
		model, params = None, None
		for i, peak in enumerate(peak_data[1]):
		    comp = lmfit.models.GaussianModel(prefix='g{}_'.format(i))
		    pars = comp.make_params(center=center_val, amplitude=peak[0]) #Hm, maybe?
		    if model is None:
		        model = comp
		        params = pars
		    else:
		        model += comp
		        params.update(pars)


	def rabi_length(self, expt_dataset, method = "Sine", to_plot = True, data_process_method = 'Amplitude'):
		"""Find the pi pulse (both length and amplitude)
		
		By fitting the time (power) Rabi to a (decaying) sinusoidal wave, find the pi pulse length (amplitude).
		Using the lmfit built-in function. ExponentialModel and SineModel both have "amplitude", having a prefix for the decaying model to distinguish.
		
		Args:
			expt_dataset ([type]): [description]
			method (str): [description] (default: `"Sine"`)
			to_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			rabi_pulse_length: although this could be the amplitude as well.
		"""


		y = expt_dataset[data_process_method].values

		# get the key for coordinate along x
		coord_key = list(expt_dataset.coords.keys())
		for key in coord_key:
			if len(expt_dataset.coords[key].dims) == 1:
				coord_key_x = key # although this could be 'y', if the 1D data comes from a slice of 2D data.

		x = expt_dataset.coords[coord_key_x].values

		mod = lmfit.models.SineModel()
		pars_sin = mod.guess(y - np.mean(y), x = x)


		if method == "Sine":
			mod = mod + lmfit.models.LinearModel()
			pars = mod.make_params()
		elif method == "Decay":
			prefix = r"decay_"
			mod = mod * lmfit.models.ExponentialModel(prefix = prefix) + lmfit.models.LinearModel()
			pars = mod.make_params()
			pars[prefix + 'decay'].set(value = x[-1])
			pars[prefix + 'amplitude'].set(value = 1, vary = False) # only the amplitude in Sine will vary
		else:
			print('-'*12 + 'model name (method) not defined ...')
			return None
		pars['amplitude'].set(min = 0, max = np.inf) # force amplitude > 0
		pars['slope'].set(value = 0, vary = False) # flat background offset

		for keys in pars_sin:
			pars[keys] = pars_sin[keys]

		out = mod.fit(y, pars, x = x)

		if to_plot:
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [6, 3]
			plt.cla()
			plt.plot(x, y, '.')
			plt.plot(x, out.best_fit, 'r--')

			plt.title(expt_dataset.attrs['long_name'])
			plt.xlabel(f"{coord_key_x} [{expt_dataset.coords[coord_key_x].attrs['units']}]")

			if data_process_method is 'Phase':
				plt.ylabel("Signal Phase [rad]")
			elif data_process_method is 'Amplitude':
				plt.ylabel("Signal Amplitude [V]")
			elif data_process_method is 'I':
				plt.ylabel("Signal I Quadrature [V]")
			elif data_process_method is 'Fidelity':
				plt.ylabel("Fidelity [%]")
			elif data_process_method is 'SNR':
				plt.ylabel("SNR")

		# find pi pulse
		rabi_pulse_length = (np.pi / 2 - out.params['shift'].value) / out.params['frequency'].value
		rabi_period = np.pi / out.params['frequency'].value

		if expt_dataset.coords[coord_key_x].attrs['units'] == 'V': # find pulse amplitude
			if rabi_pulse_length // rabi_period < 0:
				rabi_pulse_length = rabi_pulse_length - rabi_period * (rabi_pulse_length // rabi_period)
			if rabi_pulse_length < 0.005:
				rabi_pulse_length += rabi_period
			print(f"rabi_pi_amp: {rabi_pulse_length:.3f} [{expt_dataset.coords[coord_key_x].attrs['units']}]")
			print(f"pi period: {rabi_period:.3f} [{expt_dataset.coords[coord_key_x].attrs['units']}]")
			return rabi_pulse_length.item()
		elif expt_dataset.coords[coord_key_x].attrs['units'] == 'ns':
			if rabi_pulse_length // rabi_period < 0:
				rabi_pulse_length = rabi_pulse_length - rabi_period * (rabi_pulse_length // rabi_period)
			if rabi_pulse_length < 5:
				rabi_pulse_length += rabi_period
			print(f"rabi_pi_pulse: {rabi_pulse_length:.1f} [{expt_dataset.coords[coord_key_x].attrs['units']}]")
			print(f"pi period: {rabi_period:.2f} [{expt_dataset.coords[coord_key_x].attrs['units']}]")
			return int(rabi_pulse_length.item())
		else:
			print("coordinate units not recognized.")
			return None


	def T1(self, expt_dataset, to_plot = True, data_process_method = 'Amplitude'):
		"""Fit to T1 relaxation curve.
		
		This function takes the amplitude in expt_dataset, fit it to a simple exponential model (with a constant offset), and returns the T1.
		Todo: allow rotation angle, rather than just using amplitude of the signal.
		
		Args:
			expt_dataset ([type]): [description]
			to_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			qubit_T1 (int): Could directly pass to machine.
		"""


		y = expt_dataset[data_process_method].values
		
		# get the key for coordinate along x
		coord_key = list(expt_dataset.coords.keys())
		for key in coord_key:
			if len(expt_dataset.coords[key].dims) == 1:
				coord_key_x = key # although this could be 'y', if the 1D data comes from a slice of 2D data.

		x = expt_dataset.coords[coord_key_x].values

		mod = lmfit.models.ExponentialModel()
		pars_decay = mod.guess(y - y[-1], x = x)

		mod = mod + lmfit.models.LinearModel()
		pars = mod.make_params()
		pars['slope'].set(value = 0, vary = False) # flat background offset

		for keys in pars_decay:
			pars[keys] = pars_decay[keys]

		out = mod.fit(y, pars, x = x)

		qubit_T1 = out.params['decay'].value
		
		print(f"Qubit T1: {qubit_T1:.1f} [{expt_dataset.coords[coord_key_x].attrs['units']}]")
		
		if to_plot:
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [6, 3]
			plt.cla()
			plt.plot(x, y, '.')
			plt.plot(x, out.best_fit, 'r--')

			plt.title(expt_dataset.attrs['long_name'])
			plt.xlabel(f"{coord_key_x} [{expt_dataset.coords[coord_key_x].attrs['units']}]")

			if data_process_method is 'Phase':
				plt.ylabel("Signal Phase [rad]")
			elif data_process_method is 'Amplitude':
				plt.ylabel("Signal Amplitude [V]")
			elif data_process_method is 'I':
				plt.ylabel("Signal I Quadrature [V]")
			elif data_process_method is 'Fidelity':
				plt.ylabel("Fidelity [%]")
			elif data_process_method is 'SNR':
				plt.ylabel("SNR")

		return int(qubit_T1.item()) # json takes int


	def ramsey(self, expt_dataset, to_plot = True, data_process_method = 'Amplitude'):
		"""Fit to ramsey experiment.
		
		Takes the amplitude of the signal, and fit it to a decaying sinusoidal model with a constant offset. 
		The decay is in stretched exponential form. Limiting exponent to [0,6] to be slightly more general than [1,4].
		Educated guess is used for the fit.
		Todo: allow rotation angle, rather than just using amplitude of the signal.

		
		Args:
			expt_dataset ([type]): [description]
			to_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			qubit_T2 (int): Could directly pass to machine.
		"""


		y = expt_dataset[data_process_method].values
		
		# get the key for coordinate along x
		coord_key = list(expt_dataset.coords.keys())
		for key in coord_key:
			if len(expt_dataset.coords[key].dims) == 1:
				coord_key_x = key # although this could be 'y', if the 1D data comes from a slice of 2D data.
		    	
		x = expt_dataset.coords[coord_key_x].values

		prefix = r"decay_"
		mod = lmfit.models.SineModel() * lmfit.Model(self._stretched_exp, prefix = prefix) + lmfit.models.LinearModel()
		pars = mod.make_params()
		pars[prefix + 'decay'].set(value = x[-1]/2, min = 1, max = np.inf)
		pars[prefix + 'amplitude'].set(value = 1, vary = False) # only the amplitude in Sine will vary
		pars[prefix + 'exponent'].set(value = 2, min = 0, max = 6) # stretched exp, assuming gaussian noise

		# educated guess
		delta = abs(x[0] - x[1])
		Fs = 1 / delta  # Sampling frequency
		L = np.size(x)
		NFFT = int(2 * 2 ** self._next_power_of_2(L))
		Freq = Fs / 2 * np.linspace(0, 1, NFFT // 2 + 1, endpoint=True)
		Y = np.fft.fft(y - np.mean(y), NFFT) / L
		DataY = abs(Y[0:(NFFT // 2)]) ** 2
		index = np.argmax(DataY)
		det = Freq[index]
		amp = abs(max(y) - min(y)) / 2

		pars['amplitude'].set(value = amp, min = 0, max = np.inf) # force amplitude > 0
		pars['slope'].set(value = 0, vary = False) # flat background offset
		pars['intercept'].set(value = y[-1])
		pars['frequency'].set(value = 2*np.pi*det, min = 0, max = np.inf)
		pars['shift'].set(value = 0, min = -10, max = 10)

		out = mod.fit(y, pars, x = x)

		qubit_T2 = out.params[prefix + "decay"].value
		qubit_T2_exponent = out.params[prefix + "exponent"].value
		qubit_detuning = out.params["frequency"].value / (2*np.pi) # fit is to (2pi) GHz
		print(f'Qubit T2*: {qubit_T2: .1f} [ns]')
		print(f'Exponent n = {qubit_T2_exponent: .1f}')
		print(f'Detuning = {qubit_detuning * 1E3: .1f} [MHz]')

		if to_plot:
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [6, 3]
			plt.cla()
			plt.plot(x, y, '.')
			plt.plot(x, out.best_fit, 'r--')

			plt.title(expt_dataset.attrs['long_name'])
			plt.xlabel(f"{coord_key_x} [{expt_dataset.coords[coord_key_x].attrs['units']}]")

			if data_process_method is 'Phase':
				plt.ylabel("Signal Phase [rad]")
			elif data_process_method is 'Amplitude':
				plt.ylabel("Signal Amplitude [V]")
			elif data_process_method is 'I':
				plt.ylabel("Signal I Quadrature [V]")
			elif data_process_method is 'Fidelity':
				plt.ylabel("Fidelity [%]")
			elif data_process_method is 'SNR':
				plt.ylabel("SNR")

		return int(qubit_T2.item()) # json only takes int


	def two_state_discriminator(self, expt_dataset, to_plot = True, to_print = True):
		"""Discriminate the g, e state using a threshold.
		
		Given two blobs in the IQ plane representing two states, finds the optimal threshold to discriminate between them
		and calculates the fidelity. Also returns the angle in which the data needs to be rotated in order to have all the
		information in the `I` (`X`) axis.

		.. note::
			This function assumes that there are only two blobs in the IQ plane representing two states (ground and excited)
			Unexpected output will be returned in other cases.
		
		Args:
			expt_dataset ([type]): extracts Ig, Qg, Ie, Qe--the 'I', 'Q' readout signal in the qubit ground and excited states
			to_plot (bool): [description] (default: `True`)
			to_print (bool): to print the analysis results (default: `True`)
		
		Returns:
			A tuple of (angle, threshold, fidelity, gg, ge, eg, ee).
			angle - The angle (in radians) in which the IQ plane has to be rotated in order to have all the information in
				the `I` axis.
			threshold - The threshold in the rotated `I` axis. The excited state will be when the `I` is larger (>) than
				the threshold.
			fidelity - The fidelity for discriminating the states.
			gg - The matrix element indicating a state prepared in the ground state and measured in the ground state.
			ge - The matrix element indicating a state prepared in the ground state and measured in the excited state.
			eg - The matrix element indicating a state prepared in the excited state and measured in the ground state.
			ee - The matrix element indicating a state prepared in the excited state and measured in the excited state.
		"""

		Ig = expt_dataset.I_g.values
		Qg = expt_dataset.Q_g.values
		Ie = expt_dataset.I_e.values
		Qe = expt_dataset.Q_e.values

		# Condition to have the Q equal for both states:
		angle = np.arctan2(np.mean(Qe) - np.mean(Qg), np.mean(Ig) - np.mean(Ie))
		C = np.cos(angle)
		S = np.sin(angle)
		# Condition for having e > Ig
		if np.mean((Ig - Ie) * C - (Qg - Qe) * S) > 0:
			angle += np.pi
			C = np.cos(angle)
			S = np.sin(angle)

		Ig_rotated = Ig * C - Qg * S
		Qg_rotated = Ig * S + Qg * C

		Ie_rotated = Ie * C - Qe * S
		Qe_rotated = Ie * S + Qe * C

		fit = minimize(
			self._false_detections,
			0.5 * (np.mean(Ig_rotated) + np.mean(Ie_rotated)),
			(Ig_rotated, Ie_rotated),
			method="Nelder-Mead",
		)
		threshold = fit.x[0]

		gg = np.sum(Ig_rotated < threshold) / len(Ig_rotated)
		ge = np.sum(Ig_rotated > threshold) / len(Ig_rotated)
		eg = np.sum(Ie_rotated < threshold) / len(Ie_rotated)
		ee = np.sum(Ie_rotated > threshold) / len(Ie_rotated)

		fidelity = 100 * (gg + ee) / 2

		if to_print:
			# print out the confusion matrix
			print(
				f"""
			Fidelity Matrix:
			-----------------
			| {gg:.3f} | {ge:.3f} |
			----------------
			| {eg:.3f} | {ee:.3f} |
			-----------------
			IQ plane rotated by: {180 / np.pi * angle:.1f}{chr(176)}
			Threshold: {threshold:.3e}
			Fidelity: {fidelity:.1f}%
			"""
			)

		if to_plot:
			fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
			plt.rcParams['figure.figsize'] = [9, 8]
			ax1.plot(Ig, Qg, ".", alpha=0.1, label="Ground", markersize=3)
			ax1.plot(Ie, Qe, ".", alpha=0.1, label="Excited", markersize=3)
			ax1.axis("equal")
			ax1.legend(["Ground", "Excited"])
			ax1.set_xlabel("I")
			ax1.set_ylabel("Q")
			ax1.set_title("Original Data")
			ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5,
					   markerscale=5)

			ax2.plot(Ig_rotated, Qg_rotated, ".", alpha=0.1, label="Ground", markersize=3)
			ax2.plot(Ie_rotated, Qe_rotated, ".", alpha=0.1, label="Excited", markersize=3)
			ax2.axis("equal")
			ax2.set_xlabel("I")
			ax2.set_ylabel("Q")
			ax2.set_title("Rotated Data")
			ax2.legend(["Ground", "Excited"])
			ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5,
					   markerscale=5)

			ax3.hist(Ig_rotated, bins=50, alpha=0.75, label="Ground")
			ax3.hist(Ie_rotated, bins=50, alpha=0.75, label="Excited")
			ax3.axvline(x=threshold, color="k", ls="--", alpha=0.5)
			text_props = dict(
				horizontalalignment="center",
				verticalalignment="center",
				transform=ax3.transAxes,
			)
			ax3.text(0.7, 0.9, f"Threshold:\n {threshold:.3e}", text_props)
			ax3.set_xlabel("I")
			ax3.set_ylabel("Counts")
			ax3.set_title("1D Histogram")

			ax4.imshow(np.array([[gg, ge], [eg, ee]]))
			ax4.set_xticks([0, 1])
			ax4.set_yticks([0, 1])
			ax4.set_xticklabels(labels=["|g>", "|e>"])
			ax4.set_yticklabels(labels=["|g>", "|e>"])
			ax4.set_ylabel("Prepared")
			ax4.set_xlabel("Measured")
			ax4.text(0, 0, f"{100 * gg:.1f}%", ha="center", va="center", color="k")
			ax4.text(1, 0, f"{100 * ge:.1f}%", ha="center", va="center", color="w")
			ax4.text(0, 1, f"{100 * eg:.1f}%", ha="center", va="center", color="w")
			ax4.text(1, 1, f"{100 * ee:.1f}%", ha="center", va="center", color="k")
			ax4.set_title("Fidelities")
			fig.tight_layout()
			fig.subplots_adjust(hspace=.5)
			plt.show()
		return angle.item(), threshold.item(), fidelity.item(), gg.item(), ge.item(), eg.item(), ee.item()


	def _stretched_exp(self, x, amplitude, decay, exponent):
		"""Auxiliary function for stretched exponential fitting.
		
		Currently only used in self.ramsey.
		
		Args:
			x ([type]): [description]
			amplitude ([type]): [description]
			decay ([type]): [description]
			exponent ([type]): [description]
		
		Returns:
			[type]: [description]
		"""
		return amplitude * np.exp(- (x / decay) ** exponent)


	def _false_detections(self, threshold, Ig, Ie):
		"""Auxiliary function for self.two_state_discriminator.
		
		This function finds the total number of false assignment events, based on "threshold". 
		Note that although the name has "_var", it is just the total number of events.
		
		Args:
			threshold ([type]): [description]
			Ig ([type]): [description]
			Ie ([type]): [description]
		
		Returns:
			[type]: [description]
		"""
		if np.mean(Ig) < np.mean(Ie):
			false_detections_var = np.sum(Ig > threshold) + np.sum(Ie < threshold)
		else:
			false_detections_var = np.sum(Ig < threshold) + np.sum(Ie > threshold)
		return false_detections_var


	def _next_power_of_2(self,x):
		"""Auxiliary function to find the next power of 2.
		
		Helpful in performing fft to turn Time into Frequency.
		Currently only used in self.ramsey.
		
		Args:
			x ([type]): [description]
		
		Returns:
			number: [description]
		"""

		return 0 if x == 0 else math.ceil(math.log2(x))


