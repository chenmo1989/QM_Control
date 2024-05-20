"""
This file contains useful python functions meant to simplify the Jupyter notebook.
AnalysisHandle
written by Mo Chen in Oct. 2023
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from qm.octave import *
from qm.octave.octave_manager import ClockMode
from configuration import *
from scipy import signal
from qm import SimulationConfig
from qualang_tools.bakery import baking
from qualang_tools.units import unit
from qm import generate_qua_script
from qm.octave import QmOctaveConfig
#from set_octave import ElementsSettings, octave_settings
from quam import QuAM
from scipy.io import savemat
from scipy.io import loadmat
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from typing import Union
import qutip as qt
import datetime
import os
import time
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np

from AnalysisClass_1D import AH_exp1D

class AH_exp2D:
	"""
	Class for analysis of 2D experiments
	Attributes:
		ham_param: parameters for the Jaynes-Cummings Hamiltonian. 
		poly_param: parameters for the polynomial qubit tuning curve. 
		json_name:

	Methods (useful ones):
		exp1D: the class of AnalysisClass_1D
		rr_vs_dc_flux(self, res_freq_sweep, dc_flux_sweep, sig_amp, init_guess = None)
		qubit_vs_dc_flux_guess(self, ham_param = None)
		ham(self, dc_flux, wr, Ec, Ej, c, phi0, g, output_flag)
	"""


	def __init__(self, ref_to_ham_param, ref_to_poly_param, ref_to_json_name, ref_to_exp1D):
		# only for temporary storage
		self.ham_param = ref_to_ham_param
		self.poly_param = ref_to_poly_param
		self.json_name = ref_to_json_name
		self.exp1D = ref_to_exp1D

	def rr_vs_dc_flux(self, expt_dataset, to_plot = True, init_guess = None, data_process_method = 'Amplitude'):
		"""Analyze the resonator spectroscopy vs dc flux data (2D)
		
		Resonator frequency is found first by locating the minimum signal Amplitude for each dc flux.
		The resonator frequency vs dc flux curve is then fitted with the Jaynes-Cummings model (qubit-resonator).
		Input data in Hz; Fitted parameters in MHz (ham_param).
		Assumes a square experimental sweep, i.e., the same frequency sweep for all the dc flux.
		ham_param (return) is already a list (converted from numpy). Can go directly into machine.

		Args:
			expt_dataset ([type]): [description]
			to_plot (bool): [description] (default: `True`)
			init_guess ([type]): Optional initial guess [wr, Ec, Ej, c, phi0, g] for fitting to the model (default: `None`)
		
		Returns:
			[type]: [description]
		"""


		# extract data convenient for fitting
		coord_key = list(expt_dataset.coords.keys())
		if len(expt_dataset[coord_key[0]].dims) == 1:
			coord_key_x = coord_key[0]
			coord_key_y = coord_key[1]
		else:
			coord_key_x = coord_key[1]
			coord_key_y = coord_key[0]

		x = expt_dataset.coords[coord_key_x].values # DC flux sweep, only used in plot of processed values
		
		# define init_guess for the fitting
		if init_guess is None:
			wr = np.min(expt_dataset.coords[coord_key_y].values) / u.MHz  # Resonator frequency
			Ec = 180.0  # Capacitive energy
			Ej = 30.0E3  # Inductive energy
			c = 0.05  # Period in cosine function for flux
			phi0 = 0.4  # Offset in cosine function for flux
			g = 70.0  # RR-qubit coupling
		else:
			wr = init_guess[0]
			Ec = init_guess[1]
			Ej = init_guess[2]
			c = init_guess[3]
			phi0 = init_guess[4]
			g = init_guess[5]

		init_guess = [wr, Ec, Ej, c, phi0, g]

		# instead of using for loop, use argmin on xarray to do dimensional processing.
		argmin_freq = expt_dataset[data_process_method].argmin(dim="y")
		res_freq = expt_dataset[coord_key_y].isel(y = argmin_freq).values

		# Fit data from res_freq to Hamiltonian function
		ham_param, _ = curve_fit(
			lambda dc_flux_sweep, *guess: self.ham(dc_flux_sweep, *guess, output_flag = 1),
			xdata = x,
			ydata = res_freq/u.MHz,
			p0 = init_guess,
			check_finite = True,
			bounds=(
				(wr - 200, Ec - 50, Ej - 10000, 0.0001, -6, g - 50),
				(wr + 200, Ec + 100, Ej + 10000, 4, 6, g + 50)
			)
		)

		if to_plot:
			# 2D spectroscopy plot
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [8, 4]
			plt.cla()
			expt_dataset[data_process_method].plot(x = coord_key_x, y = coord_key_y, cmap="seismic")

			# plot data and fitting
			plt.plot(x, res_freq, '.')
			plt.plot(x, self.ham(x, *ham_param, 1) * u.MHz)
			plt.show()

		ham_param = ham_param.tolist()
		self.ham_param = ham_param

		return ham_param


	def qubit_vs_dc_flux_guess(self, to_plot = True, ham_param = None):
		"""Based on ham_param, make an educated guess of the poly_param for the qubit.
		
		Use ham_param to generate a simulated qubit tuning curve around a +-2V range of the sweet spot. 
		Then use 2nd order polynomial to fit to the (simulated) qubit tuning curve.
		This is generally to be applied to data from rr_vs_dc_flux experiment, to guide the qubit_vs_dc_flux experiment.
		poly_param (return) is already a list (converted from numpy). Can go directly into machine.

		Args:
			to_plot (bool): [description] (default: `True`)
			ham_param ([type]): [description] (default: `None`)
		
		Returns:
			[type]: [description]
		"""


		if ham_param is None:
			ham_param = self.ham_param

		c = ham_param[3]
		phi0 = ham_param[4]
		dc_ss = -phi0 / (2 * np.pi * c)
		dc_flux_fit = dc_ss + np.linspace(start=-2, stop=2, num=40, endpoint=True)
		qubit_freq_est = self.ham(dc_flux_fit, *ham_param, output_flag=2)
		poly_param = np.polyfit(dc_flux_fit, qubit_freq_est, deg = 2)

		if to_plot:
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [8, 4]
			plt.plot(dc_flux_fit, qubit_freq_est, '.')
			plt.plot(dc_flux_fit, np.polyval(poly_param,dc_flux_fit))

		poly_param = poly_param.tolist()
		self.poly_param = poly_param

		return poly_param


	def qubit_vs_flux(self, expt_dataset, fit_order = 4, to_plot = True, data_process_method = 'I'):
		"""Analyze the qubit spectroscopy vs flux data (2D). Work with both dc flux and fast flux.
		
		2D data is first sliced at each dc flux point, and qubit frequency is identified by fitting to a Gaussian peak (using Amplitude).
		The qubit frequency vs dc flux curve is then fitted with a "fit_order" (default = 4) polynomial.
		Input data in Hz; Fitted parameters in MHz (poly_param).
		Assumes non-square experimental sweep. i.e., one of the coords should be two-dimensional ("Qubit_Frequency").
		poly_param (return) is already a list (converted from numpy). Can go directly into machine.
		
		Args:
			qubit_freq_sweep ([type]): [description]
			dc_flux_sweep ([type]): [description]
			sig_amp ([type]): [description]
			fit_order (number): [description] (default: `4`)
			to_plot (bool): [description] (default: `True`)
		
		Returns:
			[type]: [description]
		"""


		# extract data convenient for fitting
		coord_key = list(expt_dataset.coords.keys())
		if len(expt_dataset[coord_key[0]].dims) == 1:
			coord_key_x = coord_key[0]
			coord_key_y = coord_key[1]
		else:
			coord_key_x = coord_key[1]
			coord_key_y = coord_key[0]

		x = expt_dataset.coords[coord_key_x].values # DC flux sweep
		
		qubit_freq = []
		for i in range(len(x)): # go over dc flux
			qubit_freq_tmp = self.exp1D.peak_fit(expt_dataset.isel(x=i), method="Gaussian", to_plot = False, data_process_method = data_process_method)
			qubit_freq.append(qubit_freq_tmp) # in Hz

		qubit_freq = np.array(qubit_freq)

		# fit and plot
		poly_param = np.polyfit(x, qubit_freq / u.MHz, deg=fit_order)
		self.poly_param = poly_param

		if to_plot:
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [8, 4]
			plt.cla()

			# 2D spectroscopy plot
			expt_dataset[data_process_method].plot(x = coord_key_x, y = coord_key_y, cmap = "seismic")

			plt.title(expt_dataset.attrs['long_name'])
			# plot data and fitting
			plt.plot(x, qubit_freq, '.')
			plt.plot(x, np.polyval(poly_param, x) * u.MHz)
			plt.show()

		poly_param = poly_param.tolist()
		self.poly_param = poly_param

		return poly_param


	def SWAP_fft(self, expt_dataset, to_plot = True, data_process_method = 'I'):
	    # Find the coordinate that contains 'Time'
	    for keys in expt_dataset.coords:
	        if 'Time' in keys:
	            coord_key_time = keys
			else:
				coord_key_other = keys

	    if coord_key_time is None:
	        print("No coordinate containing 'Time' found in the dataset.")
	        return None
	    
	    # find the dimension name and corresponding axis index for 'Time'
	    dim_name_time = expt_dataset.coords[coord_key_time].dims[0]
	    axis_index_time = dataset[data_process_method].get_axis_num(dim_name_time)

	    # Perform FFT along the specified time axis
	    fft_result = np.fft.fft(dataset[data_process_method].values, axis=axis_index_time)
	    
	    # Calculate the sample spacing (assuming time is in seconds)
	    sample_spacing = np.diff(expt_dataset.coords[coord_key_time][0:2])[0] # difference between [1], [0] items. Assuming equally-spaced.
	    
	    # Compute the corresponding frequencies
	    frequencies = np.fft.fftfreq(time.size, d=sample_spacing) * 1E9 # GHz to Hz
	    
	    # take the positive frequency part
	    positive_freq_indices = frequencies >= 0
	    frequencies = frequencies[positive_freq_indices]
    	fft_result = fft_result[positive_freq_indices, :] if axis_index_time == 0 else fft_result[:, positive_freq_indices]
	    
	    # generate an xarray dataset to save it

	    if axis_index_time ==0:
		    fft_dataset = xr.Dataset(
				    {
				        "FFT_Result": (["x", "y"], fft_result),
				    },
				    coords={
				        "FFT_Frequency": (["x"], frequencies),
				        coord_key_other: (["y"], expt_dataset.coords[coord_key_other]),
				    },
				)
	    else:
	    	fft_dataset = xr.Dataset(
				    {
				        "FFT_Result": (["x", "y"], fft_result),
				    },
				    coords={
				        "FFT_Frequency": (["y"], frequencies),
				        coord_key_other: (["x"], expt_dataset.coords[coord_key_other]),
				    },
				)

		
		fft_dataset.attrs['name'] = "fft_" + expt_dataset.attrs['name']
		fft_dataset.attrs['long_name'] = "FFT of " + expt_dataset.attrs['long_name']
		fft_dataset.attrs['qubit'] = expt_dataset.attrs['qubit']
		fft_dataset.attrs['TLS'] = expt_dataset.attrs['TLS']
		
		if to_plot:
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [8, 4]
			plt.cla()

			# 2D spectroscopy plot
			expt_dataset[data_process_method].plot(x = coord_key_other, y = "FFT_Frequency", cmap = "seismic")
			plt.show()

	    return fft_dataset


	def ham(self, dc_flux, wr, Ec, Ej, c, phi0, g, output_flag = 1):
		"""
		The Jaynes-Cummings Hamiltonian, all in units of MHz

		Args:
			dc_flux: dc flux voltage values
			wr: bare resonator frequency
			Ec: capacitive energy of qubit
			Ej: Josephson energy of qubit
			c, phi0: linear coefficient for the mapping between dc voltage and flux, following
				magnetic flux = 2 * np.pi * c * dc_flux + phi0
			output_flag: 1-rr, 2-qubit, otherwise-pass
			
		Return:
			freq_sys: frequency of the system, with the system being either resonator or qubit
		"""

		N = 4  # 0-3 photons
		a = qt.tensor(qt.destroy(N), qt.qeye(N))  # cavity mode
		b = qt.tensor(qt.qeye(N), qt.destroy(N))  # qubit

		freq_sys = [] # initialize an empty list

		# Hamiltonian as a function of flux
		for k in range(np.size(dc_flux)):
			H = wr * a.dag() * a + (np.sqrt(8 * Ec * Ej * np.abs(
				np.cos(self.phi_flux_rr(dc_flux[k], c, phi0)))) - Ec) * b.dag() * b - Ec / 2 * b.dag() * b.dag() * b * b + g * (
							a * b.dag() + a.dag() * b)
			w, v = H.eigenstates() # eigenenergies and states. Already normalized.

			# find the eigenstate with the largest component at corresponding positions. This will be the dominant desired state.
			idx_00 = np.argmax([np.abs(state[0]) for state in v])  # |0,0>
			idx_01 = np.argmax([np.abs(state[N]) for state in v])  # |1,0> photon
			idx_02 = np.argmax([np.abs(state[1]) for state in v])   # |0,1> qubit
			if output_flag == 1:
				freq_sys.append(np.abs(np.maximum(w[idx_01], w[idx_02]) - w[idx_00]))
			elif output_flag == 2:
				freq_sys.append(np.abs(np.minimum(w[idx_01], w[idx_02]) - w[idx_00]))
			else:
				pass
		freq_sys = np.array(freq_sys)

		return freq_sys


	def phi_flux_rr(self,dc_flux, c, phi0):
		"""
		linear mapping function from dc flux voltage to dc magnetic flux
		magnetic flux = 2 * np.pi * c * dc_flux + phi0
		Args:
			dc_flux: the voltage we apply in experiment (QDAC)
			c: slope
			phi0: offset
		Return:
			the magnetic flux
		"""
		return 2 * np.pi * c * dc_flux + phi0