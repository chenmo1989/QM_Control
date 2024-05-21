from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig, LoopbackInterface, generate_qua_script
from qm.octave import *
from configuration import *
from scipy import signal
from qualang_tools.bakery import baking
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qm.octave import QmOctaveConfig
from quam import QuAM
from scipy.optimize import curve_fit, minimize
#from qutip import *
from typing import Union
from macros import ham, readout_rotated_macro, declare_vars, wait_until_job_is_paused
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
import xarray as xr


class EH_RR:
	"""
	class in ExperimentHandle, for Readout Resonator (RR) related 2D experiments
	
	Methods:
	
	"""


	def __init__(self, ref_to_local_exp1D, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm):
		self.exp1D = ref_to_local_exp1D
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs
		self.qmm = ref_to_qmm

	def rr_vs_dc_flux(self, machine, res_freq_sweep, dc_flux_sweep, qubit_index, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False):
		"""Run resonator spectroscopy vs dc flux
		
		This is supposed to be some of the first qubit characterization experiment. Purpose is to get an initial estimate
		of the qubit-resonator system parameters. I choose to use a Jaynes-Cummings model for this fitting in Analysis.
		expt_dataset has DataArray I, Q as 2D numpy array, Coordinates res_freq_sweep and dc_flux_sweep are both 1D numpy arrays.
		
		Args:
			machine ([type]): [description]
			res_freq_sweep ([type]): [description]
			dc_flux_sweep ([type]): [description]
			qubit_index ([type]): [description]
			n_avg ([type]): [description]
			cd_time ([type]): [description]
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			live_plot (bool): [description] (default: `False`)
		
		Returns:
			[type]: [description]
		"""

		# 2D scan, RR frequency vs DC flux
		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		res_if_sweep = res_freq_sweep - res_lo
		res_if_sweep = np.floor(res_if_sweep)

		if np.max(abs(res_if_sweep)) > 400E6: # check if parameters are within hardware limit
			print("res if range > 400MHz")
			return machine, None

		# QDAC communication through set_Labber
		
		with program() as resonator_spec_2D:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			m = declare(int)  # DC sweep index
			df = declare(int)  # Resonator frequency

			with for_(m, 0, m < len(dc_flux_sweep) + 1, m + 1):
				pause() # This waits until it is resumed from python
				with for_(n, 0, n < n_avg, n + 1):
					with for_(*from_array(df, res_if_sweep)):
						update_frequency(machine.resonators[qubit_index].name, df)
						readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
						wait(cd_time * u.ns, machine.resonators[qubit_index].name)
						# Save data to the stream processing
						save(I, I_st)
						save(Q, Q_st)
				save(m, n_st)

			with stream_processing():
				# Cast the data into a 2D matrix, average the matrix along its second dimension (of size 'n_avg') and store the results
				# Mo Chen: I believe this instead of .average() to ensure batches of data with desired dimension (to work with .wait_for_values(m + 1) and .fetch(m)). But since we have the wait_until_job_is_paused(job), it probably doesn't matter. Good practice to have though.
				I_st.buffer(len(res_freq_sweep)).buffer(n_avg).map(FUNCTIONS.average()).save_all("I")
				Q_st.buffer(len(res_freq_sweep)).buffer(n_avg).map(FUNCTIONS.average()).save_all("Q")
				n_st.save_all("iteration")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		
		if to_simulate: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, resonator_spec_2D, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(resonator_spec_2D)
			# Creates results handles to fetch the data
			res_handles = job.result_handles
			I_handle = res_handles.get("I")
			Q_handle = res_handles.get("Q")
			n_handle = res_handles.get("iteration")

			# Initialize empty vectors to store the global 'I' & 'Q' results
			I_tot = []
			Q_tot = []
			# Live plotting
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			for m in range(len(dc_flux_sweep)):
				# set QDAC voltage
				dc_flux = dc_flux_sweep[m]
				machine = self.set_Labber.set_QDAC_single(machine, qubit_index, dc_flux)
				time.sleep(0.1) # let the DC flux to set

				# Resume the QUA program
				job.resume()
				# Wait until the program reaches the 'pause' statement again, indicating that the QUA program is done
				wait_until_job_is_paused(job)

				# Wait until the data of this run is processed by the stream processing
				I_handle.wait_for_values(m + 1)
				Q_handle.wait_for_values(m + 1)
				n_handle.wait_for_values(m + 1)
				
				# Fetch the data from the last OPX run corresponding to the current LO frequency
				I = np.concatenate(I_handle.fetch(m)["value"])
				Q = np.concatenate(Q_handle.fetch(m)["value"])
				iteration = n_handle.fetch(m)["value"][0]
				# Convert results into Volts
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				
				# Update the list of global results
				I_tot.append(I)
				Q_tot.append(Q)
				
				# Progress bar
				progress_counter(iteration, len(dc_flux_sweep), start_time=datetime.datetime.timestamp(timestamp_created))

				# Plot results
				if live_plot:
					plt.cla()
					plt.title("Resonator spectroscopy")
					plt.plot((res_freq_sweep) / u.MHz, np.sqrt(I**2 +  Q**2), ".")
					plt.xlabel("Frequency [MHz]")
					plt.ylabel("Signal Amplitude [V]")
					plt.pause(0.5)

			# Interrupt the FPGA program (so it's not waiting to resume??)
			timestamp_finished = datetime.datetime.now()
			job.halt()
			# into proper 2D np array. First dim is the list dimension (dc flux in this case)
			I = np.array(I_tot)
			Q = np.array(Q_tot)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
				{
					"I": (["x", "y"], I),
					"Q": (["x", "y"], Q),
				},
				coords={
					"DC_Flux": (["x"], dc_flux_sweep),
					"Readout_Frequency": (["y"], res_freq_sweep)
				},
			)

			expt_name = 'res_spec2D'
			expt_long_name = 'Resonator Spectroscopy vs DC Flux'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = []  # use t0, t1, t2, ...
			expt_sequence = """with for_(m, 0, m < len(dc_flux_sweep) + 1, m + 1):
	pause() # This waits until it is resumed from python
	with for_(n, 0, n < n_avg, n + 1):
		with for_(*from_array(df, res_if_sweep)):
			update_frequency(machine.resonators[qubit_index].name, df)
			readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
			wait(cd_time * u.ns, machine.resonators[qubit_index].name)
			# Save data to the stream processing
			save(I, I_st)
			save(Q, Q_st)
	save(m, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
											  expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				sig_amp = np.sqrt(expt_dataset.I ** 2 + expt_dataset.Q ** 2)
				sig_amp.plot(x = list(expt_dataset.coords.keys())[0], y = list(expt_dataset.coords.keys())[1], cmap = "seismic")
				plt.show()
				plt.title(expt_dataset.attrs['long_name'])

			return machine, expt_dataset


	def rr_pulse_optimize(self, machine, res_duration_sweep_abs, res_amp_sweep_abs, qubit_index, n_avg=1E3, cd_time=20E3, to_simulate=False, simulation_len=3000, final_plot=True, live_plot=False, data_process_method = 'I'):
		"""Characterize qubit decay due to a resonator pulse (with varying amplitude and duration). Used to optimize the readout pulse.
		
		The sequence is pi pulse -- variable readout pulse -- wait time (to make the interval between pi and readout constant) -- readout
		Measurement characterizes additional qubit decay due to the variable readout pulse. This could help us avoid readout conditions less favorable with short qubit T1. 
		In a sense, this sequence characterizes QNDness of the readout pulse
		We want to find the optimal readout amp and duration.
		
		Args:
			machine ([type]): [description]
			res_duration_sweep_abs ([type]): [description]
			res_amp_sweep_abs ([type]): [description]
			qubit_index ([type]): [description]
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `20E3`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `3000`)
			final_plot (bool): [description] (default: `True`)
			live_plot (bool): [description] (default: `False`)
			data_process_method (str): [description] (default: `'I'`)
		
		Returns:
			[type]: [description]
		"""
		
		
		res_amp_sweep_rel = res_amp_sweep_abs / 0.25 # 0.25 is the amplitude of "cw" pulse/"const_wf"
		res_duration_sweep_cc = res_duration_sweep_abs // 4  # in clock cycles
		res_duration_sweep_cc = np.unique(res_duration_sweep_cc)
		res_duration_sweep = res_duration_sweep_cc.astype(int)  # clock cycles
		res_duration_sweep_abs = res_duration_sweep * 4  # time in ns
		total_wait_time = max(res_duration_sweep) + 25 # total wait time. additional 100ns is added.

		with program() as rr_pulse_optimize:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			t = declare(int)
			da = declare(fixed)
			dw = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, res_duration_sweep)):
					with for_(*from_array(da, res_amp_sweep_rel)):
						assign(dw, total_wait_time - t) # in clock cycles

						play("pi", machine.qubits[qubit_index].name)
						align()
						play("cw" * amp(da), machine.resonators[qubit_index].name, duration = t)
						align()
						wait(dw, machine.resonators[qubit_index].name)
						readout_rotated_macro(machine.resonators[qubit_index].name, I, Q)
						wait(cd_time * u.ns, machine.resonators[qubit_index].name)
						save(I, I_st)
						save(Q, Q_st)
				save(n, n_st)

			with stream_processing():
				# for the progress counter
				n_st.save("iteration")
				I_st.buffer(len(res_amp_sweep_rel)).buffer(len(res_duration_sweep)).average().save("I")
				Q_st.buffer(len(res_amp_sweep_rel)).buffer(len(res_duration_sweep)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, rr_pulse_optimize, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(rr_pulse_optimize)
			results = fetching_tool(job, ["I", "Q", "iteration"], mode="live")

			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)

			while results.is_processing():
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				
				if live_plot:
					plt.title("Readout Pulse Optimization")
					if data_process_method == 'Phase':
						plt.pcolormesh(res_amp_sweep_abs, res_duration_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), cmap="seismic")
					elif data_process_method == 'Amplitude':
						plt.pcolormesh(res_amp_sweep_abs, res_duration_sweep_abs, np.abs(I + 1j * Q), cmap="seismic")
					elif data_process_method == 'I':
						plt.pcolormesh(res_amp_sweep_abs, res_duration_sweep_abs, I, cmap="seismic")
					plt.xlabel("Readout Pulse Amplitude [V]")
					plt.ylabel("Readout Pulse Duration [ns]")
					plt.show()
					plt.pause(0.5)

			# fetch all data after live-updating
			timestamp_finished = datetime.datetime.now()
			I, Q, _ = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
			
			# generate xarray dataset
			expt_dataset = xr.Dataset(
			    {
			        "I": (["x", "y"], I),
			        "Q": (["x", "y"], Q),
			    },
			    coords={
			        "Readout_Pulse_Amplitude": (["y"], res_amp_sweep_abs),
			        "Readout_Pulse_Duration": (["x"], res_duration_sweep_abs),
			    },
			)
			
			expt_name = 'res_pulse_optimize'
			expt_long_name = 'Readout Pulse Optimization'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(t, res_duration_sweep)):
		with for_(*from_array(da, res_amp_sweep_rel)):
			assign(dw, total_wait_time - t) # in clock cycles

			play("pi", machine.qubits[qubit_index].name)
			align()
			play("cw" * amp(da), machine.resonators[qubit_index].name, duration = t)
			align()
			wait(dw, machine.resonators[qubit_index].name)
			readout_rotated_macro(machine.resonators[qubit_index].name, I, Q)
			wait(cd_time * u.ns, machine.resonators[qubit_index].name)
			save(I, I_st)
			save(Q, Q_st)
	save(n, n_st)"""

			# save data
			self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x = list(expt_dataset.coords.keys())[0], y = list(expt_dataset.coords.keys())[1], cmap = "seismic")
				plt.title(expt_dataset.attrs['long_name'])

			return machine, expt_dataset

