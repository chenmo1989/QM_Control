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
from typing import Union
from macros import ham, readout_rotated_macro, declare_vars, wait_until_job_is_paused
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
import xarray as xr

#warnings.filterwarnings("ignore")


class EH_Rabi:
	"""Class in ExperimentHandle, for running Rabi sequence based 1D experiments.
	
	[description]
	
	Attributes:
		set_octave
		set_Labber
		datalogs

	Methods:
		qubit_freq
		rabi_length
		rabi_amp
		qubit_switch_delay
		qubit_switch_buffer
		TLS_freq
		TLS_rabi_length
		TLS_rabi_amp
		ef_freq
		ef_rabi_length
		ef_rabi_amp
		ef_rabi_thermal
	"""
	
	def __init__(self, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm):
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs
		self.qmm = ref_to_qmm


	def qubit_freq(self, machine, qubit_freq_sweep, qubit_index, pi_amp_rel = 1.0, ff_amp = 0.0, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'Amplitude'):
		"""Qubit spectroscopy experiment.
		
		Qubit spectroscopy to find the qubit resonance frequency, sweeping XY pulse frequency (equivalent of ESR for spin qubit).
		
		Args:
			machine ([type]): 1D array of qubit frequency sweep
			qubit_freq_sweep ([type]): [description]
			qubit_index ([type]): [description]
			pi_amp_rel (number): [description] (default: `1.0`)
			ff_amp (number): Fast flux amplitude that overlaps with the Rabi pulse. The ff pulse is 40ns longer than Rabi pulse, and share the pulse midpoint. (default: `0.0`)
			n_avg (number): Repetitions of the experiments (default: `1E3`)
			cd_time (number): Cooldown time between subsequent experiments (default: `10E3`)
			to_simulate (bool): True: run simulation; False: run experiment (default: `False`)
			simulation_len (number): Length of the sequence to simulate. In clock cycles (4ns) (default: `1000`)
			final_plot (bool): True: plot the experiment. False: do not plot (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			machine
			expt_dataset
		"""
		
		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
		qubit_if_sweep = qubit_freq_sweep - qubit_lo
		qubit_if_sweep = np.floor(qubit_if_sweep)
		ff_duration = machine.qubits[qubit_index].pi_length + 40

		if np.max(abs(qubit_if_sweep)) > 400E6: # check if parameters are within hardware limit
			print("qubit if range > 400MHz")
			return machine, None

		with program() as qubit_freq_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			df = declare(int)

			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(df,qubit_if_sweep)):
					update_frequency(machine.qubits[qubit_index].name, df)
					play("const" * amp(ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
					wait(5, machine.qubits[qubit_index].name)
					play('pi'*amp(pi_amp_rel), machine.qubits[qubit_index].name)
					wait(5, machine.qubits[qubit_index].name)
					align(machine.qubits[qubit_index].name, machine.flux_lines[qubit_index].name,
						  machine.resonators[qubit_index].name)
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					align()
					# eliminate charge accumulation
					play("const" * amp(-1 * ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
					wait(cd_time * u.ns, machine.flux_lines[qubit_index].name)
					save(I, I_st)
					save(Q, Q_st)
				save(n, n_st)
			with stream_processing():
				n_st.save('iteration')
				I_st.buffer(len(qubit_if_sweep)).average().save("I")
				Q_st.buffer(len(qubit_if_sweep)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		# Simulate or execute #
		if to_simulate: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, qubit_freq_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(qubit_freq_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

				while results.is_processing():
					# Fetch results
					I, Q, iteration = results.fetch_all()
					I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
					Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
					# progress bar
					progress_counter(iteration, n_avg, start_time=results.get_start_time())

					# Update the live plot!
					plt.cla()
					plt.title("Qubit Spectroscopy")
					if data_process_method == 'Phase':
						plt.plot((qubit_freq_sweep) / u.MHz, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.xlabel("Qubit Frequency [MHz]")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot((qubit_freq_sweep) / u.MHz, np.abs(I + 1j * Q), ".")
						plt.xlabel("Qubit Frequency [MHz]")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot((qubit_freq_sweep) / u.MHz, I, ".")
						plt.xlabel("Qubit Frequency [MHz]")
						plt.ylabel("Signal I Quadrature [V]")
					plt.pause(0.5)
			else:
				while results.is_processing():
					# Fetch results
					I, Q, iteration = results.fetch_all()
					# progress bar
					progress_counter(iteration, n_avg, start_time=results.get_start_time())

			# fetch all data after live-updating
			timestamp_finished = datetime.datetime.now()
			I, Q, iteration = results.fetch_all()
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
			    {
			        "I": (["x"], I),
			        "Q": (["x"], Q),
			    },
			    coords={
			        "Qubit_Frequency": (["x"], qubit_freq_sweep),
			    },
			)
			
			expt_name = 'spec'
			expt_long_name = 'Qubit Spectroscopy'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n+1):
	with for_(*from_array(df,qubit_if_sweep)):
		update_frequency(machine.qubits[qubit_index].name, df)
		play("const" * amp(ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
		wait(5, machine.qubits[qubit_index].name)
		play('pi'*amp(pi_amp_rel), machine.qubits[qubit_index].name)
		wait(5, machine.qubits[qubit_index].name)
		align(machine.qubits[qubit_index].name, machine.flux_lines[qubit_index].name,
			  machine.resonators[qubit_index].name)
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		align()
		# eliminate charge accumulation
		play("const" * amp(-1 * ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
		save(I, I_st)
		save(Q, Q_st)
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')				

			return machine, expt_dataset


	def rabi_length(self, machine, tau_sweep_abs, qubit_index, pi_amp_rel = 1.0, ff_amp = 0.0, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'Amplitude'):
		"""Qubit time Rabi experiment
		
		Qubit time Rabi experiment in 1D, sweeping length of the Rabi pulse. 
		Note that input argument is now in ns. Non-integer clock cycles values will be removed.
		
		Args:
			machine ([type]): [description]
			tau_sweep_abs ([type]): in ns!
			qubit_index ([type]): [description]
			pi_amp_rel (number): [description] (default: `1.0`)
			ff_amp (number): Fast flux amplitude that overlaps with the Rabi pulse. The ff pulse is 40ns longer than the Rabi pulse. (default: `0.0`)
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `10E3`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			machine
			expt_dataset: time in ns!
		"""
		
		qubit_if = machine.qubits[qubit_index].f_01 - machine.octaves[0].LO_sources[1].LO_frequency

		if min(tau_sweep_abs) < 16:
			print("some ramsey lengths shorter than 4 clock cycles, removed from run")
			tau_sweep_abs = tau_sweep_abs[tau_sweep_abs>15]

		tau_sweep_cc = tau_sweep_abs//4 # in clock cycles
		tau_sweep_cc = np.unique(tau_sweep_cc)
		tau_sweep = tau_sweep_cc.astype(int) # clock cycles, used for experiments
		tau_sweep_abs = tau_sweep * 4 # time in ns

		with program() as time_rabi:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, tau_sweep)):
					update_frequency(machine.qubits[qubit_index].name, qubit_if)
					play("const" * amp(ff_amp), machine.flux_lines[qubit_index].name, duration=t+10)
					wait(5, machine.qubits[qubit_index].name)
					play("pi" * amp(pi_amp_rel), machine.qubits[qubit_index].name, duration=t)
					wait(5, machine.qubits[qubit_index].name)
					align(machine.qubits[qubit_index].name, machine.flux_lines[qubit_index].name,
						  machine.resonators[qubit_index].name)
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					align()
					# eliminate charge accumulation
					play("const" * amp(-1 * ff_amp), machine.flux_lines[qubit_index].name, duration=t+10)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					save(I, I_st)
					save(Q, Q_st)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)  # in clock cycles
			job = self.qmm.simulate(config, time_rabi, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(time_rabi)
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

			# Live plotting
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

				while results.is_processing():
					# Fetch results
					I, Q, iteration = results.fetch_all()
					I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
					Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
					# Progress bar
					progress_counter(iteration, n_avg, start_time=results.get_start_time())

					plt.cla()
					plt.title("Qubit Time Rabi")
					if data_process_method == 'Phase':
						plt.plot(tau_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot(tau_sweep_abs, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot(tau_sweep_abs, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("Rabi Time [ns]")
					plt.pause(0.5)
			else:
				while results.is_processing():
					# Fetch results
					I, Q, iteration = results.fetch_all()
					# Progress bar
					progress_counter(iteration, n_avg, start_time=results.get_start_time())

			# fetch all data after live-updating
			timestamp_finished = datetime.datetime.now()
			I, Q, iteration = results.fetch_all()
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
			    {
			        "I": (["x"], I),
			        "Q": (["x"], Q),
			    },
			    coords={
			        "Rabi_Time": (["x"], tau_sweep_abs),
			    },
			)

			expt_name = 'time_rabi'
			expt_long_name = 'Qubit Time Rabi'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(t, tau_sweep_abs)):
		update_frequency(machine.qubits[qubit_index].name, qubit_if)
		play("const" * amp(ff_amp), machine.flux_lines[qubit_index].name, duration=t+10)
		wait(5, machine.qubits[qubit_index].name)
		play("pi" * amp(pi_amp_rel), machine.qubits[qubit_index].name, duration=t)
		wait(5, machine.qubits[qubit_index].name)
		align(machine.qubits[qubit_index].name, machine.flux_lines[qubit_index].name,
			  machine.resonators[qubit_index].name)
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		align()
		# eliminate charge accumulation
		play("const" * amp(-1 * ff_amp), machine.flux_lines[qubit_index].name, duration=t+10)
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
		save(I, I_st)
		save(Q, Q_st)
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')

			return machine, expt_dataset


	def rabi_amp(self, machine, rabi_amp_sweep_rel, qubit_index, ff_amp = 0.0, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'Amplitude'):
		"""Qubit power Rabi experiment.
		
		Qubit power Rabi experiment in 1D, sweeping amplitude of Rabi pulse, typically to calibrate a pi pulse. Note that the input argument is in relative amplitude, and return is in absolute amplitude.
		
		Args:
			machine ([type]): [description]
			rabi_amp_sweep_rel ([type]): Relative amplitude, based on pi_amp
			qubit_index ([type]): [description]
			ff_amp (number): Fast flux amplitude that overlaps with the Rabi pulse. The ff pulse is 40ns longer than the Rabi pulse. (default: `0.0`)
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `10E3`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			machine
			expt_dataset
		"""
		
		qubit_if = machine.qubits[qubit_index].f_01 - machine.octaves[0].LO_sources[1].LO_frequency
		ff_duration = machine.qubits[qubit_index].pi_length + 40

		if max(abs(rabi_amp_sweep_rel)) > 2:
			print("some relative amps > 2, removed from experiment run")
			rabi_amp_sweep_rel = rabi_amp_sweep_rel[abs(rabi_amp_sweep_rel) < 2]
		rabi_amp_sweep_abs = rabi_amp_sweep_rel * machine.qubits[qubit_index].pi_amp # actual rabi amplitude

		with program() as power_rabi:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			a = declare(fixed)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(a, rabi_amp_sweep_rel)):
					update_frequency(machine.qubits[qubit_index].name, qubit_if)
					play("const" * amp(ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
					wait(5, machine.qubits[qubit_index].name)
					play("pi" * amp(a), machine.qubits[qubit_index].name)
					wait(5, machine.qubits[qubit_index].name)
					align(machine.qubits[qubit_index].name, machine.flux_lines[qubit_index].name,
						  machine.resonators[qubit_index].name)
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					align()
					# eliminate charge accumulation
					play("const" * amp(-1 * ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					save(I, I_st)
					save(Q, Q_st)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(rabi_amp_sweep_rel)).average().save("I")
				Q_st.buffer(len(rabi_amp_sweep_rel)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)  # in clock cycles
			job = self.qmm.simulate(config, power_rabi, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(power_rabi)
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

			# Live plotting
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				
				if live_plot:
					plt.cla()
					plt.title("Qubit Power Rabi")
					if data_process_method == 'Phase':
						plt.plot(rabi_amp_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot(rabi_amp_sweep_abs, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot(rabi_amp_sweep_abs, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("Rabi Amplitude [V]")
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
			        "I": (["x"], I),
			        "Q": (["x"], Q),
			    },
			    coords={
			        "Rabi_Amplitude": (["x"], rabi_amp_sweep_abs),
			    },
			)
			
			expt_name = 'power_rabi'
			expt_long_name = 'Qubit Power Rabi'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(a, rabi_amp_sweep_rel)):
		update_frequency(machine.qubits[qubit_index].name, qubit_if)
		play("const" * amp(ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
		wait(5, machine.qubits[qubit_index].name)
		play("pi" * amp(a), machine.qubits[qubit_index].name)
		wait(5, machine.qubits[qubit_index].name)
		align(machine.qubits[qubit_index].name, machine.flux_lines[qubit_index].name,
			  machine.resonators[qubit_index].name)
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		align()
		# eliminate charge accumulation
		play("const" * amp(-1 * ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
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
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')

			return machine, expt_dataset


	def TLS_freq(self, machine, TLS_freq_sweep, qubit_index, TLS_index, pi_amp_rel = 1.0, n_avg = 1E3, cd_time_qubit = 20E3, cd_time_TLS = None, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'I', calibrate_octave = False):
		"""TLS spectroscopy experiment.
		
		TLS spectroscopy to find the TLS resonance frequency, sweeping frequency of a strong XY pulse that directly drives the TLS, then SWAP to the qubit for readout. 
		SWAP uses the iswap defined in machine.flux_lines[qubit_index].iswap.length/level[TLS_index];
		TLS pulse is a square wave defined by machine.qubits[qubit_index].pi_length_tls/pi_amp_tls[TLS_index]. These TLS pulse parameters will be passed to hardware_parameters first.
		calibrate_TLS = True could be used in the first run, to calibrate for these large amp.

		config, in particular, the intermediate_frequency is directly modified, to accommodate for very different TLS frequency from qubit frequency. Otherwise qm gets confused.
		
		Args:
			machine ([type]): [description]
			TLS_freq_sweep ([type]): [description]
			qubit_index ([type]): [description]
			TLS_index ([type]): [description]
			pi_amp_rel (number): [description] (default: `1.0`)
			n_avg (number): [description] (default: `1E3`)
			cd_time_qubit (number): [description] (default: `20E3`)
			cd_time_TLS ([type]): [description] (default: `None`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `3000`)
			final_plot (bool): [description] (default: `True`)
			live_plot (bool): [description] (default: `False`)
			data_process_method (str): [description] (default: `'I'`)
			calibrate_octave (bool): [description] (default: `False`)
		
		Returns:
			machine
			expt_dataset
		"""

		if cd_time_TLS is None:
			cd_time_TLS = cd_time_qubit

		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
		tls_if_sweep = TLS_freq_sweep - qubit_lo
		tls_if_sweep = np.floor(tls_if_sweep)

		if np.max(abs(tls_if_sweep)) > 400E6: # check if parameters are within hardware limit
			print("TLS if range > 400MHz. Setting the octave freq. Will calibrate octave.")
			machine.octaves[0].LO_sources[1].LO_frequency = int(TLS_freq_sweep.mean()) - 50E6
			qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
			tls_if_sweep = TLS_freq_sweep - qubit_lo
			tls_if_sweep = np.floor(tls_if_sweep)
			calibrate_octave = True

		# Update the hardware parameters to TLS of interest
		machine.qubits[qubit_index].hardware_parameters.pi_length_tls = machine.qubits[qubit_index].pi_length_tls[TLS_index]
		machine.qubits[qubit_index].hardware_parameters.pi_amp_tls = machine.qubits[qubit_index].pi_amp_tls[TLS_index]

		config = build_config(machine)

		def update_if_freq(new_if_freq):
			config["elements"][machine.qubits[qubit_index].name]["intermediate_frequency"] = new_if_freq

		# fLux pulse baking for SWAP
		swap_length = machine.flux_lines[qubit_index].iswap.length[TLS_index]
		swap_amp = machine.flux_lines[qubit_index].iswap.level[TLS_index]
		flux_waveform = np.array([swap_amp] * swap_length)
		def baked_swap_waveform(waveform):
			pulse_segments = []  # Stores the baking objects
			# Create the different baked sequences, each one corresponding to a different truncated duration
			with baking(config, padding_method="right") as b:
				b.add_op("flux_pulse", machine.flux_lines[qubit_index].name, waveform.tolist())
				b.play("flux_pulse", machine.flux_lines[qubit_index].name)
				pulse_segments.append(b)
			return pulse_segments

		square_TLS_swap = baked_swap_waveform(flux_waveform)
		update_if_freq(50E6) # change if frequency so it's not f_01 - LO, which is out of range.

		with program() as TLS_freq_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			df = declare(int)

			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(df,tls_if_sweep)):
					update_frequency(machine.qubits[qubit_index].name, df)
					play('pi_tls' * amp(pi_amp_rel), machine.qubits[qubit_index].name)
					align()
					square_TLS_swap[0].run()
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					wait(cd_time_qubit * u.ns, machine.resonators[qubit_index].name)
					save(I, I_st)
					save(Q, Q_st)
					# eliminate charge accumulation, also initialize TLS
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					wait(cd_time_TLS * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				n_st.save('iteration')
				I_st.buffer(len(tls_if_sweep)).average().save("I")
				Q_st.buffer(len(tls_if_sweep)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		# Simulate or execute #
		if to_simulate: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, TLS_freq_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			# calibrates octave for the TLS pulse
			if calibrate_octave:
				machine = self.set_octave.calibration(machine, qubit_index, TLS_index = TLS_index, log_flag = True, calibration_flag = True, qubit_only = True)

			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(TLS_freq_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

			# Live plotting
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
			
			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				# progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())

				if live_plot:
					plt.cla()
					plt.title("TLS Spectroscopy")
					if data_process_method == 'Phase':
						plt.plot((TLS_freq_sweep) / u.MHz, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.xlabel("TLS Frequency [MHz]")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot((TLS_freq_sweep) / u.MHz, np.abs(I + 1j * Q), ".")
						plt.xlabel("TLS Frequency [MHz]")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot((TLS_freq_sweep) / u.MHz, I, ".")
						plt.xlabel("TLS Frequency [MHz]")
						plt.ylabel("Signal I Quadrature [V]")
					plt.pause(0.5)

			# fetch all data after live-updating
			timestamp_finished = datetime.datetime.now()
			I, Q, iteration = results.fetch_all()
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
			    {
			        "I": (["x"], I),
			        "Q": (["x"], Q),
			    },
			    coords={
			        "TLS_Frequency": (["x"], TLS_freq_sweep),
			    },
			)
			
			expt_name = 'tls_spec'
			expt_long_name = 'TLS Spectroscopy'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = ['t'+str(TLS_index)] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n+1):
	with for_(*from_array(df,tls_if_sweep)):
		update_frequency(machine.qubits[qubit_index].name, df)
		play('pi_tls' * amp(pi_amp_rel), machine.qubits[qubit_index].name)
		align()
		square_TLS_swap[0].run()
		align()
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		wait(cd_time_qubit * u.ns, machine.resonators[qubit_index].name)
		save(I, I_st)
		save(Q, Q_st)
		# eliminate charge accumulation, also initialize TLS
		align()
		square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
		wait(cd_time_TLS * u.ns, machine.resonators[qubit_index].name)
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')				

			return machine, expt_dataset


	def TLS_rabi_length(self, machine, tau_sweep_abs, qubit_index, TLS_index, n_avg = 1E3, cd_time_qubit = 20E3, cd_time_TLS = None, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'I', calibrate_octave = False):
		"""TLS time Rabi experiment.
		
		TLS time Rabi experiment in 1D, sweeping length of the Rabi pulse. The state is then swapped to the qubit for readout.
		SWAP uses the iswap defined in machine.flux_lines[qubit_index].iswap.length/level[TLS_index];
		TLS pulse is a square wave defined with machine.qubits[qubit_index].pi_amp_tls[TLS_index]. These TLS pulse parameters will be passed to hardware_parameters first.
		calibrate_TLS = True could be used in the first run, to calibrate for these large amp.

		config, in particular, the intermediate_frequency is directly modified, to accommodate for very different TLS frequency from qubit frequency. Otherwise qm gets confused.
		
		Note that input argument is in ns.
		
		Args:
			machine ([type]): [description]
			tau_sweep_abs ([type]): in ns! Will be regulated to clock cycles.
			qubit_index ([type]): [description]
			TLS_index ([type]): [description]
			pi_amp_rel (number): [description] (default: `1.0`)
			n_avg (number): [description] (default: `1E3`)
			cd_time_qubit (number): [description] (default: `20E3`)
			cd_time_TLS ([type]): [description] (default: `None`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `I`)
			calibrate_octave (bool): [description] (default: `False`)
		
		Returns:
			machine
			expt_dataset
		"""

		if cd_time_TLS is None:
			cd_time_TLS = cd_time_qubit

		# regulate tau_sweep_abs into acceptable values.
		if min(tau_sweep_abs) < 16:
			print("some ramsey lengths shorter than 4 clock cycles, removed from run")
			tau_sweep_abs = tau_sweep_abs[tau_sweep_abs>15]

		tau_sweep_cc = tau_sweep_abs//4 # in clock cycles
		tau_sweep_cc = np.unique(tau_sweep_cc)
		tau_sweep = tau_sweep_cc.astype(int) # clock cycles, used for experiments
		tau_sweep_abs = tau_sweep * 4 # time in ns

		# regular LO and if frequency, and their settings in configurations.
		tls_if_freq = machine.qubits[qubit_index].f_tls[TLS_index] - machine.octaves[0].LO_sources[1].LO_frequency

		config = build_config(machine)

		if abs(tls_if_freq) > 400E6: # check if parameters are within hardware limit
			print("TLS if range > 400MHz. Setting the octave freq. Will calibrate octave.")
			machine.octaves[0].LO_sources[1].LO_frequency = machine.qubits[qubit_index].f_tls[TLS_index] - 50E6
			tls_if_freq = 50E6
			calibrate_octave = True

		if calibrate_octave:
			machine.octaves[0].LO_sources[1].LO_frequency = machine.qubits[qubit_index].f_tls[TLS_index] - 50E6
			tls_if_freq = 50E6

		# Update the hardware parameters to TLS of interest
		machine.qubits[qubit_index].hardware_parameters.pi_length_tls = machine.qubits[qubit_index].pi_length_tls[TLS_index]
		machine.qubits[qubit_index].hardware_parameters.pi_amp_tls = machine.qubits[qubit_index].pi_amp_tls[TLS_index]

		config = build_config(machine)

		def update_if_freq(new_if_freq):
			config["elements"][machine.qubits[qubit_index].name]["intermediate_frequency"] = new_if_freq

		# fLux pulse baking for SWAP
		swap_length = machine.flux_lines[qubit_index].iswap.length[TLS_index]
		swap_amp = machine.flux_lines[qubit_index].iswap.level[TLS_index]
		flux_waveform = np.array([swap_amp] * swap_length)
		def baked_swap_waveform(waveform):
			pulse_segments = []  # Stores the baking objects
			# Create the different baked sequences, each one corresponding to a different truncated duration
			with baking(config, padding_method="right") as b:
				b.add_op("flux_pulse", machine.flux_lines[qubit_index].name, waveform.tolist())
				b.play("flux_pulse", machine.flux_lines[qubit_index].name)
				pulse_segments.append(b)
			return pulse_segments

		square_TLS_swap = baked_swap_waveform(flux_waveform)
		update_if_freq(tls_if_freq)

		with program() as TLS_rabi_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			t = declare(int)

			update_frequency(machine.qubits[qubit_index].name, tls_if_freq) # important, otherwise will use the if in configuration, calculated from f_01
			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(t,tau_sweep)):
					play('pi_tls', machine.qubits[qubit_index].name, duration = t) # clock cycles
					align()
					square_TLS_swap[0].run()
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					align()
					wait(cd_time_qubit * u.ns, machine.resonators[qubit_index].name)
					save(I, I_st)
					save(Q, Q_st)
					# eliminate charge accumulation, also initialize TLS
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					wait(cd_time_TLS * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				n_st.save('iteration')
				I_st.buffer(len(tau_sweep_abs)).average().save("I")
				Q_st.buffer(len(tau_sweep_abs)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		
		# Simulate or execute #
		if to_simulate: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, TLS_rabi_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			# calibrates octave for the TLS pulse
			if calibrate_octave:
				machine = self.set_octave.calibration(machine, qubit_index, TLS_index = TLS_index, log_flag = True, calibration_flag = True, qubit_only = True)

			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(TLS_rabi_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			
			# Live plotting
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
			
			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				# progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())

				if live_plot:
					plt.cla()
					plt.title("TLS Time Rabi")
					if data_process_method == 'Phase':
						plt.plot(tau_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot(tau_sweep_abs, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot(tau_sweep_abs, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					
					plt.xlabel("Rabi Time [ns]")
					plt.pause(0.5)

			# fetch all data after live-updating
			timestamp_finished = datetime.datetime.now()
			I, Q, iteration = results.fetch_all()
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
			    {
			        "I": (["x"], I),
			        "Q": (["x"], Q),
			    },
			    coords={
			        "Rabi_Time": (["x"], tau_sweep_abs),
			    },
			)

			expt_name = 'tls_time_rabi'
			expt_long_name = 'TLS Time Rabi'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = ['t'+str(TLS_index)] # use t0, t1, t2, ...
			expt_sequence = """update_frequency(machine.qubits[qubit_index].name, tls_if_freq) # important, otherwise will use the if in configuration, calculated from f_01
with for_(n, 0, n < n_avg, n+1):
	with for_(*from_array(t,tau_sweep)):
		play('pi_tls', machine.qubits[qubit_index].name, duration = t) # clock cycles
		align()
		square_TLS_swap[0].run()
		align()
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		align()
		wait(cd_time_qubit * u.ns, machine.resonators[qubit_index].name)
		save(I, I_st)
		save(Q, Q_st)
		# eliminate charge accumulation, also initialize TLS
		align()
		square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
		wait(cd_time_TLS * u.ns, machine.resonators[qubit_index].name)
	save(n, n_st)
"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')				

			return machine, expt_dataset


	def ef_freq(self, machine, ef_freq_sweep, qubit_index, pi_amp_rel_ef = 1.0, n_avg = 1E3, cd_time = 20E3, readout_state = 'g', to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'I'):
		"""Qubit ef transition spectroscopy experiment.
		
		Qubit ef transition spectroscopy to find the ef transition frequency. Qubit is excited to e first by a ge pi pulse, then an ef driving pulse with varying frequency is sent.
		An optional ge pi pulse (if readout_state = 'g'') is applied to enhance the signal contrast.

		
		Args:
			machine ([type]): [description]
			ef_freq_sweep ([type]): 1D array of qubit ef transition frequency sweep
			qubit_index ([type]): [description]
			pi_amp_rel_ef (number): relative amplitude of pi pulse for ef transition (default: `1.0`)
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `10E3`)
			readout_state (str): If 'g' (default), a ge pi pulse before readout brings Pe to Pg; If 'e', no additional pulse is applied. (default: `'g'`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `I`)
		
		Returns:
			machine
			expt_dataset
		"""

		if readout_state not in ['g','e']:
			print('Readout state not g/e. Abort...')
			return machine, None

		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
		qubit_if = machine.qubits[qubit_index].f_01 - machine.octaves[0].LO_sources[1].LO_frequency
		ef_if_sweep = ef_freq_sweep - qubit_lo
		ef_if_sweep = np.round(ef_if_sweep)

		if abs(qubit_if) > 400E6:
			print("qubit if > 400MHz")
			return machine, None
		if abs(qubit_if) < 20E6:
			print("qubit if < 20MHz")
			return machine, None
		if np.max(abs(ef_if_sweep)) > 400E6: # check if parameters are within hardware limit
			print("ef if range > 400MHz")
			return machine, None
		if np.min(abs(ef_if_sweep)) < 20E6: # check if parameters are within hardware limit
			print("ef if range < 20MHz")
			return machine, None

		with program() as ef_freq_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			df = declare(int)
			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(df,ef_if_sweep)):
					update_frequency(machine.qubits[qubit_index].name, qubit_if)
					play('pi', machine.qubits[qubit_index].name)
					update_frequency(machine.qubits[qubit_index].name, df)
					
					if pi_amp_rel_ef==1.0:
						play('pi_ef', machine.qubits[qubit_index].name)
					else:
						play('pi_ef' * amp(pi_amp_rel_ef), machine.qubits[qubit_index].name)
					
					if readout_state == 'g':
						update_frequency(machine.qubits[qubit_index].name, qubit_if)
						play('pi', machine.qubits[qubit_index].name)
					
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					save(I, I_st)
					save(Q, Q_st)
				save(n, n_st)
			with stream_processing():
				n_st.save('iteration')
				I_st.buffer(len(ef_if_sweep)).average().save("I")
				Q_st.buffer(len(ef_if_sweep)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		# Simulate or execute #
		if to_simulate: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, ef_freq_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(ef_freq_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
		    #%matplotlib qt
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				# progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())

				if live_plot:
					plt.cla()
					plt.title(r"Qubit e-f Spectroscopy")
					if data_process_method == 'Phase':
						plt.plot((ef_freq_sweep) / u.MHz, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot((ef_freq_sweep) / u.MHz, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot((ef_freq_sweep) / u.MHz, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("ef Frequency [MHz]")
					plt.pause(0.5)


			# fetch all data after live-updating
			timestamp_finished = datetime.datetime.now()
			I, Q, _ = results.fetch_all()
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
			    {
			        "I": (["x"], I),
			        "Q": (["x"], Q),
			    },
			    coords={
			        "Qubit_Frequency": (["x"], ef_freq_sweep),
			    },
			)
			
			expt_dataset.attrs["readout_state"] = readout_state # save unique attr for this experiment

			expt_name = 'ef_spec'
			expt_long_name = 'Qubit e-f Spectroscopy'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n+1):
	with for_(*from_array(df,ef_if_sweep)):
		update_frequency(machine.qubits[qubit_index].name, qubit_if)
		play('pi', machine.qubits[qubit_index].name)
		update_frequency(machine.qubits[qubit_index].name, df)
		
		if pi_amp_rel_ef==1.0:
			play('pi_ef', machine.qubits[qubit_index].name)
		else:
			play('pi_ef' * amp(pi_amp_rel_ef), machine.qubits[qubit_index].name)
		
		if readout_state == 'g':
			update_frequency(machine.qubits[qubit_index].name, qubit_if)
			play('pi', machine.qubits[qubit_index].name)
		
		align()
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
		save(I, I_st)
		save(Q, Q_st)
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')

			return machine, expt_dataset


	def ef_rabi_length(self, machine, tau_sweep_abs, qubit_index, pi_amp_rel_ef = 1.0, n_avg = 1E3, cd_time = 20E3, readout_state = 'g', to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'I'):
		"""Qubit ef transition time Rabi experiment
		
		Qubit ef transition time Rabi experiment in 1D, sweeping length of the Rabi pulse. Qubit is excited to e first by a ge pi pulse, then an ef driving pulse with varying length is sent.
		An optional ge pi pulse (if readout_state = 'g'') is applied to enhance the signal contrast.
		Note that input argument is in ns (in older version it's clock cycle)!
		
		Args:
			machine ([type]): [description]
			tau_sweep_abs ([type]): in ns!
			qubit_index ([type]): [description]
			pi_amp_rel_ef (number): [description] (default: `1.0`)
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `10E3`)
			readout_state (str): If 'g' (default), a ge pi pulse before readout brings Pe to Pg; If 'e', no additional pulse is applied. (default: `'g'`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `I`)
		
		Returns:
			machine
			expt_dataset
		"""

		if readout_state not in ['g','e']:
			print('Readout state not g/e. Abort...')
			return machine, None

		if min(tau_sweep_abs) < 16:
			print("some ramsey lengths shorter than 4 clock cycles, removed from run")
			tau_sweep_abs = tau_sweep_abs[tau_sweep_abs>15]

		tau_sweep_cc = tau_sweep_abs//4 # in clock cycles
		tau_sweep_cc = np.unique(tau_sweep_cc)
		tau_sweep = tau_sweep_cc.astype(int) # clock cycles, used for experiments
		tau_sweep_abs = tau_sweep * 4 # time in ns

		qubit_if = machine.qubits[qubit_index].f_01 - machine.octaves[0].LO_sources[1].LO_frequency
		ef_if = machine.qubits[qubit_index].f_01 - machine.qubits[qubit_index].anharmonicity - machine.octaves[0].LO_sources[1].LO_frequency

		if abs(qubit_if) > 400E6:
			print("qubit if > 400MHz")
			return machine, None
		if abs(qubit_if) < 20E6:
			print("qubit if < 20MHz")
			return machine, None
		if abs(ef_if) > 400E6:
			print("ef if > 400MHz")
			return machine, None
		if abs(ef_if) < 20E6:
			print("ef if < 20MHz")
			return machine, None

		with program() as time_rabi_ef:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, tau_sweep)):
					update_frequency(machine.qubits[qubit_index].name, qubit_if)
					play('pi', machine.qubits[qubit_index].name)
					update_frequency(machine.qubits[qubit_index].name, ef_if)
					
					if pi_amp_rel_ef==1.0:
						play('pi_ef', machine.qubits[qubit_index].name, duration=t) # clock cycle
					else:
						play('pi_ef' * amp(pi_amp_rel_ef), machine.qubits[qubit_index].name, duration=t) # clock cycle

					if readout_state == 'g':
						update_frequency(machine.qubits[qubit_index].name, qubit_if)
						play('pi', machine.qubits[qubit_index].name)
					
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)
			with stream_processing():
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)  # in clock cycles
			job = self.qmm.simulate(config, time_rabi_ef, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(time_rabi_ef)
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

			# Live plotting
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				
				if live_plot:
					plt.cla()
					plt.title("Qubit e-f Time Rabi")
					if data_process_method == 'Phase':
						plt.plot(tau_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot(tau_sweep_abs, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot(tau_sweep_abs, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("Rabi Time [ns]")
					plt.pause(0.5)

			# fetch all data after live-updating
			timestamp_finished = datetime.datetime.now()
			I, Q, _ = results.fetch_all()
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
			    {
			        "I": (["x"], I),
			        "Q": (["x"], Q),
			    },
			    coords={
			        "Rabi_Time": (["x"], tau_sweep_abs),
			    },
			)

			expt_dataset.attrs["readout_state"] = readout_state # save unique attr for this experiment
			
			expt_name = 'time_rabi_ef'
			expt_long_name = 'Qubit e-f Time Rabi'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(t, tau_sweep)):
		update_frequency(machine.qubits[qubit_index].name, qubit_if)
		play('pi', machine.qubits[qubit_index].name)
		update_frequency(machine.qubits[qubit_index].name, ef_if)
		
		if pi_amp_rel_ef==1.0:
			play('pi_ef', machine.qubits[qubit_index].name, duration=t) # clock cycle
		else:
			play('pi_ef' * amp(pi_amp_rel_ef), machine.qubits[qubit_index].name, duration=t) # clock cycle

		if readout_state == 'g':
			update_frequency(machine.qubits[qubit_index].name, qubit_if)
			play('pi'*amp(pi_amp_rel), machine.qubits[qubit_index].name)
		
		align()
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		save(I, I_st)
		save(Q, Q_st)
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')

			return machine, expt_dataset


	def ef_rabi_amp(self, machine, rabi_amp_sweep_rel, qubit_index, n_avg = 1E3, cd_time = 20E3, readout_state = 'g', to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'I'):
		"""Qubit ef transition power Rabi experiment.
		
		Qubit ef transition power Rabi experiment in 1D, sweeping amplitude of Rabi pulse, typically to calibrate an ef pi pulse. 
		Qubit is excited to e first by a ge pi pulse, then an ef driving pulse with varying amplitude is sent.
		An optional ge pi pulse (if readout_state = 'g'') is applied to enhance the signal contrast.
		Note that the input argument is in relative amplitude, and return is in absolute amplitude.
		
		Args:
			machine ([type]): [description]
			rabi_amp_sweep_rel ([type]): Relative amplitude, based on pi_amp_ef
			qubit_index ([type]): [description]
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `10E3`)
			readout_state (str): If 'g' (default), a ge pi pulse before readout brings Pe to Pg; If 'e', no additional pulse is applied. (default: `'g'`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `I`)
		
		Returns:
			machine
			expt_dataset
		"""

		if readout_state not in ['g','e']:
			print('Readout state not g/e. Abort...')
			return machine, None

		qubit_if = machine.qubits[qubit_index].f_01 - machine.octaves[0].LO_sources[1].LO_frequency
		rabi_amp_sweep_abs = rabi_amp_sweep_rel * machine.qubits[qubit_index].pi_amp_ef # actual rabi amplitude
		ef_if = machine.qubits[qubit_index].f_01 - machine.qubits[qubit_index].anharmonicity - machine.octaves[0].LO_sources[1].LO_frequency

		if abs(qubit_if) > 400E6:
			print("qubit if > 400MHz")
			return None
		if abs(qubit_if) < 20E6:
			print("qubit if < 20MHz")
			return None
		if abs(ef_if) > 400E6:
			print("ef if > 400MHz")
			return None
		if abs(ef_if) < 20E6:
			print("ef if < 20MHz")
			return None

		with program() as power_rabi_ef:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			a = declare(fixed)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(a, rabi_amp_sweep_rel)):
					update_frequency(machine.qubits[qubit_index].name, qubit_if)
					play('pi', machine.qubits[qubit_index].name)
					update_frequency(machine.qubits[qubit_index].name, ef_if)
					play('pi_ef' * amp(a), machine.qubits[qubit_index].name)
					
					if readout_state == 'g':
						update_frequency(machine.qubits[qubit_index].name, qubit_if)
						play('pi', machine.qubits[qubit_index].name)
					
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name, I, Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(rabi_amp_sweep_rel)).average().save("I")
				Q_st.buffer(len(rabi_amp_sweep_rel)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)  # in clock cycles
			job = self.qmm.simulate(config, power_rabi_ef, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(power_rabi_ef)
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

			# Live plotting
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				
				if live_plot:
					plt.cla()
					plt.title("Qubit e-f Power Rabi")
					if data_process_method == 'Phase':
						plt.plot(rabi_amp_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot(rabi_amp_sweep_abs, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot(rabi_amp_sweep_abs, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("Rabi Amplitude [V]")
					plt.pause(0.5)

			# fetch all data after live-updating
			timestamp_finished = datetime.datetime.now()
			I, Q, _ = results.fetch_all()
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
			    {
			        "I": (["x"], I),
			        "Q": (["x"], Q),
			    },
			    coords={
			        "Rabi_Amplitude": (["x"], rabi_amp_sweep_abs),
			    },
			)

			expt_dataset.attrs["readout_state"] = readout_state # save unique attr for this experiment
			
			expt_name = 'power_rabi_ef'
			expt_long_name = 'Qubit e-f Power Rabi'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(a, rabi_amp_sweep_rel)):
		update_frequency(machine.qubits[qubit_index].name, qubit_if)
		play('pi', machine.qubits[qubit_index].name)
		update_frequency(machine.qubits[qubit_index].name, ef_if)
		play('pi_ef' * amp(a), machine.qubits[qubit_index].name)
		
		if readout_state == 'g':
			update_frequency(machine.qubits[qubit_index].name, qubit_if)
			play('pi', machine.qubits[qubit_index].name)
		
		align()
		readout_rotated_macro(machine.resonators[qubit_index].name, I, Q)
		save(I, I_st)
		save(Q, Q_st)
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')

			return machine, expt_dataset


	def ef_rabi_length_thermal(self, machine, tau_sweep_abs, qubit_index, n_avg = 1E3, cd_time = 20E3, readout_state = 'e', to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'I'):
		"""Qubit ef Rabi experiment. 
		
		Qubit from the g state is directly driven by an ef pulse with varying length. 
		An optional ge pi pulse (if readout_state = 'g'') is applied to enhance the signal contrast.
		This is to measure the oscillation of residual |e> state, A_sig in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.114.240501
		Together with ef_rabi_length experiment, one can extract the residual thermal Pe.

		
		Args:
			machine ([type]): [description]
			tau_sweep_abs ([type]): in ns!
			qubit_index ([type]): [description]
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `10E3`)
			readout_state (str): If 'g' (default), a ge pi pulse before readout brings Pe to Pg; If 'e', no additional pulse is applied. (default: `'e'`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `I`)
		
		Returns:
			[type]: [description]
		"""	

		if readout_state not in ['g','e']:
			print('Readout state not g/e. Abort...')
			return machine, None

		if min(tau_sweep_abs) < 16:
			print("some ramsey lengths shorter than 4 clock cycles, removed from run")
			tau_sweep_abs = tau_sweep_abs[tau_sweep_abs>15]

		tau_sweep_cc = tau_sweep_abs//4 # in clock cycles
		tau_sweep_cc = np.unique(tau_sweep_cc)
		tau_sweep = tau_sweep_cc.astype(int) # clock cycles, used for experiments
		tau_sweep_abs = tau_sweep * 4 # time in ns

		qubit_if = machine.qubits[qubit_index].f_01 - machine.octaves[0].LO_sources[1].LO_frequency
		ef_if = machine.qubits[qubit_index].f_01 - machine.qubits[qubit_index].anharmonicity - machine.octaves[0].LO_sources[1].LO_frequency

		if abs(qubit_if) > 400E6:
			print("qubit if > 400MHz")
			return machine, None
		if abs(qubit_if) < 20E6:
			print("qubit if < 20MHz")
			return machine, None
		if abs(ef_if) > 400E6:
			print("ef if > 400MHz")
			return machine, None
		if abs(ef_if) < 20E6:
			print("ef if < 20MHz")
			return machine, None

		with program() as time_rabi_ef_thermal:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, tau_sweep)):
					update_frequency(machine.qubits[qubit_index].name, ef_if)
					play('pi_ef' * amp(pi_amp_rel_ef), machine.qubits[qubit_index].name, duration=t)
					if readout_state == 'g':
						update_frequency(machine.qubits[qubit_index].name, qubit_if)
						play('pi', machine.qubits[qubit_index].name)
					
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)
			with stream_processing():
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)  # in clock cycles
			job = self.qmm.simulate(config, time_rabi_ef_thermal, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(time_rabi_ef_thermal)
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

			# Live plotting
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				
				if live_plot:
					plt.cla()
					plt.title("Residual e-state Time Rabi")
					if data_process_method == 'Phase':
						plt.plot(tau_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot(tau_sweep_abs, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot(tau_sweep_abs, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("Rabi Time [ns]")
					plt.pause(0.5)

			# fetch all data after live-updating
			timestamp_finished = datetime.datetime.now()
			I, Q, _ = results.fetch_all()
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
			    {
			        "I": (["x"], I),
			        "Q": (["x"], Q),
			    },
			    coords={
			        "Rabi_Time": (["x"], tau_sweep_abs),
			    },
			)

			expt_dataset.attrs["readout_state"] = readout_state # save unique attr for this experiment
			
			expt_name = 'time_rabi_ef_thermal'
			expt_long_name = 'Residual e-state Time Rabi'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(t, tau_sweep)):
		update_frequency(machine.qubits[qubit_index].name, ef_if)
		play('pi_ef' * amp(pi_amp_rel_ef), machine.qubits[qubit_index].name, duration=t)
		if readout_state == 'g':
			update_frequency(machine.qubits[qubit_index].name, qubit_if)
			play('pi', machine.qubits[qubit_index].name)
		
		align()
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		save(I, I_st)
		save(Q, Q_st)
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')

			return machine, expt_dataset

	def qubit_switch_delay(self, machine, qubit_switch_delay_sweep, qubit_index, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'Amplitude'):
		"""Calibrate RF switch delay for qubit
		
		[description]
		
		Args:
			machine ([type]): [description]
			qubit_switch_delay_sweep ([type]): [description]
			qubit_index ([type]): [description]
			n_avg ([type]): [description]
			cd_time ([type]): [description]
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			machine
			expt_dataset
		"""
		

		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
		qubit_if = machine.qubits[qubit_index].f_01 - qubit_lo

		if abs(qubit_if) > 400E6: # check if parameters are within hardware limit
			print("qubit if > 400MHz")
			return machine, None

		with program() as qubit_switch_delay_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()

			with for_(n, 0, n < n_avg, n+1):
				update_frequency(machine.qubits[qubit_index].name, qubit_if)
				play('pi2', machine.qubits[qubit_index].name)
				align()
				readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(I, I_st)
				save(Q, Q_st)
			with stream_processing():
				I_st.average().save("I")
				Q_st.average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		# Simulate or execute #
		if to_simulate: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, qubit_switch_delay_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			start_time = time.time()
			I_tot = []
			Q_tot = []
			if final_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]

			for delay_index, delay_value in enumerate(qubit_switch_delay_sweep):
				machine = self.set_digital_delay(machine, "qubits", int(delay_value))
				config = build_config(machine)

				qm = self.qmm.open_qm(config)
				timestamp_created = datetime.datetime.now()
				job = qm.execute(qubit_switch_delay_prog)
				# Get results from QUA program
				results = fetching_tool(job, data_list=["I", "Q"], mode = "live")

				if final_plot:
					interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
				while results.is_processing():
					# Fetch results
					time.sleep(0.1)

				I, Q = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				I_tot.append(I)
				Q_tot.append(Q)

				# progress bar
				progress_counter(delay_index, len(qubit_switch_delay_sweep), start_time=start_time)

			I_tot = np.array(I_tot)
			Q_tot = np.array(Q_tot)
			sigs_qubit = I_tot + 1j * Q_tot
			sig_amp = np.abs(sigs_qubit)  # Amplitude
			sig_phase = np.angle(sigs_qubit)  # Phase

			if final_plot:
				plt.cla()
				plt.title("qubit switch delay")
				plt.plot(qubit_switch_delay_sweep, sig_amp, ".")
				plt.xlabel("switch delay [ns]")
				plt.ylabel("Signal Amplitude [V]")

			# save data
			exp_name = 'qubit_switch_delay'
			qubit_name = 'Q' + str(qubit_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"qubit_delay": qubit_switch_delay_sweep, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset


	def qubit_switch_buffer(self, machine, qubit_switch_buffer_sweep, qubit_index, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'Amplitude'):
		"""Calibrate RF switch buffer for qubit.
		
		The buffer accounts for rise/fall time, and is added to both sides of the switch time (x2).
		
		Args:
			machine ([type]): [description]
			qubit_switch_buffer_sweep ([type]): in ns
			qubit_index ([type]): [description]
			n_avg ([type]): [description]
			cd_time ([type]): [description]
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key in expt_dataset to be used. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			machine, expt_dataset
		"""

		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
		qubit_if = machine.qubits[qubit_index].f_01 - qubit_lo

		if abs(qubit_if) > 400E6: # check if parameters are within hardware limit
			print("qubit if > 400MHz")
			return machine, None

		with program() as qubit_switch_buffer_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()

			with for_(n, 0, n < n_avg, n+1):
				update_frequency(machine.qubits[qubit_index].name, qubit_if)
				play('pi2', machine.qubits[qubit_index].name)
				align()
				readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(I, I_st)
				save(Q, Q_st)
			with stream_processing():
				I_st.average().save("I")
				Q_st.average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		# Simulate or execute #
		if to_simulate: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, qubit_switch_buffer_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			start_time = time.time()
			I_tot = []
			Q_tot = []
			if final_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]

			for buffer_index, buffer_value in enumerate(qubit_switch_buffer_sweep):
				machine = self.set_digital_buffer(machine, "qubits", int(buffer_value))
				config = build_config(machine)

				qm = self.qmm.open_qm(config)
				timestamp_created = datetime.datetime.now()
				job = qm.execute(qubit_switch_buffer_prog)
				# Get results from QUA program
				results = fetching_tool(job, data_list=["I", "Q"], mode = "live")

				if final_plot:
					interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
				while results.is_processing():
					# Fetch results
					time.sleep(0.1)

				I, Q = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				I_tot.append(I)
				Q_tot.append(Q)

				# progress bar
				progress_counter(buffer_index, len(qubit_switch_buffer_sweep), start_time=start_time)

			I_tot = np.array(I_tot)
			Q_tot = np.array(Q_tot)
			sigs_qubit = I_tot + 1j * Q_tot
			sig_amp = np.abs(sigs_qubit)  # Amplitude
			sig_phase = np.angle(sigs_qubit)  # Phase

			if final_plot:
				plt.cla()
				plt.title("qubit switch buffer")
				plt.plot(qubit_switch_buffer_sweep, sig_amp, ".")
				plt.xlabel("switch buffer [ns]")
				plt.ylabel("Signal Amplitude [V]")

			# save data
			exp_name = 'qubit_switch_buffer'
			qubit_name = 'Q' + str(qubit_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"qubit_buffer": qubit_switch_buffer_sweep, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset



