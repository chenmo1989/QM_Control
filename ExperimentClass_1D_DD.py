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


class EH_DD:
	def __init__(self, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm):
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs
		self.qmm = ref_to_qmm

	def TLS_echo(self, machine, tau_sweep_abs, qubit_index, TLS_index, pi_over_2_phase = 'y', n_avg = 1E3, cd_time_qubit = 20E3, cd_time_TLS = None, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'I', calibrate_octave = False):
		"""Run TLS echo experiment.
		
		[description]
		
		Args:
			machine ([type]): [description]
			tau_sweep_abs ([type]): [description]
			qubit_index ([type]): [description]
			TLS_index ([type]): [description]
			pi_over_2_phase (str): [description] (default: `'y'`)
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
			[type]: [description]
		"""

		if pi_over_2_phase not in ['x','y']:
			print("pi_over_2_phase must be either x or y. Abort...")
			return machine, None

		if cd_time_TLS is None:
			cd_time_TLS = cd_time_qubit

		# regulate the free evolution time in ramsey
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

		with program() as tls_echo:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			update_frequency(machine.qubits[qubit_index].name, tls_if_freq) # important, otherwise will use the if in configuration, calculated from f_01
			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, tau_sweep)):
					with strict_timing_():
						if pi_over_2_phase=='x':
							play("pi2_tls", machine.qubits[qubit_index].name)
						else:
							play("pi2y_tls", machine.qubits[qubit_index].name)
						
						wait(t, machine.qubits[qubit_index].name)
						play("pi_tls", machine.qubits[qubit_index].name)
						wait(t, machine.qubits[qubit_index].name)

						if pi_over_2_phase == 'x':
							play("pi2_tls", machine.qubits[qubit_index].name)
						else:
							play("pi2y_tls", machine.qubits[qubit_index].name)

					align()
					square_TLS_swap[0].run()
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					wait(cd_time_qubit * u.ns, machine.resonators[qubit_index].name)
					save(I, I_st)
					save(Q, Q_st)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					wait(cd_time_TLS * u.ns, machine.flux_lines[qubit_index].name)
				save(n, n_st)	

			with stream_processing():
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")
				n_st.save("iteration")

		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)  # in clock cycles
			job = self.qmm.simulate(config, tls_echo, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			# calibrates octave for the TLS pulse
			if calibrate_octave:
				machine = self.set_octave.calibration(machine, qubit_index, TLS_index=TLS_index, log_flag=True,
													  calibration_flag=True, qubit_only=True)

			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(tls_echo)
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
					plt.title("TLS Echo")

					if data_process_method == 'Phase':
						plt.plot(tau_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot(tau_sweep_abs, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot(tau_sweep_abs, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("Half Pulse Interval [ns]")
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
			        "Half_Pulse_Interval": (["x"], tau_sweep_abs),
			    },
			)
			
			expt_name = 'tls_echo'
			expt_long_name = 'TLS Echo'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = ['t'+str(TLS_index)] # use t0, t1, t2, ...
			expt_sequence = """update_frequency(machine.qubits[qubit_index].name, tls_if_freq) # important, otherwise will use the if in configuration, calculated from f_01
with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(t, tau_sweep)):
		with strict_timing_():
			if pi_over_2_phase=='x':
				play("pi2_tls", machine.qubits[qubit_index].name)
			else:
				play("pi2y_tls", machine.qubits[qubit_index].name)
			
			wait(t, machine.qubits[qubit_index].name)
			play("pi_tls", machine.qubits[qubit_index].name)
			wait(t, machine.qubits[qubit_index].name)

			if pi_over_2_phase == 'x':
				play("pi2_tls", machine.qubits[qubit_index].name)
			else:
				play("pi2y_tls", machine.qubits[qubit_index].name)

		align()
		square_TLS_swap[0].run()
		align()
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		wait(cd_time_qubit * u.ns, machine.resonators[qubit_index].name)
		save(I, I_st)
		save(Q, Q_st)
		align()
		square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
		wait(cd_time_TLS * u.ns, machine.flux_lines[qubit_index].name)
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')
				plt.title(expt_dataset.attrs['long_name'])

			return machine, expt_dataset


	def TLS_CPMG(self, machine, tau_sweep_abs, qubit_index, TLS_index, pi_over_2_phase = 'y', N_CPMG = 8, n_avg = 1E3, cd_time_qubit = 20E3, cd_time_TLS = None, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'I', calibrate_octave = False):
		"""Run TLS CPMG experiment. pi pulse # determined by N_CPMG.
		
		[description]
		
		Args:
			machine ([type]): [description]
			tau_sweep_abs ([type]): [description]
			qubit_index ([type]): [description]
			TLS_index ([type]): [description]
			pi_over_2_phase (str): [description] (default: `'y'`)
			N_CPMG (number): [description] (default: `8`)
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
			[type]: [description]
		"""
		
		if pi_over_2_phase not in ['x','y']:
			print("pi_over_2_phase must be x or y. Abort...")
			return machine, None

		if cd_time_TLS is None:
			cd_time_TLS = cd_time_qubit

		# regulate the free evolution time in ramsey
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

		with program() as tls_echo:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			update_frequency(machine.qubits[qubit_index].name, tls_if_freq) # important, otherwise will use the if in configuration, calculated from f_01
			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, tau_sweep)):
					with strict_timing_():
						if pi_over_2_phase=='x':
							play("pi2_tls", machine.qubits[qubit_index].name)
						else:
							play("pi2y_tls", machine.qubits[qubit_index].name)

						wait(t, machine.qubits[qubit_index].name)

						for i in range(N_CPMG - 1):
							play("pi_tls", machine.qubits[qubit_index].name)
							wait(t * 2, machine.qubits[qubit_index].name)

						play("pi_tls", machine.qubits[qubit_index].name)
						wait(t, machine.qubits[qubit_index].name)

						if pi_over_2_phase=='x':
							play("pi2_tls", machine.qubits[qubit_index].name)
						else:
							play("pi2y_tls", machine.qubits[qubit_index].name)

					align()
					square_TLS_swap[0].run()
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					wait(cd_time_qubit * u.ns, machine.resonators[qubit_index].name)
					save(I, I_st)
					save(Q, Q_st)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					wait(cd_time_TLS * u.ns, machine.flux_lines[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")
				n_st.save("iteration")

		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)  # in clock cycles
			job = self.qmm.simulate(config, tls_echo, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			# calibrates octave for the TLS pulse
			if calibrate_octave:
				machine = self.set_octave.calibration(machine, qubit_index, TLS_index=TLS_index, log_flag=True,
													  calibration_flag=True, qubit_only=True)

			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(tls_echo)
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

			# Live plotting
			if final_plot:
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
				
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				
				if live_plot:
					plt.cla()
					plt.title("TLS CPMG" + '-' + str(N_CPMG))

					if data_process_method == 'Phase':
						plt.plot(tau_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot(tau_sweep_abs, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot(tau_sweep_abs, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("Half Pulse Interval [ns]")
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
			        "Half_Pulse_Interval": (["x"], tau_sweep_abs),
			    },
			)
			
			expt_name = 'tls_cpmg' + str(N_CPMG)
			expt_long_name = 'TLS CPMG' + '-' + str(N_CPMG)
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = ['t'+str(TLS_index)] # use t0, t1, t2, ...
			expt_sequence = """update_frequency(machine.qubits[qubit_index].name, tls_if_freq) # important, otherwise will use the if in configuration, calculated from f_01
with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(t, tau_sweep)):
		with strict_timing_():
			if pi_over_2_phase=='x':
				play("pi2_tls", machine.qubits[qubit_index].name)
			else:
				play("pi2y_tls", machine.qubits[qubit_index].name)

			wait(t, machine.qubits[qubit_index].name)

			for i in range(N_CPMG - 1):
				play("pi_tls", machine.qubits[qubit_index].name)
				wait(t * 2, machine.qubits[qubit_index].name)

			play("pi_tls", machine.qubits[qubit_index].name)
			wait(t, machine.qubits[qubit_index].name)

			if pi_over_2_phase=='x':
				play("pi2_tls", machine.qubits[qubit_index].name)
			else:
				play("pi2y_tls", machine.qubits[qubit_index].name)

		align()
		square_TLS_swap[0].run()
		align()
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		wait(cd_time_qubit * u.ns, machine.resonators[qubit_index].name)
		save(I, I_st)
		save(Q, Q_st)
		align()
		square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
		wait(cd_time_TLS * u.ns, machine.flux_lines[qubit_index].name)
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')
				plt.title(expt_dataset.attrs['long_name'])

			return machine, expt_dataset

			