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


class EH_SWAP:


	def __init__(self, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm):
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs
		self.qmm = ref_to_qmm


	def rabi_SWAP(self, machine, rabi_duration_sweep, qubit_index, TLS_index, pi_amp_rel = 1.0, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False):
		"""
		1D experiment: qubit rabi (length sweep) - SWAP - measure
		qubit rabi duration in clock cycle

		Args:
			machine
			rabi_duration_sweep (): in clock cycle!
			qubit_index ():
			TLS_index ():
			pi_amp_rel ():
			n_avg ():
			cd_time ():
			to_simulate ():
			simulation_len ():
			final_plot ():

		Returns:
			machine
			rabi_duration_sweep * 4: in ns
			sig_amp
		"""
		

		if min(rabi_duration_sweep) < 4:
			print("some rabi lengths shorter than 4 clock cycles, removed from run")
			rabi_duration_sweep = rabi_duration_sweep[rabi_duration_sweep>3]

		rabi_duration_sweep = rabi_duration_sweep.astype(int)

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

		with program() as time_rabi:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, rabi_duration_sweep)):
					wait(5, machine.qubits[qubit_index].name)
					play("pi" * amp(pi_amp_rel), machine.qubits[qubit_index].name, duration=t)
					wait(5, machine.qubits[qubit_index].name)
					align()
					square_TLS_swap[0].run()
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					align()
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(rabi_duration_sweep)).average().save("I")
				Q_st.buffer(len(rabi_duration_sweep)).average().save("Q")
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
			if final_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				sig_amp = np.sqrt(I ** 2 + Q ** 2)
				sig_phase = np.angle(I + 1j * Q)
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				if final_plot:
					plt.cla()
					plt.title("Time Rabi")
					plt.plot(rabi_duration_sweep * 4, sig_amp, "b.")
					plt.xlabel("tau [ns]")
					plt.ylabel("Signal Amplitude [V]")
					plt.pause(0.02)

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
			sig_amp = np.sqrt(I ** 2 + Q ** 2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'time_rabi'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"Q_rabi_duration": rabi_duration_sweep * 4, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset


	def swap(self, machine, tau_sweep_abs, qubit_index, TLS_index, n_avg = 1E3, cd_time = 20E3, to_simulate=False, simulation_len=3000, final_plot=True, live_plot = False, data_process_method = 'I'):
		"""Run 1D qubit SWAP spectroscopy, i.e., vacuum-Rabi between qubit and TLS, sweeping fast flux pulse duration. 1ns time resolution.
		
		Note that time resolution is 1ns with baking. Allows tau < 16ns. 
		Require flux amp determined (from 2D swap), and saved as iswap.level/length for the TLS_index.
		
		Args:
			machine ([type]): [description]
			tau_sweep_abs ([type]): [description]
			ff_sweep_abs ([type]): [description]
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


		config = build_config(machine) # must be here before def of baking
		
		# set up variables
		max_pulse_duration = int(max(tau_sweep_abs))
		min_pulse_duration = int(min(tau_sweep_abs))
		dt_pulse_duration = int(tau_sweep_abs[2] - tau_sweep_abs[1])

		# fLux pulse waveform generation, 250 ns/points
		flux_waveform = np.array([machine.flux_lines[qubit_index].iswap.level[TLS_index]] * max_pulse_duration) # flux pulse amp = iswap.level[TLS_index]

		def baked_ff_waveform(waveform, pulse_duration):
			pulse_segments = []  # Stores the baking objects
			# Create the different baked sequences, each one corresponding to a different truncated duration
			for i in range(0, pulse_duration + 1):
				with baking(config, padding_method="right") as b:
					if i == 0:  # Otherwise, the baking will be empty and will not be created
						wf = [0.0] * 16
					else:
						wf = waveform[:i].tolist()
					b.add_op("flux_pulse", machine.flux_lines[qubit_index].name, wf)
					b.play("flux_pulse", machine.flux_lines[qubit_index].name)
				# Append the baking object in the list to call it from the QUA program
				pulse_segments.append(b)
			return pulse_segments

		# Baked flux pulse segments
		square_pulse_segments = baked_ff_waveform(flux_waveform, max_pulse_duration)

		with program() as iswap:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)
			da = declare(fixed)
			segment = declare(int)  # Flux pulse segment

			with for_(n, 0, n < n_avg, n + 1):
				with for_(segment, min_pulse_duration, segment <= max_pulse_duration, segment + dt_pulse_duration):
					play("pi", machine.qubits[qubit_index].name)
					align()
					with switch_(segment):
						for j in range(min_pulse_duration,max_pulse_duration+1, dt_pulse_duration):
							with case_(j):
								square_pulse_segments[j].run(amp_array=[(machine.flux_lines[qubit_index].name, da)])
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					save(I, I_st)
					save(Q, Q_st)
					align()
					with switch_(segment):
						for j in range(min_pulse_duration,max_pulse_duration+1,dt_pulse_duration):
							with case_(j):
								square_pulse_segments[j].run(amp_array=[(machine.flux_lines[qubit_index].name, -da)])
					align()
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				# for the progress counter
				n_st.save("iteration")
				I_st.buffer(len(tau_sweep_abs)).average().save("I")
				Q_st.buffer(len(tau_sweep_abs)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, iswap, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(iswap)
			results = fetching_tool(job, ["I", "Q", "iteration"], mode="live")

			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8,4]
				interrupt_on_close(fig, job)

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				# progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				
				if live_plot:
					plt.cla()

					if data_process_method == 'Phase':
						plt.plot(tau_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), '.')
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot(tau_sweep_abs, np.abs(I + 1j * Q), '.')
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot(tau_sweep_abs, I, '.')
						plt.ylabel("Signal I Quadrature [V]")

					plt.title("SWAP Spectroscopy")
					plt.xlabel("Interaction Time [ns]")
					plt.pause(0.5)

			# fetch all data after live-updating
			timestamp_finished = datetime.datetime.now()
			I, Q, iteration = results.fetch_all()
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
			        "Interaction_Time": (["x"], tau_sweep_abs),
			    },
			)
			
			expt_name = 'swap1D'
			expt_long_name = 'SWAP Spectroscopy'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(segment, min_pulse_duration, segment <= max_pulse_duration, segment + dt_pulse_duration):
		play("pi", machine.qubits[qubit_index].name)
		align()
		with switch_(segment):
			for j in range(min_pulse_duration,max_pulse_duration+1, dt_pulse_duration):
				with case_(j):
					square_pulse_segments[j].run(amp_array=[(machine.flux_lines[qubit_index].name, da)])
		align()
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
		save(I, I_st)
		save(Q, Q_st)
		align()
		with switch_(segment):
			for j in range(min_pulse_duration,max_pulse_duration+1,dt_pulse_duration):
				with case_(j):
					square_pulse_segments[j].run(amp_array=[(machine.flux_lines[qubit_index].name, -da)])
		align()
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
	save(n, n_st)"""

			expt_extra = {
				'n_ave': str(n_avg),
				'Qubit CD [ns]': str(cd_time)
			}

			# save data
			self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence, expt_extra)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')				

			return machine, expt_dataset


	def swap_coarse(self, machine, tau_sweep_abs, qubit_index, TLS_index, n_avg = 1E3, cd_time = 20E3, to_simulate=False, simulation_len=3000, final_plot=True, live_plot = False, data_process_method = 'I'):
		"""
		1D SWAP spectroscopy. qubit pi - SWAP (sweep Z duration) - measure
		tau_sweep in ns, only takes multiples of 4ns

		Args:
			machine
			tau_sweep_abs ():
			qubit_index ():
			TLS_index ():
			n_avg ():
			cd_time ():
			to_simulate ():
			simulation_len ():
			final_plot ():

		Returns:
			machine
			tau_sweep_abs
			sig_amp
		"""		

		tau_sweep_cc = tau_sweep_abs // 4  # in clock cycles
		tau_sweep_cc = np.unique(tau_sweep_cc)
		tau_sweep = tau_sweep_cc.astype(int)  # clock cycles
		tau_sweep_abs = tau_sweep * 4  # time in ns

		with program() as iswap:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)
			da = declare(fixed)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, tau_sweep)):
					play("pi", machine.qubits[qubit_index].name)
					align()
					play("iswap", machine.flux_lines[qubit_index].name, duration=t)
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					align()
					wait(50)
					play("iswap" * amp(-1), machine.flux_lines[qubit_index].name, duration=t)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				# for the progress counter
				n_st.save("iteration")
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, iswap, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(iswap)
			results = fetching_tool(job, ["I", "Q", "iteration"], mode="live")

			if final_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8,4]
				interrupt_on_close(fig, job)

			while results.is_processing():
				I, Q, iteration = results.fetch_all()
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				time.sleep(0.1)

			I, Q, _ = results.fetch_all()
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
			sig_amp = np.sqrt(I ** 2 + Q ** 2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'SWAP1D'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"ff_amp": machine.flux_lines[qubit_index].iswap.level[TLS_index], "sig_amp": sig_amp, "sig_phase": sig_phase,
					 "tau_sweep": tau_sweep_abs})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			if final_plot:
				plt.cla()
				plt.plot(tau_sweep_abs, sig_amp)
				plt.ylabel("Signal Amplitude (V)")
				plt.xlabel("interaction time (ns)")

		return machine, expt_dataset


	def SWAP_rabi(self, machine, rabi_duration_sweep, qubit_index, TLS_index, pi_amp_rel = 1.0, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False):
		"""
		1D experiment for debug: SWAP - qubit rabi (sweep duration) - measure

		Args:
			machine
			rabi_duration_sweep (): in clock cycle
			qubit_index ():
			TLS_index ():
			pi_amp_rel ():
			n_avg ():
			cd_time ():
			to_simulate ():
			simulation_len ():
			final_plot ():
			
		Returns:
			machine
			rabi_duration_sweep * 4
			sig_amp
		"""

		

		if min(rabi_duration_sweep) < 4:
			print("some rabi lengths shorter than 4 clock cycles, removed from run")
			rabi_duration_sweep = rabi_duration_sweep[rabi_duration_sweep>3]
		
		rabi_duration_sweep = rabi_duration_sweep.astype(int)

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

		with program() as time_rabi:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, rabi_duration_sweep)):
					square_TLS_swap[0].run()
					align()
					wait(5, machine.qubits[qubit_index].name)
					play("pi" * amp(pi_amp_rel), machine.qubits[qubit_index].name, duration=t)
					wait(5, machine.qubits[qubit_index].name)
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					align()
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(rabi_duration_sweep)).average().save("I")
				Q_st.buffer(len(rabi_duration_sweep)).average().save("Q")
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
			if final_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				sig_amp = np.sqrt(I ** 2 + Q ** 2)
				sig_phase = np.angle(I + 1j * Q)
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				if final_plot:
					plt.cla()
					plt.title("Time Rabi")
					plt.plot(rabi_duration_sweep * 4, sig_amp, "b.")
					plt.xlabel("tau [ns]")
					plt.ylabel("Signal Amplitude [V]")
					plt.pause(0.02)

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
			sig_amp = np.sqrt(I ** 2 + Q ** 2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'time_rabi'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"Q_rabi_duration": rabi_duration_sweep * 4, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset


	def rabi_SWAP2(self, machine, rabi_duration_sweep, qubit_index, TLS_index, pi_amp_rel = 1.0, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False):
		"""
		1D experiment: qubit rabi (sweep duration) - SWAP - SWAP, to see if the state comes back

		Args:
			machine
			rabi_duration_sweep (): in clock cycle
			qubit_index ():
			TLS_index ():
			pi_amp_rel ():
			n_avg ():
			cd_time ():
			to_simulate ():
			simulation_len ():
			final_plot ():
			
		Returns:
			machine
			rabi_duration_sweep * 4
			sig_amp
		"""
		

		if min(rabi_duration_sweep) < 4:
			print("some rabi lengths shorter than 4 clock cycles, removed from run")
			rabi_duration_sweep = rabi_duration_sweep[rabi_duration_sweep>3]

		rabi_duration_sweep = rabi_duration_sweep.astype(int)

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

		with program() as time_rabi:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, rabi_duration_sweep)):
					wait(5, machine.qubits[qubit_index].name)
					play("pi" * amp(pi_amp_rel), machine.qubits[qubit_index].name, duration=t)
					wait(5, machine.qubits[qubit_index].name)
					align()
					square_TLS_swap[0].run()
					wait(5)
					square_TLS_swap[0].run()
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					wait(5)
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					align()
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(rabi_duration_sweep)).average().save("I")
				Q_st.buffer(len(rabi_duration_sweep)).average().save("Q")
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
			if final_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				sig_amp = np.sqrt(I ** 2 + Q ** 2)
				sig_phase = np.angle(I + 1j * Q)
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				if final_plot:
					plt.cla()
					plt.title("Time Rabi")
					plt.plot(rabi_duration_sweep * 4, sig_amp, "b.")
					plt.xlabel("tau [ns]")
					plt.ylabel("Signal Amplitude [V]")
					plt.pause(0.02)

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
			sig_amp = np.sqrt(I ** 2 + Q ** 2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'time_rabi'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"Q_rabi_duration": rabi_duration_sweep * 4, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset
