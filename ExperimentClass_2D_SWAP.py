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
	"""
	class in ExperimentHandle, for SWAP related 2D experiments
	Methods:
		update_tPath
		update_str_datetime
	"""

	def __init__(self, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm):
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs
		self.qmm = ref_to_qmm

	def swap_coarse(self, machine, tau_sweep_abs, ff_sweep_abs, qubit_index, n_avg = 1E3, cd_time = 20E3, to_simulate=False, simulation_len=3000, final_plot=True, live_plot = False, data_process_method = 'I'):
		"""Run 2D qubit SWAP spectroscopy, sweeping fast flux amplitude and duration. 4ns time resolution.
		
		Note that time resolution is 4ns (clock cycle), hence the name `coarse`. tau > 16ns is required.
		
		Args:
			machine ([type]): [description]
			tau_sweep_abs ([type]): Interaction time sweep, in ns. Will be regulated to multiples of 4ns, starting from 16ns.
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
		

		ff_sweep_rel = ff_sweep_abs / machine.flux_lines[qubit_index].flux_pulse_amp
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
					with for_(*from_array(da, ff_sweep_rel)):
						play("pi", machine.qubits[qubit_index].name)
						align()
						play("const" * amp(da), machine.flux_lines[qubit_index].name, duration = t) # in clock cycle!
						align()
						readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
						wait(cd_time * u.ns, machine.resonators[qubit_index].name)
						save(I, I_st)
						save(Q, Q_st)
						align()
						play("const" * amp(-da), machine.flux_lines[qubit_index].name, duration = t)
						align()
						wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				# for the progress counter
				n_st.save("iteration")
				I_st.buffer(len(ff_sweep_rel)).buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(ff_sweep_rel)).buffer(len(tau_sweep)).average().save("Q")

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
						plt.pcolormesh(ff_sweep_abs, tau_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), cmap="seismic")
					elif data_process_method == 'Amplitude':
						plt.pcolormesh(ff_sweep_abs, tau_sweep_abs, np.abs(I + 1j * Q), cmap="seismic")
					elif data_process_method == 'I':
						plt.pcolormesh(ff_sweep_abs, tau_sweep_abs, I, cmap="seismic")

					plt.title("SWAP Spectroscopy")
					plt.xlabel("Fast Flux [V]")
					plt.ylabel("Interaction Time [ns]")
					plt.show()
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
			        "I": (["x", "y"], I),
			        "Q": (["x", "y"], Q),
			    },
			    coords={
			        "Fast_Flux": (["y"], ff_sweep_abs),
			        "Interaction_Time": (["x"], tau_sweep_abs),
			    },
			)
			
			expt_name = 'swap2D'
			expt_long_name = 'SWAP Spectroscopy'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(t, tau_sweep)):
		with for_(*from_array(da, ff_sweep_rel)):
			play("pi", machine.qubits[qubit_index].name)
			align()
			play("const" * amp(da), machine.flux_lines[qubit_index].name, duration = t) # in clock cycle!
			align()
			readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
			wait(cd_time * u.ns, machine.resonators[qubit_index].name)
			save(I, I_st)
			save(Q, Q_st)
			align()
			play("const" * amp(-da), machine.flux_lines[qubit_index].name, duration = t) 
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
				expt_dataset[data_process_method].plot(x = list(expt_dataset.coords.keys())[0], y = list(expt_dataset.coords.keys())[1], cmap = "seismic")
				plt.title(expt_dataset.attrs['long_name'])

			return machine, expt_dataset


	def swap(self, machine, tau_sweep_abs, ff_sweep_abs, qubit_index, n_avg = 1E3, cd_time = 20E3, to_simulate=False, simulation_len=3000, final_plot=True, live_plot = False, data_process_method = 'I'):
		"""Run 2D qubit SWAP spectroscopy, sweeping fast flux amplitude and duration. 1ns time resolution.
		
		Note that time resolution is 1ns with baking. Allows tau < 16ns. (Used to be called `swap_fine`)
		
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
		ff_sweep_rel = ff_sweep_abs / machine.flux_lines[qubit_index].flux_pulse_amp
		
		# set up variables
		max_pulse_duration = int(max(tau_sweep_abs))
		min_pulse_duration = int(min(tau_sweep_abs))
		dt_pulse_duration = int(tau_sweep_abs[2] - tau_sweep_abs[1])

		# fLux pulse waveform generation, 250 ns/points
		flux_waveform = np.array([machine.flux_lines[qubit_index].flux_pulse_amp] * max_pulse_duration)

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
				with for_(*from_array(da, ff_sweep_rel)):
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
				I_st.buffer(len(tau_sweep_abs)).buffer(len(ff_sweep_rel)).average().save("I")
				Q_st.buffer(len(tau_sweep_abs)).buffer(len(ff_sweep_rel)).average().save("Q")

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
						plt.pcolormesh(ff_sweep_abs, tau_sweep_abs, np.transpose(np.unwrap(np.angle(I + 1j * Q))), cmap="seismic")
					elif data_process_method == 'Amplitude':
						plt.pcolormesh(ff_sweep_abs, tau_sweep_abs, np.transpose(np.abs(I + 1j * Q)), cmap="seismic")
					elif data_process_method == 'I':
						plt.pcolormesh(ff_sweep_abs, tau_sweep_abs, np.transpose(I), cmap="seismic")

					plt.title("SWAP Spectroscopy")
					plt.xlabel("Fast Flux [V]")
					plt.ylabel("Interaction Time [ns]")
					plt.show()
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
			        "I": (["x", "y"], I),
			        "Q": (["x", "y"], Q),
			    },
			    coords={
			        "Fast_Flux": (["x"], ff_sweep_abs),
			        "Interaction_Time": (["y"], tau_sweep_abs),
			    },
			)
			
			expt_name = 'swap2D'
			expt_long_name = 'SWAP Spectroscopy'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(da, ff_sweep_rel)):
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
				expt_dataset[data_process_method].plot(x = list(expt_dataset.coords.keys())[0], y = list(expt_dataset.coords.keys())[1], cmap = "seismic")
				plt.title(expt_dataset.attrs['long_name'])

			return machine, expt_dataset


