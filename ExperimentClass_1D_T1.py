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


class EH_T1:
	def __init__(self, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm):
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs
		self.qmm = ref_to_qmm
		

	def qubit_T1(self, machine, tau_sweep_abs, qubit_index, ff_amp = 0.0, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'Amplitude'):
		"""Run Qubit T1 measurement.
		
		Currently at fixed 0 fast flux. 
		tau_sweep_abs is in ns. Values not at multiples of clock cycles will be removed.
		
		Args:
			machine ([type]): in ns. Is regulated to integer clock cycles before running experiment.
			tau_sweep_abs ([type]): [description]
			qubit_index ([type]): [description]
			ff_amp (number): Fast flux amplitude that overlaps with the pi pulse.
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `10E3`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key to be plotted. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			machine
			expt_dataset
		"""
		
		tau_sweep_cc = tau_sweep_abs//4 # in clock cycles
		tau_sweep_cc = np.unique(tau_sweep_cc)
		tau_sweep = tau_sweep_cc.astype(int) # clock cycles
		tau_sweep_abs = tau_sweep * 4 # time in ns


		with program() as t1_prog:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			tau = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(tau, tau_sweep)):
					play("const" * amp(ff_amp), machine.flux_lines[qubit_index].name, duration=tau+(machine.qubits[qubit_index].pi_length//4)+10)
					wait(5, machine.qubits[qubit_index].name)
					play("pi", machine.qubits[qubit_index].name)
					wait(tau, machine.qubits[qubit_index].name)
					align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name, machine.flux_lines[qubit_index].name)
					wait(5, machine.resonators[qubit_index].name)
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					if ff_amp != 0:
						# eliminate charge accumulation
						play("const" * amp(-1 * ff_amp), machine.flux_lines[qubit_index].name, duration=tau+(machine.qubits[qubit_index].pi_length//4)+10)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)
			with stream_processing():
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)

		# Simulate or execute #
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, t1_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(t1_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [12, 8]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure    while results.is_processing():

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				#time.sleep(0.05)

				if live_plot:
					plt.cla()
					plt.title("T1")
					if data_process_method == 'Phase':
						plt.plot(tau_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot(tau_sweep_abs, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot(tau_sweep_abs, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("tau [ns]")
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
			        "Relaxation_Time": (["x"], tau_sweep_abs),
			    },
			)
			
			expt_name = 'T1'
			expt_long_name = 'Qubit T1'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(tau, tau_sweep)):
					play("const" * amp(ff_amp), machine.flux_lines[qubit_index].name, duration=tau+(machine.qubits[qubit_index].pi_length//4)+10)
					wait(5, machine.qubits[qubit_index].name)
					play("pi", machine.qubits[qubit_index].name)
					wait(tau, machine.qubits[qubit_index].name)
					align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					if ff_amp != 0:
						# eliminate charge accumulation
						play("const" * amp(-1 * ff_amp), machine.flux_lines[qubit_index].name, duration=tau+(machine.qubits[qubit_index].pi_length//4)+10)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)"""

			expt_extra = {
				'ff_amp [V]': str(ff_amp),
				'n_ave': str(n_avg),
				'Qubit CD [ns]': str(cd_time)
			}
			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence, expt_extra)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')
				plt.title(expt_dataset.attrs['long_name'])

			return machine, expt_dataset


	def TLS_T1(self, machine, tau_sweep_abs, qubit_index, TLS_index, n_avg = 1E3, cd_time_qubit = 20E3, cd_time_TLS = None, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False,  data_process_method = 'I'):
		"""TLS T1 measurement with iswap.
		
		qubit pi pulse -- iswap -- delay -- iswap -- readout.
		Opposite sign iswaps are applied at the end of the sequence, with delay of cd_time_qubit, cd_time_TLS after each opposite iswap. 
		4 iswaps in total.
		Turns out a few hundred us cd_time_TLS is needed, to get high quality TLS T1 measurement. Likely due to build up in transients?

		Args:
			machine ([type]): [description]
			tau_sweep_abs ([type]): [description]
			qubit_index ([type]): [description]
			TLS_index ([type]): [description]
			n_avg (number): [description] (default: `1E3`)
			cd_time_qubit (number): [description] (default: `20E3`)
			cd_time_TLS ([type]): [description] (default: `None`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `3000`)
			final_plot (bool): [description] (default: `True`)
			live_plot (bool): [description] (default: `False`)
			data_process_method (str): [description] (default: `'I'`)
		
		Returns:
			[type]: [description]
		"""

		config = build_config(machine)

		if cd_time_TLS is None:
			cd_time_TLS = cd_time_qubit

		swap_length = machine.flux_lines[qubit_index].iswap.length[TLS_index]
		swap_amp = machine.flux_lines[qubit_index].iswap.level[TLS_index]
		tau_sweep_cc = tau_sweep_abs//4 # in clock cycles
		tau_sweep_cc = np.unique(tau_sweep_cc)
		tau_sweep = tau_sweep_cc.astype(int) # clock cycles
		tau_sweep_abs = tau_sweep * 4 # time in ns

		# fLux pulse baking for SWAP
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

		with program() as t1_prog:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			tau = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(tau, tau_sweep)):
					play("pi", machine.qubits[qubit_index].name)
					align()
					square_TLS_swap[0].run()
					wait(tau, machine.flux_lines[qubit_index].name)
					square_TLS_swap[0].run()
					align()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					wait(cd_time_qubit * u.ns, machine.resonators[qubit_index].name)
					save(I, I_st)
					save(Q, Q_st)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					wait(cd_time_qubit * u.ns, machine.flux_lines[qubit_index].name)
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					wait(cd_time_TLS * u.ns, machine.flux_lines[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")
				n_st.save("iteration")

		# Simulate or execute #
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, t1_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(t1_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
			if live_plot is True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [12, 8]
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
					plt.title("TLS T1 (swap)")
					if data_process_method == 'Phase':
						plt.plot(tau_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot(tau_sweep_abs, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot(tau_sweep_abs, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("tau [ns]")
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
			        "Relaxation_Time": (["x"], tau_sweep_abs),
			    },
			)
			
			expt_name = 'tls_T1_swap'
			expt_long_name = r'TLS T1 (swap)'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = ['t'+str(TLS_index)] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(tau, tau_sweep)):
		play("pi", machine.qubits[qubit_index].name)
		align()
		square_TLS_swap[0].run()
		wait(tau, machine.flux_lines[qubit_index].name)
		square_TLS_swap[0].run()
		align()
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		wait(cd_time_qubit * u.ns, machine.resonators[qubit_index].name)
		save(I, I_st)
		save(Q, Q_st)
		align()
		square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
		wait(cd_time_qubit * u.ns, machine.flux_lines[qubit_index].name)
		square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
		wait(cd_time_TLS * u.ns, machine.flux_lines[qubit_index].name)
	save(n, n_st)"""

			expt_extra = {
				'n_ave': str(n_avg),
				'Qubit CD [ns]': str(cd_time_qubit),
				'TLS CD [ns]': str(cd_time_TLS)
			}

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence, expt_extra)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')
				plt.title(expt_dataset.attrs['long_name'])

			return machine, expt_dataset


	def TLS_T1_drive(self, machine, tau_sweep_abs, qubit_index, TLS_index, n_avg = 1E3, cd_time_qubit = 10E3, cd_time_TLS = None, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'I', calibrate_octave = False):
		"""[TLS T1 measurement with TLS drive pulse.
		
		TLS pi pulse -- delay -- iswap -- readout.
		Opposite sign iswaps are applied at the end of the sequence, with delay of cd_time_qubit, cd_time_TLS after the opposite iswap. 
		2 iswaps in total.
		
		Args:
			machine ([type]): [description]
			tau_sweep_abs ([type]): [description]
			qubit_index ([type]): [description]
			TLS_index ([type]): [description]
			n_avg (number): [description] (default: `1E3`)
			cd_time_qubit (number): [description] (default: `10E3`)
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

		with program() as t1_prog:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			tau = declare(int)

			update_frequency(machine.qubits[qubit_index].name, tls_if_freq) # important, otherwise will use the if in configuration, calculated from f_01
			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(tau, tau_sweep)):
					play("pi_tls", machine.qubits[qubit_index].name)
					wait(tau, machine.qubits[qubit_index].name)
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

		# Simulate or execute #
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, t1_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			# calibrates octave for the TLS pulse
			if calibrate_octave:
				machine = self.set_octave.calibration(machine, qubit_index, TLS_index=TLS_index, log_flag=True,
													  calibration_flag=True, qubit_only=True)

			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(t1_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [12, 8]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure    while results.is_processing():
			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())

				if live_plot:
					plt.cla()
					plt.title("TLS T1 (drive)")
					if data_process_method == 'Phase':
						plt.plot(tau_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method == 'Amplitude':
						plt.plot(tau_sweep_abs, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method == 'I':
						plt.plot(tau_sweep_abs, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("tau [ns]")
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
			        "Relaxation_Time": (["x"], tau_sweep_abs),
			    },
			)
			
			expt_name = 'tls_T1_drive'
			expt_long_name = r'TLS T1 (drive)'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = ['t'+str(TLS_index)] # use t0, t1, t2, ...
			expt_sequence = """wupdate_frequency(machine.qubits[qubit_index].name, tls_if_freq) # important, otherwise will use the if in configuration, calculated from f_01
with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(tau, tau_sweep)):
		play("pi_tls", machine.qubits[qubit_index].name)
		wait(tau, machine.qubits[qubit_index].name)
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

			expt_extra = {
				'n_ave': str(n_avg),
				'Qubit CD [ns]': str(cd_time_qubit),
				'TLS CD [ns]': str(cd_time_TLS)
			}
			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence, expt_extra)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')
				plt.title(expt_dataset.attrs['long_name'])

			return machine, expt_dataset
