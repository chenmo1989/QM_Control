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
from macros import *
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
import xarray as xr

class EH_Ramsey:
	def __init__(self, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm):
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs
		self.qmm = ref_to_qmm

	def ramsey(self, machine, tau_sweep_abs, qubit_index, n_avg = 1E3, detuning = 1E6, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'Amplitude'):
		"""Run qubit ramsey experiment.
		
		Detuning realized by a virtual rotation on the phase of the second pi/2 pulse. 
		tau_sweep_abs in ns. Values not at multiples of clock cycles will be removed.
		
		Args:
			machine ([type]): [description]
			tau_sweep_abs ([type]): in ns! Values not multiples of clock cycles will be regulated and removed.
			qubit_index ([type]): [description]
			n_avg (number): [description] (default: `1E3`)
			detuning (number): [description] (default: `1E6`)
			cd_time (number): [description] (default: `10E3`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key to be plotted. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			machine
			expt_dataset
		"""
		
		if min(tau_sweep_abs) < 16:
			print("some ramsey lengths shorter than 4 clock cycles, removed from run")
			tau_sweep_abs = tau_sweep_abs[tau_sweep_abs>15]

		tau_sweep_cc = tau_sweep_abs//4 # in clock cycles
		tau_sweep_cc = np.unique(tau_sweep_cc)
		tau_sweep = tau_sweep_cc.astype(int) # clock cycles, used for experiments
		tau_sweep_abs = tau_sweep * 4 # time in ns
	
		with program() as ramsey_vr:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)
			phase = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, tau_sweep)):
					assign(phase, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
					with strict_timing_():
						play("pi2", machine.qubits[qubit_index].name)
						wait(t, machine.qubits[qubit_index].name)
						frame_rotation_2pi(phase, machine.qubits[qubit_index].name)
						play("pi2", machine.qubits[qubit_index].name)
					align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					reset_frame(machine.qubits[qubit_index].name) # to avoid phase accumulation
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)  # in clock cycles
			job = self.qmm.simulate(config, ramsey_vr, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(ramsey_vr)
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
				#time.sleep(0.05)

				if live_plot:
					plt.cla()
					plt.title(f"Ramsey with detuning = {detuning/1E6:.1f} MHz")

					if data_process_method is 'Phase':
						plt.plot(tau_sweep_abs, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method is 'Amplitude':
						plt.plot(tau_sweep_abs, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method is 'I':
						plt.plot(tau_sweep_abs, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("Free Evolution Time [ns]")
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
			        "Free_Evolution_Time": (["x"], tau_sweep_abs),
			    },
			)
			
			expt_name = 'ramsey'
			expt_long_name = 'Qubit Ramsey'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(t, tau_sweep)):
		assign(phase, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
		with strict_timing_():
			play("pi2", machine.qubits[qubit_index].name)
			wait(t, machine.qubits[qubit_index].name)
			frame_rotation_2pi(phase, machine.qubits[qubit_index].name)
			play("pi2", machine.qubits[qubit_index].name)
		align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		save(I, I_st)
		save(Q, Q_st)
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
		reset_frame(machine.qubits[qubit_index].name) # to avoid phase accumulation
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


	def TLS_ramsey(self, machine, ramsey_duration_sweep, qubit_index, TLS_index, n_avg = 1E3, detuning = 1E6, cd_time_qubit = 20E3, cd_time_TLS = None, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False):
		"""
		TLS Ramsey in 1D. Detuning realized by tuning the phase of second pi/2 pulse
		sequence given by pi/2 - wait - pi/2 for various wait times
		the frame of the last pi/2 pulse is rotated rather than using actual driving freq. detuning

		:param machine
		:param ramsey_duration_sweep: in clock cycles!
		:param qubit_index:
		:param TLS_index:
		:param n_avg:
		:param detuning:
		:param cd_time_qubit:
		:param cd_time_TLS:
		:param to_simulate:
		:param simulation_len:
		:param final_plot:
		
		:return:
			machine
			ramsey_duration_sweep * 4: in ns!
			sig_amp
		"""
		if cd_time_TLS is None:
			cd_time_TLS = cd_time_qubit

		

		# important, need to update if for qua program
		TLS_if = machine.qubits[qubit_index].f_tls[TLS_index] - machine.octaves[0].LO_sources[1].LO_frequency

		# Update the hardware parameters to TLS of interest
		machine.qubits[qubit_index].hardware_parameters.pi_length_tls = machine.qubits[qubit_index].pi_length_tls[TLS_index]
		machine.qubits[qubit_index].hardware_parameters.pi_amp_tls = machine.qubits[qubit_index].pi_amp_tls[TLS_index]

		if min(ramsey_duration_sweep) < 4:
			print("some ramsey lengths shorter than 4 clock cycles, removed from run")
			ramsey_duration_sweep = ramsey_duration_sweep[ramsey_duration_sweep>3]

		ramsey_duration_sweep = ramsey_duration_sweep.astype(int)
		
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

		with program() as tls_ramsey:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)
			phase = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)

			update_frequency(machine.qubits[qubit_index].name, TLS_if) # important, otherwise will use the if in configuration, calculated from f_01
			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, ramsey_duration_sweep)):
					assign(phase, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
					with strict_timing_():
						play("pi2_tls", machine.qubits[qubit_index].name)
						wait(t, machine.qubits[qubit_index].name)
						frame_rotation_2pi(phase, machine.qubits[qubit_index].name)
						play("pi2_tls", machine.qubits[qubit_index].name)
					align()
					square_TLS_swap[0].run()
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					align()
					wait(cd_time_qubit * u.ns, machine.resonators[qubit_index].name)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					wait(cd_time_TLS * u.ns, machine.resonators[qubit_index].name)
					reset_frame(machine.qubits[qubit_index].name) # to avoid phase accumulation
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(ramsey_duration_sweep)).average().save("I")
				Q_st.buffer(len(ramsey_duration_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)  # in clock cycles
			job = self.qmm.simulate(config, tls_ramsey, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(tls_ramsey)
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
					plt.title("Ramsey with detuning = %i MHz" % (detuning/1E6))
					#plt.plot(ramsey_duration_sweep * 4, sig_amp, "b.")
					plt.plot(ramsey_duration_sweep * 4, sig_amp, "b.")
					plt.xlabel("tau [ns]")
					#plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [V]")
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
			exp_name = 'ramsey'
			qubit_name = 'Q' + str(qubit_index + 1) + 'TLS' + str(TLS_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"TLS_ramsey_duration": ramsey_duration_sweep * 4, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset
