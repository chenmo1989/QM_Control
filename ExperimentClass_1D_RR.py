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
from macros import *
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
import xarray as xr

warnings.filterwarnings("ignore")

class EH_RR: # sub
	"""
	class in ExperimentHandle, for Readout Resonator (RR) related 1D experiments
	Methods:
		
	"""


	def __init__(self, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm):
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs
		self.qmm = ref_to_qmm


	def time_of_flight(self, machine, qubit_index, n_avg = 5E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000):
		"""
		time of flight 1D experiment
		this experiment calibrates
			1. the time delay between when the RO tone is sent, and when it is received by the adc/digitizer.
				Note it could be different for different qubit, due to resonator Q. Although I do not calibrate for individual qubits.
			2. the dc offset for I, Q readout signal.
			3. if we need to adjust the attenuation of RO on the output side, s.t. the signal fully span the +-0.5V adc range, but does not exceed it (and be cutoff)

		Args:
		:param machine:
		:param qubit_index:
		:param n_avg: repetition of expeirment
		:param cd_time: cooldown time between subsequent experiments
		:param to_simulate: True-run simulation; False (default)-run experiment.
		:param simulation_len: Length of the sequence to simulate. In clock cycles (4ns).
		:param machine: None (default), will read from quam_state.json
		Return:
			machine
			adc1
			adc2
			adc1_single_run
			adc2_single_run
		"""
		with program() as raw_trace_prog:
			n = declare(int)
			adc_st = declare_stream(adc_trace=True)
			n_st = declare_stream()

			with for_(n, 0, n < n_avg, n + 1):
				reset_phase(machine.resonators[qubit_index].name)
				measure("readout", machine.resonators[qubit_index].name, adc_st)
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n,n_st)
			with stream_processing():
				# Will save average:
				adc_st.input1().average().save("I")
				adc_st.input2().average().save("Q")
				# # Will save only last run:
				adc_st.input1().save("I_single_run")
				adc_st.input2().save("Q_single_run")
				n_st.save('iteration')

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, raw_trace_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(raw_trace_prog)

			# Fetch all data
			results = fetching_tool(job, data_list=["I", "Q", "I_single_run", "Q_single_run", "iteration"], mode = "wait_for_all") # wait_for_all is the default
			adc1, adc2, adc1_single_run, adc2_single_run, iteration = results.fetch_all()
			timestamp_finished = datetime.datetime.now()
			progress_counter(iteration, n_avg, start_time=datetime.datetime.timestamp(timestamp_created))

			# Convert I & Q to Volts
			adc1 = u.raw2volts(adc1)
			adc2 = u.raw2volts(adc2)
			adc1_single_run = u.raw2volts(adc1_single_run)
			adc2_single_run = u.raw2volts(adc2_single_run)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
				{
					"I": (["x"], adc1),
					"Q": (["x"], adc2),
					"I_single_run": (["x"], adc1_single_run),
					"Q_single_run": (["x"], adc2_single_run),
				},
				coords={
					"Time": (["x"], np.arange(0,len(adc1),1)),
				},
			)

			expt_name = 'time_of_flight'
			expt_long_name = 'Time of Flight'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = []  # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	reset_phase(machine.resonators[qubit_index].name)
	measure("readout", machine.resonators[qubit_index].name, adc_st)
	wait(cd_time * u.ns, machine.resonators[qubit_index].name)
	save(n,n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
											  expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			return machine, expt_dataset


	def rr_freq(self, machine, res_freq_sweep, qubit_index, n_avg = 1E3, cd_time = 20E3, readout_state = 'g', to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'Amplitude'):
		"""Run resonator spectroscopy experiment.
		
		Pulsed resonator spectroscopy for either the qubit in the g(round)- or e(xcited)-state. 
		Intended for finding the resonance frequency, by localizing the minima in pulsed transmission signal.
		
		Args:
			machine ([type]): [description]
			res_freq_sweep ([type]): [description]
			qubit_index ([type]): [description]
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `20E3`)
			readout_state (str): If 'e'/'f', qubit is prepared to |e> or |f> then readout. If anything else, return error. (default: `'g'`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key to be plotted. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			[type]: [description]
		"""

		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		res_if_sweep = res_freq_sweep - res_lo
		res_if_sweep = np.floor(res_if_sweep)

		if np.max(abs(res_if_sweep)) > 400E6: # check if parameters are within hardware limit
			print("res if range > 400MHz")
			return machine, None

		with program() as rr_freq_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			df = declare(int)

			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(df,res_if_sweep)):
					if readout_state == 'g':
						pass
					elif readout_state == 'e':
						play("pi", machine.qubits[qubit_index].name)
						align()
					elif readout_state == 'f':
						play("pi", machine.qubits[qubit_index].name)
						play("pi_ef", machine.qubits[qubit_index].name)
						align()
					else:
						print("readout state does not exist")
						return
					update_frequency(machine.resonators[qubit_index].name, df)
					readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					save(I, I_st)
					save(Q, Q_st)
				save(n, n_st)
			with stream_processing():
				n_st.save('iteration')
				I_st.buffer(len(res_if_sweep)).average().save("I")
				Q_st.buffer(len(res_if_sweep)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		# Simulate or execute #
		if to_simulate: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, rr_freq_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(rr_freq_prog)
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
					plt.title("Resonator spectroscopy")
					if data_process_method is 'Phase':
						plt.plot(res_freq_sweep, np.unwrap(np.angle(I + 1j * Q)), ".")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method is 'Amplitude':
						plt.plot(res_freq_sweep, np.abs(I + 1j * Q), ".")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method is 'I':
						plt.plot(res_freq_sweep, I, ".")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("Frequency [MHz]")
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
					"Readout_Frequency": (["x"], res_freq_sweep),
				},
			)
			
			expt_dataset.attrs["readout_state"] = readout_state # save unique attr for this experiment

			expt_name = 'res_spec'
			expt_long_name = 'Resonator Spectroscopy'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = []  # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n+1):
	with for_(*from_array(df,res_if_sweep)):
		if readout_state == 'g':
			pass
		elif readout_state == 'e':
			play("pi", machine.qubits[qubit_index].name)
			align()
		elif readout_state == 'f':
			play("pi", machine.qubits[qubit_index].name)
			play("pi_ef", machine.qubits[qubit_index].name)
			align()
		else:
			print("readout state does not exist")
			return
		update_frequency(machine.resonators[qubit_index].name, df)
		readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
		save(I, I_st)
		save(Q, Q_st)
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
											  expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker = '.')

			return machine, expt_dataset


	def rr_freq_ge(self, machine, res_freq_sweep, qubit_index, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'Phase'):
		"""Run resonator spectroscopy experiment for qubit in g and e.
		
		Pulsed resonator spectroscopy for both the qubit in the ground state, and then in the excited state. 
		Intended for finding the readout frequency, by localizing the max difference in signal phase. This readout frequency is best for single-shot readout.
		
		Args:
			machine ([type]): [description]
			res_freq_sweep ([type]): [description]
			qubit_index ([type]): [description]
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `20E3`)
			readout_state (str): If 'e'/'f', qubit is prepared to |e> or |f> then readout. If anything else, return error. (default: `'g'`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key to be plotted. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Amplitude`)
		
		Returns:
			[type]: [description]
		"""

		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		res_if_sweep = res_freq_sweep - res_lo
		res_if_sweep = np.floor(res_if_sweep)

		if np.max(abs(res_if_sweep)) > 400E6: # check if parameters are within hardware limit
			print("res if range > 400MHz")
			return machine, None

		with program() as rr_freq_prog:
			n = declare(int)
			I_g = declare(fixed)
			Q_g = declare(fixed)
			I_g_st = declare_stream()
			Q_g_st = declare_stream()
			I_e = declare(fixed)
			Q_e = declare(fixed)
			I_e_st = declare_stream()
			Q_e_st = declare_stream()
			n_st = declare_stream()

			df = declare(int)

			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(df,res_if_sweep)):
					update_frequency(machine.resonators[qubit_index].name, df)
					# measure g					
					measure("readout",
						machine.resonators[qubit_index].name,
						None,
						dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
						dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
					)
					save(I_g, I_g_st)
					save(Q_g, Q_g_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)

					# measure e
					align()  # global align
					play("pi", machine.qubits[qubit_index].name)
					align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
					measure(
						"readout",
						machine.resonators[qubit_index].name,
						None,
						dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
						dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
						)
					save(I_e, I_e_st)
					save(Q_e, Q_e_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)
				
			with stream_processing():
				n_st.save('iteration')
				I_g_st.buffer(len(res_if_sweep)).average().save("I_g")
				Q_g_st.buffer(len(res_if_sweep)).average().save("Q_g")
				I_e_st.buffer(len(res_if_sweep)).average().save("I_e")
				Q_e_st.buffer(len(res_if_sweep)).average().save("Q_e")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		# Simulate or execute #
		if to_simulate: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, rr_freq_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(rr_freq_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I_g", "Q_g", "I_e", "Q_e", "iteration"], mode="live")
			# Live plotting
		    #%matplotlib qt
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				Ig, Qg, Ie, Qe, iteration = results.fetch_all()
				Ig = u.demod2volts(Ig, machine.resonators[qubit_index].readout_pulse_length)
				Qg = u.demod2volts(Qg, machine.resonators[qubit_index].readout_pulse_length)
				Ie = u.demod2volts(Ie, machine.resonators[qubit_index].readout_pulse_length)
				Qe = u.demod2volts(Qe, machine.resonators[qubit_index].readout_pulse_length)
				# progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())

				if live_plot:
					plt.cla()
					plt.title("Resonator spectroscopy")
					if data_process_method is 'Phase':
						plt.plot((res_freq_sweep) / u.MHz, np.unwrap(np.angle(Ig + 1j * Qg)), ".", label="Ground")
						plt.plot((res_freq_sweep) / u.MHz, np.unwrap(np.angle(Ie + 1j * Qe)), ".", label="Excited")
						plt.ylabel("Signal Phase [rad]")
					elif data_process_method is 'Amplitude':
						plt.plot((res_freq_sweep) / u.MHz, np.abs(Ig + 1j * Qg), ".", label="Ground")
						plt.plot((res_freq_sweep) / u.MHz, np.abs(Ie + 1j * Qe), ".", label="Excited")
						plt.ylabel("Signal Amplitude [V]")
					elif data_process_method is 'I':
						plt.plot((res_freq_sweep) / u.MHz, Ig, ".", label="Ground")
						plt.plot((res_freq_sweep) / u.MHz, Ie, ".", label="Excited")
						plt.ylabel("Signal I Quadrature [V]")
					plt.xlabel("Frequency [MHz]")
					plt.legend(["Ground", "Excited"])
					plt.pause(0.5)


			# fetch all data after live-updating
			timestamp_finished = datetime.datetime.now()
			Ig, Qg, Ie, Qe, _ = results.fetch_all()
			# Convert I & Q to Volts
			Ig = u.demod2volts(Ig, machine.resonators[qubit_index].readout_pulse_length)
			Qg = u.demod2volts(Qg, machine.resonators[qubit_index].readout_pulse_length)
			Ie = u.demod2volts(Ie, machine.resonators[qubit_index].readout_pulse_length)
			Qe = u.demod2volts(Qe, machine.resonators[qubit_index].readout_pulse_length)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
				{
					"I_g": (["x"], Ig),
					"Q_g": (["x"], Qg),
					"I_e": (["x"], Ie),
					"Q_e": (["x"], Qe),
				},
				coords={
					"Readout_Frequency": (["x"], res_freq_sweep),
				},
			)
			
			expt_name = 'res_spec_ge'
			expt_long_name = 'Resonator Spectroscopy ge'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = []  # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n+1):
	with for_(*from_array(df,res_if_sweep)):
		update_frequency(machine.resonators[qubit_index].name, df)
		# measure g					
		measure("readout",
			machine.resonators[qubit_index].name,
			None,
			dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
			dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
		)
		save(I_g, I_g_st)
		save(Q_g, Q_g_st)
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)

		# measure e
		align()  # global align
		play("pi", machine.qubits[qubit_index].name)
		align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
		measure(
			"readout",
			machine.resonators[qubit_index].name,
			None,
			dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
			dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
			)
		save(I_e, I_e_st)
		save(Q_e, Q_e_st)
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
											  expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				plt.title("Resonator spectroscopy")
				for hlp in [r'_g',r'_e']:
					plt.plot((res_freq_sweep) / u.MHz, expt_dataset[data_process_method + hlp].values, '.', label = hlp)
					plt.xlabel("Frequency [MHz]")
					plt.ylabel("Signal" + data_process_method)
					plt.legend([r"_g", r"_e"])

			return machine, expt_dataset


	def rr_switch_delay(self, machine, rr_switch_delay_sweep, qubit_index, n_avg = 10E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False):
		"""
		1D experiment to calibrate switch delay for the resonator.

		Args:
			machine: 
			rr_switch_delay_sweep (): in ns
			qubit_index ():
			n_avg ():
			cd_time ():
			to_simulate ():
			simulation_len ():
			final_plot ():
			
		Returns:
			machine
			rr_switch_delay_sweep
			sig_amp
		"""
		

		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		res_if = machine.resonators[qubit_index].f_readout - res_lo

		if abs(res_if) > 400E6: # check if parameters are within hardware limit
			print("res if > 400MHz")
			return machine, None

		with program() as rr_switch_delay_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()

			with for_(n, 0, n < n_avg, n+1):
				update_frequency(machine.resonators[qubit_index].name, res_if)
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
			job = self.qmm.simulate(config, rr_switch_delay_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			start_time = time.time()
			I_tot = []
			Q_tot = []
			if final_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]

			for delay_index, delay_value in enumerate(rr_switch_delay_sweep):
				#machine.resonators[qubit_index].digital_marker.delay = int(delay_value)
				machine = self.set_digital_delay(machine, "resonators", int(delay_value))
				
				config = build_config(machine)
				qm = self.qmm.open_qm(config)
				timestamp_created = datetime.datetime.now()
				job = qm.execute(rr_switch_delay_prog)
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
				progress_counter(delay_index, len(rr_switch_delay_sweep), start_time=start_time)

			I_tot = np.array(I_tot)
			Q_tot = np.array(Q_tot)
			sigs_qubit = I_tot + 1j * Q_tot
			sig_amp = np.abs(sigs_qubit)  # Amplitude
			sig_phase = np.angle(sigs_qubit)  # Phase

			if final_plot:
				plt.cla()
				plt.title("res. switch delay")
				plt.plot(rr_switch_delay_sweep, sig_amp, ".")
				plt.xlabel("switch delay [ns]")
				plt.ylabel("Signal [V]")

			# save data
			exp_name = 'res_switch_delay'
			qubit_name = 'Q' + str(qubit_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"res_delay": rr_switch_delay_sweep, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset


	def rr_switch_buffer(self, machine, rr_switch_buffer_sweep, qubit_index, n_avg = 10E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False):
		"""
		1D experiment to calibrate switch delay for the resonator.

		Args:
			machine: 
			rr_switch_buffer_sweep (): in ns, this will be added to both sides of the switch (x2), to account for the rise and fall
			qubit_index ():
			n_avg ():
			cd_time ():
			to_simulate ():
			simulation_len ():
			final_plot ():
			
		Returns:
			machine:
			rr_switch_buffer_sweep:
			sig_amp:
		"""
		

		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		res_if = machine.resonators[qubit_index].f_readout - res_lo

		if abs(res_if) > 400E6: # check if parameters are within hardware limit
			print("res if > 400MHz")
			return machine, None

		with program() as rr_switch_buffer_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()

			with for_(n, 0, n < n_avg, n+1):
				update_frequency(machine.resonators[qubit_index].name, res_if)
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
			job = self.qmm.simulate(config, rr_switch_buffer_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			start_time = time.time()
			I_tot = []
			Q_tot = []
			if final_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]

			for buffer_index, buffer_value in enumerate(rr_switch_buffer_sweep):
				machine = self.set_digital_buffer(machine, "resonators", int(buffer_value))
				config = build_config(machine)
				qm = self.qmm.open_qm(config)
				timestamp_created = datetime.datetime.now()
				job = qm.execute(rr_switch_buffer_prog)
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
				progress_counter(buffer_index, len(rr_switch_buffer_sweep), start_time=start_time)

			I_tot = np.array(I_tot)
			Q_tot = np.array(Q_tot)
			sigs_qubit = I_tot + 1j * Q_tot
			sig_amp = np.abs(sigs_qubit)  # Amplitude
			sig_phase = np.angle(sigs_qubit)  # Phase

			if final_plot:
				plt.cla()
				plt.title("res. switch buffer")
				plt.plot(rr_switch_buffer_sweep, sig_amp, ".")
				plt.xlabel("switch buffer [ns]")
				plt.ylabel("Signal [V]")

			# save data
			exp_name = 'res_switch_buffer'
			qubit_name = 'Q' + str(qubit_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"res_buffer": rr_switch_buffer_sweep, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset


	def single_shot_IQ_blob(self, machine, qubit_index, res_freq = None, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False):
		"""Runs single-shot readout experiment to get the IQ blobs
		
		Measures ground state I, Q, then excited state I, Q. Data saved as Ig, Qg, Ie, Qe in expt_dataset.
		Demodulation with machine.resonators[qubit_index].rotation_angle.
		res_freq is optional--if not provided, will use f_readout.
		
		Args:
			machine ([type]): [description]
			qubit_index ([type]): [description]
			res_freq ([type]): [description] (default: `None`)
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `20E3`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
		
		Returns:
			machine
			expt_dataset
		"""

		if res_freq is None:
			res_freq = machine.resonators[qubit_index].f_readout

		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		res_if = np.floor(res_freq - res_lo)

		if abs(res_if) > 400E6: # check if parameters are within hardware limit
			print("res if > 400MHz")
			return machine, None

		with program() as rr_IQ_prog:
			n = declare(int)
			I_g = declare(fixed)
			Q_g = declare(fixed)
			I_g_st = declare_stream()
			Q_g_st = declare_stream()
			I_e = declare(fixed)
			Q_e = declare(fixed)
			I_e_st = declare_stream()
			Q_e_st = declare_stream()
			n_st = declare_stream()

			with for_(n, 0, n < n_avg, n+1):
				measure(
					"readout",
					machine.resonators[qubit_index].name,
					None,
					dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
					dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
				)
				save(I_g, I_g_st)
				save(Q_g, Q_g_st)
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)

				align()  # global align

				play("pi", machine.qubits[qubit_index].name)
				align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
				measure(
					"readout",
					machine.resonators[qubit_index].name,
					None,
					dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
					dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
				)

				save(I_e, I_e_st)
				save(Q_e, Q_e_st)
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				I_g_st.save_all("I_g")
				Q_g_st.save_all("Q_g")
				I_e_st.save_all("I_e")
				Q_e_st.save_all("Q_e")
				n_st.save_all("iteration")


		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		# Simulate or execute #
		if to_simulate: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, rr_IQ_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(rr_IQ_prog)
			# Get progress counter to monitor runtime of the program
			results = fetching_tool(job, data_list=["I_g", "Q_g", "I_e", "Q_e", "iteration"], mode="live")
			# Live plotting
			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
				ax = plt.gca()

			while results.is_processing():
				# Fetch results
				Ig, Qg, Ie, Qe, iteration = results.fetch_all()
				# Convert I & Q to Volts
				Ig = u.demod2volts(Ig, machine.resonators[qubit_index].readout_pulse_length)
				Qg = u.demod2volts(Qg, machine.resonators[qubit_index].readout_pulse_length)
				Ie = u.demod2volts(Ie, machine.resonators[qubit_index].readout_pulse_length)
				Qe = u.demod2volts(Qe, machine.resonators[qubit_index].readout_pulse_length)

				# Progress bar
				progress_counter(iteration[0], n_avg, start_time=results.get_start_time())

				if live_plot:
					plt.cla()
					plt.plot(Ig, Qg, ".", alpha=0.1, label="Ground", markersize=3)
					plt.plot(Ie, Qe, ".", alpha=0.1, label="Excited", markersize=3)
					ax.set_aspect("equal", "box")
					plt.legend(["Ground", "Excited"])
					plt.xlabel("I")
					plt.ylabel("Q")
					plt.title("Original Data")
					ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5,
							  markerscale=5)
					plt.show()
					plt.pause(0.5)


			# Fetch all data
			timestamp_finished = datetime.datetime.now()
			Ig, Qg, Ie, Qe, _ = results.fetch_all()
			# Convert I & Q to Volts
			Ig = u.demod2volts(Ig, machine.resonators[qubit_index].readout_pulse_length)
			Qg = u.demod2volts(Qg, machine.resonators[qubit_index].readout_pulse_length)
			Ie = u.demod2volts(Ie, machine.resonators[qubit_index].readout_pulse_length)
			Qe = u.demod2volts(Qe, machine.resonators[qubit_index].readout_pulse_length)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
				{
					"I_g": (["x"], Ig),
					"Q_g": (["x"], Qg),
					"I_e": (["x"], Ie),
					"Q_e": (["x"], Qe),
				},
			)
			
			expt_name = 'single_shot_IQ'
			expt_long_name = 'Single Shot Readout IQ'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = []  # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n+1):
	measure(
		"readout",
		machine.resonators[qubit_index].name,
		None,
		dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
		dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
	)
	save(I_g, I_g_st)
	save(Q_g, Q_g_st)
	wait(cd_time * u.ns, machine.resonators[qubit_index].name)

	align()  # global align

	play("pi", machine.qubits[qubit_index].name)
	align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
	measure(
		"readout",
		machine.resonators[qubit_index].name,
		None,
		dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
		dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
	)

	save(I_e, I_e_st)
	save(Q_e, Q_e_st)
	wait(cd_time * u.ns, machine.resonators[qubit_index].name)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
											  expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				ax = plt.gca()
				plt.plot(Ig, Qg, ".", alpha=0.1, label="Ground", markersize=3)
				plt.plot(Ie, Qe, ".", alpha=0.1, label="Excited", markersize=3)
				ax.set_aspect("equal","box")
				plt.legend(["Ground", "Excited"])
				plt.xlabel("I")
				plt.ylabel("Q")
				plt.title("Original Data")
				ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5,
						   markerscale=5)
				plt.show()

			return machine, expt_dataset


	def single_shot_freq_optimization(self, machine, res_freq_sweep, qubit_index, n_avg = 10E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, live_plot = False, data_process_method = 'SNR'):
		"""Run frequency optimization for single-shot readout.
		
		This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
		(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
		|e> state). This is done while varying the readout frequency.
		The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to
		determine the Signal-to-Noise Ratio (SNR). The readout frequency that yields the highest SNR is selected as the
		optimal choice.
		
		Args:
			machine ([type]): [description]
			res_freq_sweep ([type]): [description]
			qubit_index ([type]): [description]
			n_avg (number): [description] (default: `10E3`)
			cd_time (number): [description] (default: `20E3`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key to be plotted. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `SNR`)
		
		Returns:
			[type]: [description]
		"""

		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		res_if_sweep = res_freq_sweep - res_lo
		res_if_sweep = np.floor(res_if_sweep)

		if np.max(abs(res_if_sweep)) > 400E6:  # check if parameters are within hardware limit
			print("res if range > 400MHz")
			return machine, None

		with program() as ro_freq_opt:
			n = declare(int)  # QUA variable for the averaging loop
			df = declare(int)  # QUA variable for the readout IF frequency
			I_g = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |g>
			Q_g = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |g>
			Ig_st = declare_stream()
			Qg_st = declare_stream()
			I_e = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |e>
			Q_e = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |e>
			Ie_st = declare_stream()
			Qe_st = declare_stream()
			n_st = declare_stream()

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(df, res_if_sweep)):
					# Update the frequency of the digital oscillator linked to the resonator element
					update_frequency(machine.resonators[qubit_index].name, df)
					# Measure the state of the resonator
					measure(
						"readout",
						machine.resonators[qubit_index].name,
						None,
						dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
						dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
					)
					# Wait for the qubit to decay to the ground state
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					# Save the 'I_e' & 'Q_e' quadratures to their respective streams
					save(I_g, Ig_st)
					save(Q_g, Qg_st)

					align()  # global align
					# Play the x180 gate to put the qubit in the excited state
					play("pi", machine.qubits[qubit_index].name)
					# Align the two elements to measure after playing the qubit pulse.
					align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
					# Measure the state of the resonator
					measure(
						"readout",
						machine.resonators[qubit_index].name,
						None,
						dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
						dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
					)
					# Wait for the qubit to decay to the ground state
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					# Save the 'I_e' & 'Q_e' quadratures to their respective streams
					save(I_e, Ie_st)
					save(Q_e, Qe_st)
				# Save the averaging iteration to get the progress bar
				save(n, n_st)

			with stream_processing():
				n_st.save("iteration")
				# mean values
				Ig_st.buffer(len(res_if_sweep)).average().save("Ig_avg")
				Qg_st.buffer(len(res_if_sweep)).average().save("Qg_avg")
				Ie_st.buffer(len(res_if_sweep)).average().save("Ie_avg")
				Qe_st.buffer(len(res_if_sweep)).average().save("Qe_avg")
				# variances to get the SNR
				(
						((Ig_st.buffer(len(res_if_sweep)) * Ig_st.buffer(len(res_if_sweep))).average())
						- (Ig_st.buffer(len(res_if_sweep)).average() * Ig_st.buffer(len(res_if_sweep)).average())
				).save("Ig_var")
				(
						((Qg_st.buffer(len(res_if_sweep)) * Qg_st.buffer(len(res_if_sweep))).average())
						- (Qg_st.buffer(len(res_if_sweep)).average() * Qg_st.buffer(len(res_if_sweep)).average())
				).save("Qg_var")
				(
						((Ie_st.buffer(len(res_if_sweep)) * Ie_st.buffer(len(res_if_sweep))).average())
						- (Ie_st.buffer(len(res_if_sweep)).average() * Ie_st.buffer(len(res_if_sweep)).average())
				).save("Ie_var")
				(
						((Qe_st.buffer(len(res_if_sweep)) * Qe_st.buffer(len(res_if_sweep))).average())
						- (Qe_st.buffer(len(res_if_sweep)).average() * Qe_st.buffer(len(res_if_sweep)).average())
				).save("Qe_var")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		# Simulate or execute #
		if to_simulate:  # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, ro_freq_opt, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			# Open the quantum machine
			qm = self.qmm.open_qm(config)
			# Send the QUA program to the OPX, which compiles and executes it
			timestamp_created = datetime.datetime.now()
			job = qm.execute(ro_freq_opt)  # execute QUA program
			# Get results from QUA program
			results = fetching_tool(
				job,
				data_list=["Ig_avg", "Qg_avg", "Ie_avg", "Qe_avg", "Ig_var", "Qg_var", "Ie_var", "Qe_var", "iteration"],
				mode="live",
			)

			if live_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				Ig_avg, Qg_avg, Ie_avg, Qe_avg, Ig_var, Qg_var, Ie_var, Qe_var, iteration = results.fetch_all()
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				# Derive the SNR
				Z = (Ie_avg - Ig_avg) + 1j * (Qe_avg - Qg_avg)
				var = (Ig_var + Qg_var + Ie_var + Qe_var) / 4
				SNR = ((np.abs(Z)) ** 2) / (2 * var)
				
				# Plot results
				if live_plot:
					if data_process_method is not 'SNR':
						print(r'data_process_method is not SNR. Abort...')
						return machine, None

					plt.cla()
					plt.plot(res_freq_sweep, SNR, ".")
					plt.title("Readout Frequency Optimization")
					plt.xlabel("Readout Frequency [Hz]")
					plt.ylabel("SNR")
					#plt.grid("on")
					plt.pause(0.5)

			# Fetch all data
			timestamp_finished = datetime.datetime.now()
			Ig_avg, Qg_avg, Ie_avg, Qe_avg, Ig_var, Qg_var, Ie_var, Qe_var, _ = results.fetch_all()
			
			# No conversion to Volt!! Not sure how to do that for variance

			# Derive the SNR
			Z = (Ie_avg - Ig_avg) + 1j * (Qe_avg - Qg_avg)
			var = (Ig_var + Qg_var + Ie_var + Qe_var) / 4
			SNR = ((np.abs(Z)) ** 2) / (2 * var)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
				{
					"I_g_avg": (["x"], Ig_avg),
					"Q_g_avg": (["x"], Qg_avg),
					"I_e_avg": (["x"], Ie_avg),
					"Q_e_avg": (["x"], Qe_avg),
					"I_g_var": (["x"], Ig_var),
					"Q_g_var": (["x"], Qg_var),
					"I_e_var": (["x"], Ie_var),
					"Q_e_var": (["x"], Qe_var),
					"SNR": (["x"], SNR),
				},
				coords={
					"Readout_Frequency": (["x"], res_freq_sweep),
				},
			)
			
			expt_name = 'single_shot_freq'
			expt_long_name = 'Readout Frequency Optimization'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = []  # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(df, res_if_sweep)):
		# Update the frequency of the digital oscillator linked to the resonator element
		update_frequency(machine.resonators[qubit_index].name, df)
		# Measure the state of the resonator
		measure(
			"readout",
			machine.resonators[qubit_index].name,
			None,
			dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
			dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
		)
		# Wait for the qubit to decay to the ground state
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
		# Save the 'I_e' & 'Q_e' quadratures to their respective streams
		save(I_g, Ig_st)
		save(Q_g, Qg_st)

		align()  # global align
		# Play the x180 gate to put the qubit in the excited state
		play("pi", machine.qubits[qubit_index].name)
		# Align the two elements to measure after playing the qubit pulse.
		align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
		# Measure the state of the resonator
		measure(
			"readout",
			machine.resonators[qubit_index].name,
			None,
			dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
			dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
		)
		# Wait for the qubit to decay to the ground state
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
		# Save the 'I_e' & 'Q_e' quadratures to their respective streams
		save(I_e, Ie_st)
		save(Q_e, Qe_st)
	# Save the averaging iteration to get the progress bar
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
											  expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				if live_plot is False:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
				plt.cla()
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker='.')
				plt.ylabel("SNR")

			print(f"The optimal readout frequency is {res_freq_sweep[np.argmax(SNR)]} Hz (SNR={max(SNR)})")

			return machine, expt_dataset


	def single_shot_amp_optimization(self, machine, res_amp_rel_sweep, qubit_index, n_avg = 10E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, data_process_method = 'Fidelity'):
		"""Run amplitude optimization for single-shot readout.
		
		The sequence consists in measuring the state of the resonator after thermalization (qubit in |g>) and after
		playing a pi pulse to the qubit (qubit in |e>) successively while sweeping the readout amplitude.
		The 'I' & 'Q' quadratures when the qubit is in |g> and |e> are extracted to derive the readout fidelity.
		The optimal readout amplitude is chosen as to maximize the readout fidelity.
		
		Args:
			machine ([type]): [description]
			res_amp_rel_sweep ([type]): [description]
			qubit_index ([type]): [description]
			n_avg (number): [description] (default: `10E3`)
			cd_time (number): [description] (default: `20E3`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			data_process_method (str): variable name/key to be plotted. e.g. Amplitude, Phase, SNR, I, Q, etc (default: `Fidelity`)
		
		Returns:
			[type]: [description]
		"""

		if max(abs(res_amp_rel_sweep)) > 2.0:
			print("some rel amps > 2.0, removed from experiment run")
			res_amp_rel_sweep = res_amp_rel_sweep[abs(res_amp_rel_sweep) < 2.0]

		readout_amp = machine.resonators[qubit_index].readout_pulse_amp
		res_amp_abs_sweep = readout_amp * res_amp_rel_sweep
		if max(abs(res_amp_abs_sweep)) > 0.5:
			print("some abs amps > 0.5, removed from experiment run")
			res_amp_rel_sweep = res_amp_rel_sweep[abs(res_amp_abs_sweep) < 0.5]
			res_amp_abs_sweep = readout_amp * res_amp_rel_sweep

		with program() as ro_amp_opt:
			n = declare(int)  # QUA variable for the number of runs
			counter = declare(int, value=0)  # Counter for the progress bar
			a = declare(fixed)  # QUA variable for the readout amplitude
			I_g = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |g>
			Q_g = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |g>
			Ig_st = declare_stream()
			Qg_st = declare_stream()
			I_e = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |e>
			Q_e = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |e>
			Ie_st = declare_stream()
			Qe_st = declare_stream()
			n_st = declare_stream()

			with for_(*from_array(a, res_amp_rel_sweep)):
				save(counter, n_st)
				with for_(n, 0, n < n_avg, n + 1):
					# Measure the state of the resonator
					measure(
						"readout" * amp(a),
						machine.resonators[qubit_index].name,
						None,
						dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
						dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
					)
					# Wait for the qubit to decay to the ground state
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					# Save the 'I_e' & 'Q_e' quadratures to their respective streams
					save(I_g, Ig_st)
					save(Q_g, Qg_st)

					align()  # global align
					# Play the x180 gate to put the qubit in the excited state
					play("pi", machine.qubits[qubit_index].name)
					# Align the two elements to measure after playing the qubit pulse.
					align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
					# Measure the state of the resonator
					measure(
						"readout" * amp(a),
						machine.resonators[qubit_index].name,
						None,
						dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
						dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
					)
					# Wait for the qubit to decay to the ground state
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					# Save the 'I_e' & 'Q_e' quadratures to their respective streams
					save(I_e, Ie_st)
					save(Q_e, Qe_st)
				# Save the counter to get the progress bar
				assign(counter, counter + 1)

			with stream_processing():
				Ig_st.buffer(n_avg).buffer(len(res_amp_rel_sweep)).save("I_g")
				Qg_st.buffer(n_avg).buffer(len(res_amp_rel_sweep)).save("Q_g")
				Ie_st.buffer(n_avg).buffer(len(res_amp_rel_sweep)).save("I_e")
				Qe_st.buffer(n_avg).buffer(len(res_amp_rel_sweep)).save("Q_e")
				n_st.save("iteration")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		# Simulate or execute #
		if to_simulate:  # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, ro_amp_opt, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			# Open the quantum machine
			qm = self.qmm.open_qm(config)
			# Send the QUA program to the OPX, which compiles and executes it
			timestamp_created = datetime.datetime.now()
			job = qm.execute(ro_amp_opt)  # execute QUA program
			# Get results from QUA program
			results = fetching_tool(job, data_list=["iteration"], mode="live")
			# Get progress counter to monitor runtime of the program
			while results.is_processing():
				# Fetch results
				iteration = results.fetch_all()
				# Progress bar
				progress_counter(iteration[0], len(res_amp_rel_sweep), start_time=results.get_start_time())

			# Fetch the results at the end
			timestamp_finished = datetime.datetime.now()
			results = fetching_tool(job, data_list=["I_g", "Q_g", "I_e", "Q_e"])
			Ig, Qg, Ie, Qe = results.fetch_all()
			
			# Convert I & Q to Volts
			Ig = u.demod2volts(Ig, machine.resonators[qubit_index].readout_pulse_length)
			Qg = u.demod2volts(Qg, machine.resonators[qubit_index].readout_pulse_length)
			Ie = u.demod2volts(Ie, machine.resonators[qubit_index].readout_pulse_length)
			Qe = u.demod2volts(Qe, machine.resonators[qubit_index].readout_pulse_length)

			# Process the data
			fidelity_vec = []
			for i in range(len(res_amp_rel_sweep)):
				angle, threshold, fidelity, _, _, _, _ = self._two_state_discriminator(Ig[i], Qg[i], Ie[i], Qe[i], final_plot = False, to_print = False)
				fidelity_vec.append(fidelity)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
				{
					"I_g": (["x", "n"], Ig), # this dimension might be wrong...but maybe it doesn't matter in xarray?
					"Q_g": (["x", "n"], Qg),
					"I_e": (["x", "n"], Ie),
					"Q_e": (["x", "n"], Qe),
					"Fidelity": (["x"], fidelity_vec),
				},
				coords={
					"Readout_Amplitude": (["x"], res_amp_abs_sweep),
				},
			)
			
			expt_name = 'single_shot_amp'
			expt_long_name = 'Readout Amplitude Optimization'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = []  # use t0, t1, t2, ...
			expt_sequence = """with for_(*from_array(a, res_amp_rel_sweep)):
	save(counter, n_st)
	with for_(n, 0, n < n_avg, n + 1):
		# Measure the state of the resonator
		measure(
			"readout" * amp(a),
			machine.resonators[qubit_index].name,
			None,
			dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
			dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
		)
		# Wait for the qubit to decay to the ground state
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
		# Save the 'I_e' & 'Q_e' quadratures to their respective streams
		save(I_g, Ig_st)
		save(Q_g, Qg_st)

		align()  # global align
		# Play the x180 gate to put the qubit in the excited state
		play("pi", machine.qubits[qubit_index].name)
		# Align the two elements to measure after playing the qubit pulse.
		align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
		# Measure the state of the resonator
		measure(
			"readout" * amp(a),
			machine.resonators[qubit_index].name,
			None,
			dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
			dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
		)
		# Wait for the qubit to decay to the ground state
		wait(cd_time * u.ns, machine.resonators[qubit_index].name)
		# Save the 'I_e' & 'Q_e' quadratures to their respective streams
		save(I_e, Ie_st)
		save(Q_e, Qe_st)
	# Save the counter to get the progress bar
	assign(counter, counter + 1)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
											  expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			# Plot the data
			if final_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker='.')
				plt.ylabel("Readout Fidelity [%]")

			res_amp_opt = res_amp_abs_sweep[np.argmax(fidelity_vec)]
			print(
				f"The optimal readout amplitude is {res_amp_opt / u.mV:.3f} mV (Fidelity={max(fidelity_vec):.1f}%)"
			)

			return machine, expt_dataset


	def single_shot_duration_optimization(self, machine, division_length, qubit_index, readout_len = 1E3, ringdown_len = 0, n_avg = 10E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, final_plot = True, data_process_method = 'Fidelity'):
		"""
				READOUT OPTIMISATION: DURATION
		This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
		(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
		|e> state). The "demod.accumulated" method is employed to assess the state of the resonator over varying durations.
		Reference: (https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Guides/features/?h=accumulated#accumulated-demodulation)
		The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to determine
		the Signal-to-Noise Ratio (SNR). The readout duration that offers the highest SNR is identified as the optimal choice.
		Note: To observe the resonator's behavior during ringdown, the integration weights length should exceed the readout_pulse length.
		
		:param machine:
		:param readout_len: Readout pulse duration, something much longer than what I typically use, like 2us. In ns
		:param ringdown_len: integration time after readout pulse to observe the ringdown of the resonator, could be 0. In ns
		:param division_length : Size of each demodulation slice in ns
		:param qubit_index:
		:param n_avg:
		:param cd_time:
		:param to_simulate:
		:param simulation_len:
		:param final_plot:
		:return:
			machine
			x_plot: in ns, different readout duration
			SNR
			opt_readout_length
		"""

		# regulate to multiples of clock cycles
		if division_length%4 > 0:
			print(r"division_length not multiples of 4ns (clock cycle). Abort...")
			return machine, None

		division_length = int(division_length/4) # convert to clock cycles

		def update_readout_length(new_readout_length, ringdown_length):
			config["pulses"][f"readout_pulse_q{qubit_index}"]["length"] = new_readout_length
			config["integration_weights"][f"cosine_weights{qubit_index}"] = {
				"cosine": [(1.0, new_readout_length + ringdown_length)],
				"sine": [(0.0, new_readout_length + ringdown_length)],
			}
			config["integration_weights"][f"sine_weights{qubit_index}"] = {
				"cosine": [(0.0, new_readout_length + ringdown_length)],
				"sine": [(1.0, new_readout_length + ringdown_length)],
			}
			config["integration_weights"][f"minus_sine_weights{qubit_index}"] = {
				"cosine": [(0.0, new_readout_length + ringdown_length)],
				"sine": [(-1.0, new_readout_length + ringdown_length)],
			}

		###################
		# The QUA program #
		###################
		# Set maximum readout duration for this scan and update the configuration accordingly
		config = build_config(machine)
		update_readout_length(readout_len * u.ns, ringdown_len * u.ns)
		# Set the accumulated demod parameters
		number_of_divisions = int((readout_len + ringdown_len) / (4 * division_length))
		print("Integration weights chunk-size length in ns:", division_length * 4)
		print("The readout has been sliced in the following number of divisions", number_of_divisions)

		# Time axis for the plots at the end
		readout_len_sweep = np.arange(division_length * 4, readout_len + ringdown_len + 1, division_length * 4)

		with program() as ro_duration_opt:
			n = declare(int)
			II = declare(fixed, size=number_of_divisions)
			IQ = declare(fixed, size=number_of_divisions)
			QI = declare(fixed, size=number_of_divisions)
			QQ = declare(fixed, size=number_of_divisions)
			I = declare(fixed, size=number_of_divisions)
			Q = declare(fixed, size=number_of_divisions)
			ind = declare(int)

			n_st = declare_stream()
			Ig_st = declare_stream()
			Qg_st = declare_stream()
			Ie_st = declare_stream()
			Qe_st = declare_stream()

			with for_(n, 0, n < n_avg, n + 1):
				# Measure the ground state.
				# With demod.accumulated, the results are QUA vectors with 1 point for each accumulated chunk
				measure(
					"readout",
					machine.resonators[qubit_index].name,
					None,
					demod.accumulated("cos", II, division_length, "out1"),
					demod.accumulated("sin", IQ, division_length, "out2"),
					demod.accumulated("minus_sin", QI, division_length, "out1"),
					demod.accumulated("cos", QQ, division_length, "out2"),
				)
				# Save the QUA vectors to their corresponding streams
				with for_(ind, 0, ind < number_of_divisions, ind + 1):
					assign(I[ind], II[ind] + IQ[ind])
					save(I[ind], Ig_st)
					assign(Q[ind], QQ[ind] + QI[ind])
					save(Q[ind], Qg_st)
				# Wait for the qubit to decay to the ground state
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)

				align()

				# Measure the excited state.
				# With demod.accumulated, the results are QUA vectors with 1 point for each accumulated chunk
				play("pi", machine.qubits[qubit_index].name)
				align()
				measure(
					"readout",
					machine.resonators[qubit_index].name,
					None,
					demod.accumulated("cos", II, division_length, "out1"),
					demod.accumulated("sin", IQ, division_length, "out2"),
					demod.accumulated("minus_sin", QI, division_length, "out1"),
					demod.accumulated("cos", QQ, division_length, "out2"),
				)
				# Save the QUA vectors to their corresponding streams
				with for_(ind, 0, ind < number_of_divisions, ind + 1):
					assign(I[ind], II[ind] + IQ[ind])
					save(I[ind], Ie_st)
					assign(Q[ind], QQ[ind] + QI[ind])
					save(Q[ind], Qe_st)

				# Wait for the qubit to decay to the ground state
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				# Save the averaging iteration to get the progress bar
				save(n, n_st)

			with stream_processing():
				n_st.save("iteration")
				# mean values
				Ig_st.buffer(number_of_divisions).buffer(n_avg).save("I_g")
				Qg_st.buffer(number_of_divisions).buffer(n_avg).save("Q_g")
				Ie_st.buffer(number_of_divisions).buffer(n_avg).save("I_e")
				Qe_st.buffer(number_of_divisions).buffer(n_avg).save("Q_e")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		# Simulate or execute #
		if to_simulate:  # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, ro_duration_opt, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			# Open the quantum machine
			qm = self.qmm.open_qm(config)
			# Send the QUA program to the OPX, which compiles and executes it
			timestamp_created = datetime.datetime.now()
			job = qm.execute(ro_duration_opt)  # execute QUA program

			# Get progress counter to monitor runtime of the program
			results = fetching_tool(job, data_list=["iteration"], mode="live")
			while results.is_processing():
				# Fetch results
				iteration = results.fetch_all()
				# Progress bar
				progress_counter(iteration[0], n_avg, start_time=results.get_start_time())

			# Fetch the results at the end
			timestamp_finished = datetime.datetime.now()
			results = fetching_tool(job, data_list=["I_g", "Q_g", "I_e", "Q_e"])
			Ig, Qg, Ie, Qe = results.fetch_all()
			
			# Convert I & Q to Volts
			Ig = u.demod2volts(Ig, machine.resonators[qubit_index].readout_pulse_length)
			Qg = u.demod2volts(Qg, machine.resonators[qubit_index].readout_pulse_length)
			Ie = u.demod2volts(Ie, machine.resonators[qubit_index].readout_pulse_length)
			Qe = u.demod2volts(Qe, machine.resonators[qubit_index].readout_pulse_length)

			# Process the data
			fidelity_vec = []
			for i in range(len(readout_len_sweep)):
				angle, threshold, fidelity, _, _, _, _ = self._two_state_discriminator(Ig[:,i], Qg[:,i], Ie[:,i], Qe[:,i], final_plot = False, to_print = False)
				fidelity_vec.append(fidelity)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
				{
					"I_g": (["n", "x"], Ig), # this dimension might be wrong...but maybe it doesn't matter in xarray?
					"Q_g": (["n", "x"], Qg),
					"I_e": (["n", "x"], Ie),
					"Q_e": (["n", "x"], Qe),
					"Fidelity": (["x"], fidelity_vec),
				},
				coords={
					"Readout_Length": (["x"], readout_len_sweep),
				},
			)
			
			expt_name = 'single_shot_len'
			expt_long_name = 'Readout Length Optimization'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = []  # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	# Measure the ground state.
	# With demod.accumulated, the results are QUA vectors with 1 point for each accumulated chunk
	measure(
		"readout",
		machine.resonators[qubit_index].name,
		None,
		demod.accumulated("cos", II, division_length, "out1"),
		demod.accumulated("sin", IQ, division_length, "out2"),
		demod.accumulated("minus_sin", QI, division_length, "out1"),
		demod.accumulated("cos", QQ, division_length, "out2"),
	)
	# Save the QUA vectors to their corresponding streams
	with for_(ind, 0, ind < number_of_divisions, ind + 1):
		assign(I[ind], II[ind] + IQ[ind])
		save(I[ind], Ig_st)
		assign(Q[ind], QQ[ind] + QI[ind])
		save(Q[ind], Qg_st)
	# Wait for the qubit to decay to the ground state
	wait(cd_time * u.ns, machine.resonators[qubit_index].name)

	align()

	# Measure the excited state.
	# With demod.accumulated, the results are QUA vectors with 1 point for each accumulated chunk
	play("pi", machine.qubits[qubit_index].name)
	align()
	measure(
		"readout",
		machine.resonators[qubit_index].name,
		None,
		demod.accumulated("cos", II, division_length, "out1"),
		demod.accumulated("sin", IQ, division_length, "out2"),
		demod.accumulated("minus_sin", QI, division_length, "out1"),
		demod.accumulated("cos", QQ, division_length, "out2"),
	)
	# Save the QUA vectors to their corresponding streams
	with for_(ind, 0, ind < number_of_divisions, ind + 1):
		assign(I[ind], II[ind] + IQ[ind])
		save(I[ind], Ie_st)
		assign(Q[ind], QQ[ind] + QI[ind])
		save(Q[ind], Qe_st)

	# Wait for the qubit to decay to the ground state
	wait(cd_time * u.ns, machine.resonators[qubit_index].name)
	# Save the averaging iteration to get the progress bar
	save(n, n_st)"""

			# save data
			expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
											  expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			# Plot the data
			if final_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0], marker='.')
				plt.ylabel("Readout Fidelity [%]")
				plt.ylabel("Readout Fidelity [%]")

			res_len_opt = readout_len_sweep[np.argmax(fidelity_vec)]
			print(
				f"The optimal readout duration is {res_len_opt:.0f} ns (Fidelity={max(fidelity_vec):.1f}%)"
			)

			return machine, expt_dataset

	# these are functions required for the single shot optimization
	def _two_state_discriminator(self, Ig, Qg, Ie, Qe, final_plot = True, live_plot = False, to_print = True):
		"""
		Given two blobs in the IQ plane representing two states, finds the optimal threshold to discriminate between them
		and calculates the fidelity. Also returns the angle in which the data needs to be rotated in order to have all the
		information in the `I` (`X`) axis.

		.. note::
			This function assumes that there are only two blobs in the IQ plane representing two states (ground and excited)
			Unexpected output will be returned in other cases.


		:param float Ig: A vector containing the `I` quadrature of data points in the ground state
		:param float Qg: A vector containing the `Q` quadrature of data points in the ground state
		:param float Ie: A vector containing the `I` quadrature of data points in the excited state
		:param float Qe: A vector containing the `Q` quadrature of data points in the excited state
		:param bool final_plot: When true (default), plot the results
		:param bool to_print: When true (default), print the results
		:returns: A tuple of (angle, threshold, fidelity, gg, ge, eg, ee).
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
			self._false_detections,0.5 * (np.mean(Ig_rotated) + np.mean(Ie_rotated)),(Ig_rotated, Ie_rotated),method="Nelder-Mead",)
		threshold = fit.x[0]

		gg = np.sum(Ig_rotated < threshold) / len(Ig_rotated)
		ge = np.sum(Ig_rotated > threshold) / len(Ig_rotated)
		eg = np.sum(Ie_rotated < threshold) / len(Ie_rotated)
		ee = np.sum(Ie_rotated > threshold) / len(Ie_rotated)

		fidelity = 100 * (gg + ee) / 2

		if to_print == True:
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

		if final_plot:
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
		return angle, threshold, fidelity, gg, ge, eg, ee


	def _false_detections(self, threshold, Ig, Ie):
		"""Auxiliary function for self._two_state_discriminator.
		
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
