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
#from qutip import *
from typing import Union
from macros import ham, readout_rotated_macro, declare_vars, wait_until_job_is_paused
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime


class EH_1D:
	"""
	class for some 1D experiments used for 2D scans
	"""
	def __init__(self, ref_to_qmm):
		self.qmm = ref_to_qmm

	def res_freq(self, machine, res_freq_sweep, qubit_index, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, fig = None):
		"""Run resonator spectroscopy (1D)
		
		This experiment is designed to work with `self.res_freq_analysis` to find the resonance frequency by localizing the minima in pulsed transmission signal.
		Result is not automatically saved.
		This function (instead of the standard 1D resonator spectroscopy `rr_freq`) to avoid constructing the xarray dataset, which will be not so straightforward to assemble them further. But this could be done in the future.
		
		Args:
			machine ([type]): [description]
			res_freq_sweep ([type]): [description]
			qubit_index ([type]): [description]
			n_avg ([type]): [description]
			cd_time ([type]): [description]
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `3000`)
			fig ([type]): Fig reference, mainly to have the ability to interupt the experiment. (default: `None`)
		
		Returns:
			machine
			I
			Q
		"""


		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		if res_lo < 2E9:
			print("LO < 2GHz, abort")
			return machine, None, None
		res_if_sweep = res_freq_sweep - res_lo
		res_if_sweep = np.floor(res_if_sweep)

		if np.max(abs(res_if_sweep)) > 400E6: # check if parameters are within hardware limit
			print("res if range > 400MHz")
			return machine, None, None

		with program() as rr_freq_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			df = declare(int)

			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(df,res_if_sweep)):
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
			return machine, None, None

		else:
			qm = self.qmm.open_qm(config)
			job = qm.execute(rr_freq_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode = "live") # wait for all (default), rather than live mode

			if fig is not None:
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				plt.pause(0.5)

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)

			return machine, I, Q


	def qubit_freq(self, machine, qubit_freq_sweep, qubit_index, ff_amp = 0.0, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, fig = None):
		"""Run 1D qubit spectroscopy.
		
		Result is not automatically saved.
		This function (instead of the standard 1D qubit spectroscopy `qubit_freq`) to avoid constructing the xarray dataset, which will be not so straightforward to assemble them further. But this could be done in the future.
		
		
		Args:
			machine ([type]): [description]
			qubit_freq_sweep ([type]): [description]
			qubit_index ([type]): [description]
			ff_amp (number): [description] (default: `0.0`)
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `20E3`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `3000`)
			fig ([type]): Fig reference, mainly to have the ability to interupt the experiment. (default: `None`)
		
		Returns:
			machine
			I
			Q
		"""


		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency

		if qubit_lo < 2E9:
			print("LO < 2GHz, abort")
			return machine, None, None

		qubit_if_sweep = qubit_freq_sweep - qubit_lo
		qubit_if_sweep = np.floor(qubit_if_sweep)
		ff_duration = machine.qubits[qubit_index].pi_length + 40

		if np.max(abs(qubit_if_sweep)) > 400E6: # check if parameters are within hardware limit
			print("qubit if range > 400MHz")
			return machine, None, None

		with program() as qubit_freq_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			df = declare(int)

			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(df,qubit_if_sweep)):
					update_frequency(machine.qubits[qubit_index].name, df)
					play("const" * amp(ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
					wait(5, machine.qubits[qubit_index].name)
					play('pi', machine.qubits[qubit_index].name)
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
			return machine, None, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(qubit_freq_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode = "live") # wait for all (default), rather than live mode

			if fig is not None:
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
			
			while results.is_processing():
				plt.pause(0.5)

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)

			return machine, I, Q


	def res_freq_analysis(self, res_freq_sweep, I, Q, data_process_method = 'Amplitude'):
		"""Analysis for the 1D resonator spectroscopy experiment.
		
		Designed to work with `self.res_freq`. Take the data, identify the minimal of data Amplitude, and return the corresponding frequency.
		
		Args:
			res_freq_sweep ([type]): [description]
			I ([type]): [description]
			Q ([type]): [description]
			data_process_method (str): [description] (default: `'Amplitude'`)
		
		Returns:
			res_freq (np.float64??)
		"""


		if data_process_method is 'Phase':
			y = np.unwrap(np.angle(I + 1j * Q))
		elif data_process_method is 'Amplitude':
			y = np.abs(I + 1j * Q)
		elif data_process_method is 'I':
			y = I

		idx = np.argmin(y)  # find minimum

		return res_freq_sweep[idx]
