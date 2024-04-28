"""
This file contains Octave controls
written by Mo Chen in April, 2024
"""
from qm import QuantumMachinesManager
from qm.octave import *
from qm.octave.octave_manager import ClockMode
from qm.octave.octave_mixer_calibration import AutoCalibrationParams
from configuration import *
import json
from quam import QuAM

class EH_Octave:
	"""
	Class for controlling Octave
	Attributes:

	Methods (useful ones):
		update_tPath: reference to Experiment.update_tPath
		update_str_datetime: reference to Experiment.update_str_datetime
	"""
	def __init__(self,ref_to_update_tPath, ref_to_update_str_datetime):
		self.update_tPath = ref_to_update_tPath
		self.update_str_datetime = ref_to_update_str_datetime

	def set_clock(self, external_clock = True):
		"""
		should set the octave clock to external 10MHz. Note this is automatically run at the initialization of ExperimentClass
		:return:
		"""
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		qm = qmm.open_qm(config)

		# Clock settings #
		if external_clock:
			qm.octave.set_clock(octave, clock_mode=ClockMode.External_10MHz)
		#     qm.octave.set_clock(octave, clock_mode=ClockMode.External_100MHz)
		#     qm.octave.set_clock(octave, clock_mode=ClockMode.External_1000MHz)
		else:
			qm.octave.set_clock(octave, clock_mode=ClockMode.Internal)
		
		return

	def calibration(self,machine, qubit_index, res_index, TLS_index = None, log_flag = True, calibration_flag = True, qubit_only = False):
		"""
		calibrates octave, using parameters saved in machine
		:param machine: must provide
		:param qubit_index:
		:param res_index:
		:param TLS_index: None (default). If not None, then use f_tls[TLS_index] for calibration.
		:param log_flag: True (default), have warnings from QM; False: ERROR log only
		:param calibration_flag: True (default). If false, will still update octave configuration, but not run calibration
		:param qubit_only: False (default). only calibrates the qubit (skip resonator calibration).
		
		:return: machine
		"""
		if log_flag:
			qmm = QuantumMachinesManager(host = machine.network.qop_ip, port='9510', octave=octave_config)
		else:
			qmm = QuantumMachinesManager(host = machine.network.qop_ip, port='9510', octave=octave_config, log_level = "ERROR")
		qm = qmm.open_qm(config)

		qubits = [machine.qubits[qubit_index].name]
		resonators = [machine.resonators[res_index].name]

		params = AutoCalibrationParams() # so I can calibrate for the pi pulse amplitude! Default is 125 mV, which is generally too high!
		
		if calibration_flag:
			if TLS_index is None:
				params.if_amplitude = machine.qubits[qubit_index].pi_amp
				if_freq_tmp = machine.qubits[qubit_index].f_01 - machine.octave.LO2.LO_frequency
			else:
				params.if_amplitude = machine.qubits[qubit_index].pi_amp_TLS[TLS_index]
				if_freq_tmp = machine.qubits[qubit_index].f_tls[TLS_index] - machine.octave.LO2.LO_frequency

			print("-" * 37 + " Octave calibration starts...")
			for element in qubits:
		        print("-" * 37 + f" Calibrates {element}")
		        # print(f"------------------------------------- Calibrates q{qubit_index:.0f} for (LO, IF) = ({machine.qubits[qubit_index].lo/1E9:.3f} GHz, {(machine.qubits[qubit_index].f_01 - machine.qubits[qubit_index].lo)/1E6: .3f} MHz)")
		        qm.calibrate_element(element, {machine.octave.LO2.LO_frequency: (if_freq_tmp,)}, params = params)  # can provide many IFs in the tuple for the same LO
		    if qubit_only is not True:
		    	params.if_amplitude = machine.resonators[res_index].readout_pulse_amp
		        for element in resonators:
			        print("-" * 37 + f" Calibrates {element}")
			        # print(f"------------------------------------- Calibrates r{res_index:.0f} for (LO, IF) = ({machine.resonators[res_index].lo/1E9:.3f} GHz, {(machine.resonators[res_index].f_readout - machine.resonators[res_index].lo)/1E6: .3f} MHz)")
					qm.calibrate_element(element, {machine.octave.LO1.LO_frequency: (machine.resonators[res_index].f_readout - machine.octave.LO1.LO_frequency,)}, params = params)  # can provide many IFs in the tuple for the same LO
			print("-" * 37 + " Octave calibration finished.")
		else: # only setting Octave LO, no calibration
			print("-" * 37 + " Setting Octave LO only...")
			qm.octave.set_lo_frequency(machine.qubits[qubit_index].name, machine.octave.LO2.LO_frequency) 
			qm.octave.set_lo_frequency(machine.resonators[res_index].name, machine.octave.LO1.LO_frequency) 
		return machine

	def set_digital_delay(self, machine, element, delay_value = 57):
		"""
		Sets the digital delay for "element" to "delay_value". Not sent to octave yet.
		:param machine: must have input!
		:param element: "resonators" or "qubits"
		:param delay_value: the value delay; 57 (default)

		:return: machine; should take it!
		"""
		#qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		#qm = qmm.open_qm(config)
		
		# Set delay, including machine
			# note delay is defined directly for Octave
		if element == "resonators":
		#	for i in range(len(machine.resonators)):
		#		qm.set_digital_delay(machine.resonators[i].name,"output_switch",delay_value) # this sets configuration
			machine.octaves[0].LO_sources[0].digital_marker.delay = delay_value # this sets quam
		elif element =="qubits":
		#	for i in range(len(machine.qubits)):
		#		qm.set_digital_delay(machine.qubits[i].name,"output_switch",delay_value)
			machine.octaves[0].LO_sources[1].digital_marker.delay = delay_value
		return machine

	def set_digital_buffer(self, machine, element, buffer_value = 18):
		"""
		Sets the digital buffer for "element" to "buffer_value". Not sent to octave yet.
		:param machine: must have input!
		:param element: "resonators" or "qubits"
		:param buffer_value: the value buffer; 18 (default)

		:return: machine; should take it!
		"""
		#qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		#qm = qmm.open_qm(config)
		
		# Set buffer, including machine
			# note buffer is defined directly for Octave
		if element == "resonators":
		#	for i in range(len(machine.resonators)):
		#		qm.set_digital_buffer(machine.resonators[i].name,"output_switch",buffer_value) # this sets configuration
			machine.octaves[0].LO_sources[0].digital_marker.buffer = buffer_value # this sets quam
		elif element =="qubits":
		#	for i in range(len(machine.qubits)):
		#		qm.set_digital_buffer(machine.qubits[i].name,"output_switch",buffer_value)
			machine.octaves[0].LO_sources[1].digital_marker.buffer = buffer_value
		return machine

	def set_LO_frequency(self, machine, LO_channel, LO_frequency, octave_index = 0):
		"""
		Sets the LO frequency of octave in configuration. Not sent to octave yet.
		:param machine: must have input!
		:param LO_channel: 0, 1, 2 for the three LO in an octave
		:param LO_frequency: set value (Hz)
		:param octave_index: if have multiple octaves

		:return: machine; should take it!
		"""
		machine.octaves[octave_index].LO_sources[LO_channel].LO_frequency = LO_frequency

		# this will open a quantum machine and apply the settings
		#qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		#qm = qmm.open_qm(config) 
		return machine

	def set_rf_output_mode(self, machine, LO_channel, rf_output_mode, octave_index = 0):
		"""
		Sets the rf output mode of octave. Not sent to octave yet.
		:param machine: must have input!
		:param LO_channel: 0, 1, 2 for the three LO in an octave
		:param rf_output_mode: "always_on" / "always_off" (default)/ "triggered" / "triggered_reversed"
		:param octave_index: if have multiple octaves

		:return: machine; should take it!
		"""

		if rf_output_mode not in ["always_on","always_off","triggered","triggered_reversed"]:
			print("Input rf_output_mode not accepted value. Abort...")
			return machine

		machine.octaves[octave_index].LO_sources[LO_channel].output_mode = rf_output_mode

		# this will open a quantum machine and apply the settings
		#qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		#qm = qmm.open_qm(config) 
		return machine

	def set_rf_output_gain_qubit(self, machine, qubit_index, gain_value = 0, octave_index = 0):
		"""
		Sets the RF_output_gain for qubits[qubit_index]. Not sent to octave yet.
		:param machine: must have input!
		:param qubit_index: 
		:param gain_value: # can be in the range [-20 : 0.5 : 20]dB

		:return: machine; should take it!
		"""
		machine.qubits[qubit_index].hardware_parameters.RF_output_gain = gain_value
		machine.octaves[octave_index].LO_sources[1].gain = gain_value
		return machine

	def set_rf_output_gain_resonator(self, machine, res_index, gain_value = 0, octave_index = 0):
		"""
		Sets the RF_output_gain for resonators[res_index]. Not sent to octave yet.
		:param machine: must have input!
		:param res_index: 
		:param gain_value: # can be in the range [-20 : 0.5 : 20]dB

		:return: machine; should take it!
		"""
		machine.resonators[res_index].hardware_parameters.RF_output_gain = gain_value
		machine.octaves[octave_index].LO_sources[0].gain = gain_value
		return machine

	def set_input_attenuators(self, machine, LO_channel, input_attenuators, octave_index = 0):
		if input_attenuators not in ["ON","OFF"]:
			print("input_attenuators value not ON/OFF. Abort...")
			return machine
		machine.octaves[octave_index].LO_sources[LO_channel].input_attenuators = intput_attenuators
		return machine
			
