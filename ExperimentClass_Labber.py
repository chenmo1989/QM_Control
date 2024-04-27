"""
This file contains Labber controls
written by Mo Chen in April, 2024
"""
import Labber
import json
from quam import QuAM

class EH_Labber:
	"""
	Class for operating Labber (for controlling non-QM instruments)
	Attributes:

	Methods (useful ones):
		update_tPath: reference to Experiment.update_tPath
		update_str_datetime: reference to Experiment.update_str_datetime
		initialize_QDAC: initialize all QDAC channels to 0V and update ``machine''
		set_QDAC_single: set QDAC single channel to dc_value for flux_index
		set_QDAC_all: set QDAC all channels according to ``machine''
		set_Labber: initialize all Labber controled instruments according to ``machine'' and res_index
	"""
	def __init__(self,ref_to_update_tPath, ref_to_update_str_datetime):
		self.update_tPath = ref_to_update_tPath
		self.update_str_datetime = ref_to_update_str_datetime


	def initialize_QDAC(self, machine = None):
		"""
		function to initialize QDAC, and update object "machine"
		1. set QDAC to 0 V
		2. update dc_voltage in machine to 0 V

		Args:
			machine: initially from quam_state.json or from input
		Return:
			machine: with updated dc_voltage = 0.0
		"""
		# connect to server
		client = Labber.connectToServer('localhost')  # get list of instruments

		# reset all QDevil channels to 0 V
		QDevil = client.connectToInstrument('QDevil QDAC', dict(interface='Serial', address='3'))
		for n in range(24):
			if n + 1 < 10:
				QDevil.setValue("CH0" + str(n + 1) + " Voltage", 0.0)
			else:
				QDevil.setValue("CH" + str(n + 1) + " Voltage", 0.0)
		client.close()

		# set dc_voltage in machine to 0 V
		if machine is None:
			machine = QuAM("quam_state.json")

		for n in range(len(machine.flux_lines)):
			machine.flux_lines[n].hardware_parameters.dc_voltage = 0.0 + 0E1
		machine._save("quam_state.json") # save machine

		return machine

	def set_QDAC_single(self,machine,flux_index,dc_value):
		"""
		function to set QDAC for single flux (specified by flux_index), and update object "machine"
		1. set QDAC for flux_lines[flux_index] to dc_value
		2. update dc_voltage in machine

		Args:
			machine: from input
		Return:
			machine: with updated dc_voltage
		"""
		# update dc_value to machine
		machine.flux_lines.hardware_parameters[flux_index].dc_voltage = dc_value + 0E1
		machine._save("quam_state.json") # save machine
		# connect to server
		client = Labber.connectToServer('localhost')  # get list of instruments
		QDevil = client.connectToInstrument('QDevil QDAC', dict(interface='Serial', address='3'))
		QDevil.setValue("CH0" + str(flux_index + 1) + " Voltage", dc_value)
		client.close()
		return machine

	def set_QDAC_all(self,machine = None):
		"""
		function to set QDAC for all fluxes, and update object "machine"
		1. set QDAC for flux_lines[flux_index] to dc_value
		2. update dc_voltage in machine

		Args:
			machine: initially from quam_state.json or from input
		Return:
			machine: no change
		"""

		if machine is None:
			machine = QuAM("quam_state.json")

		# connect to server
		client = Labber.connectToServer('localhost')  # get list of instruments
		QDevil = client.connectToInstrument('QDevil QDAC', dict(interface='Serial', address='3'))

		# set dc_value to QDAC
		for flux_index in range(len(machine.flux_lines)):
			QDevil.setValue("CH0" + str(flux_index + 1) + " Voltage", machine.flux_lines.hardware_parameters[flux_index].dc_voltage)
		# machine._save("quam_state.json") # no need to save, since nothing is changed
		client.close()

		return machine

	def set_Labber(self, machine = None, res_index):
		"""
		function to set Labber controlled hardware, according to object "machine"
		1. set QDAC CH (flux_index+1) to the saved dc_voltage saved in "machine"
		2. set Vaunix digital attenuators to ROI, ROO values saved in "machine"
		3. set TWPA pumping frequency and power to values saved in "machine"

		Args:
			machine: initially from quam_state.json or from input
			res_index: changing TWPA, attenuator, and octave LO1 settings accordingly
		Return:
			machine
		"""
		if machine is None:	
			machine = QuAM("quam_state.json")

		# connect to server
		client = Labber.connectToServer('localhost')  # get list of instruments

		# Set qubits to desired dc value
		for flux_index_tmp in range(len(machine.flux_lines)):
			QDevil.setValue("CH0" + str(flux_index_tmp + 1) + " Voltage", machine.flux_lines[flux_index_tmp].hardware_parameters.dc_voltage)

		# digital attenuators
		Vaunix1 = client.connectToInstrument('Painter Vaunix Lab Brick Digital Attenuator',
											 dict(interface='USB', address='25606'))
		Vaunix2 = client.connectToInstrument('Painter Vaunix Lab Brick Digital Attenuator',
											 dict(interface='USB', address='25607'))
		Vaunix1.setValue("Attenuation", machine.resonators[res_index].hardware_parameters.RO_attenuation[0])
		Vaunix2.setValue("Attenuation", machine.resonators[res_index].hardware_parameters.RO_attenuation[1])

		# TWPA pump
		SG = client.connectToInstrument('Rohde&Schwarz RF Source', dict(interface='TCPIP', address='192.168.88.2'))
		SG.setValue('Frequency', machine.resonators[res_index].hardware_parameters.TWPA[0])
		SG.setValue('Power', machine.resonators[res_index].hardware_parameters.TWPA[1])
		SG.setValue('Output', True)
		client.close()

		# set global input parameters to that of the res_index
		machine.octave.LO1.RO_delay = machine.resonators[res_index].hardware_parameters.RO_delay
		machine.octave.LO1.time_of_flight = machine.resonators[res_index].hardware_parameters.time_of_flight
        machine.octave.LO1.con1_downconversion_offset_I = machine.resonators[res_index].hardware_parameters.con1_downconversion_offset_I
        machine.octave.LO1.con1_downconversion_offset_Q = machine.resonators[res_index].hardware_parameters.con1_downconversion_offset_Q
        machine.octave.LO1.con1_downconversion_gain = machine.resonators[res_index].hardware_parameters.con1_downconversion_offset_gain
        # save machine and return
        machine._save("quam_state.json") # save machine

		return machine		
