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
		initialize_QDAC: initialize all QDAC channels to 0V and update ``machine''
		set_QDAC_single: set QDAC single channel to dc_value for qubit_index
		set_QDAC_all: set QDAC all channels according to ``machine''
		set_Labber: initialize all Labber controled instruments according to ``machine'' and qubit_index
	"""
	def __init__(self):
		pass


	def get_value_from_QDAC(self, machine):
		client = Labber.connectToServer('localhost')  # get list of instruments
		QDevil = client.connectToInstrument('QDevil QDAC', dict(interface='Serial', address='3'))

		for n in range(len(machine.dc_flux)):
			if n + 1 < 10:
				machine.dc_flux[n].dc_voltage = QDevil.getValue("CH0" + str(n + 1) + " Voltage")
			else:
				machine.dc_flux[n].dc_voltage = QDevil.getValue("CH" + str(n + 1) + " Voltage")
		client.close()

		return machine


	def initialize_QDAC(self, machine):
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
		for n in range(len(machine.flux_lines)):
			machine.dc_flux[n].dc_voltage = 0.0 + 0E1
		
		return machine


	def set_QDAC_single(self,machine,qubit_index,dc_value):
		"""
		function to set QDAC for single flux (specified by qubit_index), and update object "machine"
		1. set QDAC for flux_lines[qubit_index] to dc_value
		2. update dc_voltage in machine

		Args:
			machine: from input
		Return:
			machine: with updated dc_voltage
		"""

		# find out the mapping between qubit_index and physical index by the name
		channel_index = int(machine.qubits[qubit_index].name[1:])
		# update dc_value to machine
		machine.dc_flux[channel_index].dc_voltage = dc_value + 0E1
		# connect to server
		client = Labber.connectToServer('localhost')  # get list of instruments
		QDevil = client.connectToInstrument('QDevil QDAC', dict(interface='Serial', address='3'))
		QDevil.setValue("CH0" + str(channel_index + 1) + " Voltage", dc_value)
		client.close()
		return machine


	def set_QDAC_all(self,machine):
		"""
		function to set QDAC for all fluxes, and update object "machine"
		1. set QDAC for flux_lines[qubit_index] to dc_value
		2. update dc_voltage in machine

		Args:
			machine: initially from quam_state.json or from input
		Return:
			machine: no change
		"""

		# connect to server
		client = Labber.connectToServer('localhost')  # get list of instruments
		QDevil = client.connectToInstrument('QDevil QDAC', dict(interface='Serial', address='3'))

		# Set qubits to desired dc value
		for dc_flux_index in range(len(machine.dc_flux)):
			QDevil.setValue("CH0" + str(dc_flux_index + 1) + " Voltage", machine.dc_flux[dc_flux_index].dc_voltage)

		client.close()

		return machine


	def set_Labber(self, machine, qubit_index):
		"""
		function to set Labber controlled hardware, according to object "machine"
		1. set QDAC CH (qubit_index+1) to the saved dc_voltage saved in "machine"
		2. set Vaunix digital attenuators to ROI, ROO values saved in "machine"
		3. set TWPA pumping frequency and power to values saved in "machine"

		Args:
			machine:
			qubit_index: changing TWPA, attenuator, and octave LO1 settings accordingly
		Return:
			machine
		"""
		# connect to server
		client = Labber.connectToServer('localhost')  # get list of instruments

		# Set qubits to desired dc value
		QDevil = client.connectToInstrument('QDevil QDAC', dict(interface='Serial', address='3'))
		for dc_flux_index in range(len(machine.dc_flux)):
			QDevil.setValue("CH0" + str(dc_flux_index + 1) + " Voltage", machine.dc_flux[dc_flux_index].dc_voltage)

		# digital attenuators
		Vaunix1 = client.connectToInstrument('Painter Vaunix Lab Brick Digital Attenuator',
											 dict(interface='USB', address='25606'))
		Vaunix2 = client.connectToInstrument('Painter Vaunix Lab Brick Digital Attenuator',
											 dict(interface='USB', address='25607'))
		Vaunix1.setValue("Attenuation", machine.resonators[qubit_index].RO_attenuation[0])
		Vaunix2.setValue("Attenuation", machine.resonators[qubit_index].RO_attenuation[1])

		# TWPA pump
		SG = client.connectToInstrument('Rohde&Schwarz RF Source', dict(interface='TCPIP', address='192.168.88.2'))
		SG.setValue('Frequency', machine.resonators[qubit_index].TWPA[0])
		SG.setValue('Power', machine.resonators[qubit_index].TWPA[1])
		SG.setValue('Output', True)

		client.close()
		
		return machine		
