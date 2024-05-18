"""
This file contains Labber controls
written by Mo Chen in April, 2024
"""
import Labber
import json
from quam import QuAM
import numpy as np

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
		self.client = Labber.connectToServer('localhost')  # get list of instruments
		self.QDevil = self.client.connectToInstrument('QDevil QDAC', dict(interface='Serial', address='3'))
		# digital attenuators
		self.Vaunix1 = self.client.connectToInstrument('Painter Vaunix Lab Brick Digital Attenuator',
											 dict(interface='USB', address='25606'))
		self.Vaunix2 = self.client.connectToInstrument('Painter Vaunix Lab Brick Digital Attenuator',
											 dict(interface='USB', address='25607'))
		# TWPA pump
		self.SG = self.client.connectToInstrument('Rohde&Schwarz RF Source', dict(interface='TCPIP', address='192.168.88.2'))


	def get_value_from_QDAC(self, machine):
		for n in range(len(machine.dc_flux)):
			if n + 1 < 10:
				machine.dc_flux[n].dc_voltage = self.QDevil.getValue("CH0" + str(n + 1) + " Voltage")
			else:
				machine.dc_flux[n].dc_voltage = self.QDevil.getValue("CH" + str(n + 1) + " Voltage")

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
		
		# reset all QDevil channels to 0 V
		for n in range(24):
			if n + 1 < 10:
				self.QDevil.setValue("CH0" + str(n + 1) + " Voltage", 0.0)
			else:
				self.QDevil.setValue("CH" + str(n + 1) + " Voltage", 0.0)

		# set dc_voltage in machine to 0 V
		for n in range(len(machine.flux_lines)):
			machine.dc_flux[n].dc_voltage = 0.0 + 0E1
		
		return machine


	def set_QDAC_single(self, machine, qubit_index, dc_value):
		"""
		function to set QDAC for single flux (specified by qubit_index), and update object "machine"
		1. set QDAC for flux_lines[qubit_index] to dc_value
		2. update dc_voltage in machine

		Args:
			machine: from input
		Return:
			machine: with updated dc_voltage
		"""
		if isinstance(dc_value, np.floating): # turn numpy into python float
			dc_value = dc_value.item()
		# find out the mapping between qubit_index and physical index by the name
		channel_index = int(machine.qubits[qubit_index].name[1:])
		# update dc_value to machine
		machine.dc_flux[channel_index].dc_voltage = dc_value + 0E1
		# connect to server
		self.QDevil.setValue("CH0" + str(channel_index + 1) + " Voltage", dc_value)
		
		return machine


	def set_QDAC_all(self, machine):
		"""
		function to set QDAC for all fluxes, and update object "machine"
		1. set QDAC for flux_lines[qubit_index] to dc_value
		2. update dc_voltage in machine

		Args:
			machine: initially from quam_state.json or from input
		Return:
			machine: no change
		"""

		# Set qubits to desired dc value
		for dc_flux_index in range(len(machine.dc_flux)):
			self.QDevil.setValue("CH0" + str(dc_flux_index + 1) + " Voltage", machine.dc_flux[dc_flux_index].dc_voltage)

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
		for dc_flux_index in range(len(machine.dc_flux)):
			self.QDevil.setValue("CH0" + str(dc_flux_index + 1) + " Voltage", machine.dc_flux[dc_flux_index].dc_voltage)

		
		self.Vaunix1.setValue("Attenuation", machine.resonators[qubit_index].RO_attenuation[0])
		self.Vaunix2.setValue("Attenuation", machine.resonators[qubit_index].RO_attenuation[1])

		
		self.SG.setValue('Frequency', machine.resonators[qubit_index].TWPA[0])
		self.SG.setValue('Power', machine.resonators[qubit_index].TWPA[1])
		self.SG.setValue('Output', True)

		return machine		

	def close_client(self):
		self.client.close()
