"""
This file contains useful python functions to log data
written by Mo Chen in April 2024
"""
from quam import QuAM
from typing import Union
import json
import numpy as np
import xarray as xr
import pandas as pd

import datetime
import os
import time
import glob
from configuration import datetime_format_string

import Labber

class DataLoggingHandle:

	def __init__(self):
		pass

	def save(self, expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence):
		# save base attributes
		expt_dataset.attrs['created'] = timestamp_created.strftime(datetime_format_string())
		expt_dataset.attrs['finished'] = timestamp_finished.strftime(datetime_format_string())
		expt_dataset.attrs['name'] = expt_name
		expt_dataset.attrs['long_name'] = expt_long_name
		expt_dataset.attrs['qubit'] = expt_qubits
		expt_dataset.attrs['TLS'] = expt_TLS
		expt_dataset.attrs['sequence'] = expt_sequence

		# add units to coordinates
		expt_dataset = self.add_attrs_units(expt_dataset)
		# add description
		expt_dataset = self.add_description(expt_dataset)
		# add non-QM machine settings
		#expt_dataset.attrs['machine_params_non_QM'] = self.save_machine_params_non_QM()

		# generate path, filename, save directory
		tPath = self.generate_save_path(timestamp_created)
		result_filepath = os.path.join(tPath, self.generate_filename(expt_dataset.attrs["description"], timestamp_created, tPath)) # without extension
		expt_dataset.attrs["directory"] = result_filepath
		# save data and json settings
		print('-'*10 + 'saved to ' + result_filepath) 
		expt_dataset.to_netcdf(result_filepath + '.nc')
		machine._save(result_filepath + '.json')
		
		return expt_dataset

	def add_description(self, expt_dataset):
		describ_text = ''
		for qubit_index, qubit in enumerate(expt_dataset.attrs['qubit']):
			if qubit_index >0:
				describ_text = describ_text + '-' + qubit
			else:
				describ_text = describ_text + qubit
		for tls_index, tls in enumerate(expt_dataset.attrs['TLS']):
			if tls_index >0:
				describ_text = describ_text + '-' + tls
			else:
				describ_text = describ_text + tls_index
		expt_dataset.attrs['description'] = "{}_{}".format(describ_text, expt_dataset.attrs['name'])
		return expt_dataset

	def generate_save_path(self, timestamp_created):
		year = timestamp_created.strftime("%Y")
		month = timestamp_created.strftime("%m")
		day = timestamp_created.strftime("%d")
		tPath = os.path.join(r'Z:\QM_Data_DF5',year,month,'Data_'+month+day)
		if not os.path.exists(tPath):
			os.makedirs(tPath)
		self.tPath = tPath
		return tPath

	def generate_filename(self, expt_prefix, timestamp_created, tPath):
		date = '{}'.format(timestamp_created.strftime('%Y-%m-%d'))
		tFilename = "{}_{}".format(date, expt_prefix)

		num_file = len(glob.glob(os.path.join(tPath,tFilename)+'*.nc'))

		if num_file > 0:
			tFilename = "{}_{}".format(tFilename, num_file+1)

		return tFilename

	def save_machine_params_QM(self, expt_dataset, machine):
		machine_params_QM = {}
		for keys in machine:
			machine_params_QM[keys] = machine[keys]
		expt_dataset.attrs['machine_params_QM'] = machine_params_QM
		return expt_dataset

	def save_machine_params_non_QM(self):
		machine_params_non_QM = {}
		client = Labber.connectToServer('localhost')  # get list of instruments
		for keys in client.getListOfInstruments():
			if keys[1]['interface'] != 'None':
				instru = client.connectToInstrument(keys[0], dict(interface = keys[1]['interface'], address = keys[1]['address'])) # connect to Instrument
				if instru.isRunning():
					machine_params_non_QM[keys[0]] = instrument_obj.getInstrConfig()
		
		client.close()

		return machine_params_non_QM

	def add_attrs_units(self, expt_dataset):
		# coordinate units
		for keys in expt_dataset.coords:
			if 'Flux' in keys:
				expt_dataset[keys].attrs['units'] = 'V'
			elif 'Volt' in keys:
				expt_dataset[keys].attrs['units'] = 'V'
			elif 'Amplitude' in keys:
				expt_dataset[keys].attrs['units'] = 'V'		
			elif 'Time' in keys:
				expt_dataset[keys].attrs['units'] = 'ns'
			elif 'Duration' in keys:
				expt_dataset[keys].attrs['units'] = 'ns'
			elif 'Length' in keys:
				expt_dataset[keys].attrs['units'] = 'ns'
			elif 'Frequency' in keys:
				expt_dataset[keys].attrs['units'] = 'Hz'
			elif 'Detuning' in keys:
				expt_dataset[keys].attrs['units'] = 'Hz'
			elif 'Phase' in keys:
				expt_dataset[keys].attrs['units'] = 'rad'
			elif 'Delay' in keys:
				expt_dataset[keys].attrs['units'] = 'ns'
		# data units
		if 'pe' in expt_dataset:
			expt_dataset.attrs['units'] = '' # population Pe, unit is 1
		elif 'I' in expt_dataset.attrs:
			expt_dataset.attrs['units'] = 'V'
		return expt_dataset

