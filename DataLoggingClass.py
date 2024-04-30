"""
This file contains useful python functions to log data
written by Mo Chen in April 2024
"""
from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig, LoopbackInterface, generate_qua_script
from scipy import signal
from qualang_tools.units import unit
from qm.octave import QmOctaveConfig
from quam import QuAM
from typing import Union
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

import datetime
import os
import time
import glob
from macros import datetime_format_string

import Labber

class DataLoggingHandle:

	def __init__(self):


	def save(self, xrdataset):
		# timestamp
		created_timestamp_string = xrdataset.attrs["base_params"]["created"]
		created_timestamp = datetime.strptime(created_timestamp_string, datetime_format_string)
		# add non-QM machine settings
		xrdataset.attrs['machine_params_non_QM'] = self.save_machien_params_non_QM()
		
		tPath = self.generate_save_path(created_timestamp)
		xrdataset.attrs["base_params"]["directory"] = tPath
		result_filepath = os.path.join(tPath, self.generate_filename(xrdataset.attrs["base_params"]["description"], created_timestamp, tPath))
		xrdataset.to_netcdf(result_filepath)
		print('-'*10 + 'saved to ' + result_filepath) 

	def generate_save_path(self, created_timestamp):
		year = created_timestamp.strftime("%Y")
		month = created_timestamp.strftime("%m")
		day = created_timestamp.strftime("%d")
		tPath = os.path.join(r'Z:/QM_Data_DF5',year,month,'Data_'+month+day+'/')
		if not os.path.exists(tPath):
			os.makedirs(tPath)
		self.tPath = tPath
		return tPath

	def generate_filename(self, expt_prefix, created_timestamp, tPath):
		num_file = len(glob.glob(tPath + expt_prefix+'*'))

		date = '{}'.format(created_timestamp.strftime('%Y-%m-%d'))

    	if num_file > 0:
    		tFilename = "{}_{}.nc".format(date, expt_prefix)
    	else:
    		tFilename = "{}_{}_{}.nc".format(date, expt_prefix, num_file+1)

		return tFilename

	def save_machien_params_non_QM(self):
		machine_params_non_QM = {}
		client = Labber.connectToServer('localhost')  # get list of instruments
		for keys in client.getListOfInstruments():
			if keys[1]['interface'] != 'None':
				instru = client.connectToInstrument(keys[0], dict(interface = keys[1]['interface'], address = keys[1]['address'])) # connect to Instrument
				if instru.isRunning():
					machine_params_non_QM[keys[0]] = instrument_obj.getInstrConfig()
		
		client.close()

		return machine_params_non_QM


