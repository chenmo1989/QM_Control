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
		now = datetime.datetime.now()
		year = now.strftime("%Y")
		month = now.strftime("%m")
		day = now.strftime("%d") 
		tPath = os.path.join(r'Z:/QM_Data_DF5',year,month,'Data_'+month+day+'/')
		if not os.path.exists(tPath):
			os.makedirs(tPath)
		self.tPath = tPath

	def save(self, xrdataset):
		created_timestamp_string = xrdataset.attrs["base_params"]["created"]
		created_timestamp = datetime.strptime(created_timestamp_string, datetime_format_string)

		tPath = self.generate_save_path(created_timestamp)

		result_filepath = os.path.join(tPath, self.generate_filename(xrdataset.attrs["base_params"]["description"], created_timestamp, tPath))

		xrdataset.to_netcdf(result_filepath)

		return 

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
    		tFilename = "{}_{}.nc".format(expt_prefix, date)
    	else:
    		tFilename = "{}_{}_{}.nc".format(expt_prefix, date, num_file+1)

		return tFilename

	def save_machien_params_non_QM(self):
		client = Labber.connectToServer('localhost')  # get list of instruments

		# reset all QDevil channels to 0 V
		QDevil = client.connectToInstrument('QDevil QDAC', dict(interface='Serial', address='3'))

		client.close()


	def update_tPath(self):
		now = datetime.datetime.now()
		year = now.strftime("%Y")
		month = now.strftime("%m")
		day = now.strftime("%d") 
		tPath = os.path.join(r'Z:/QM_Data_DF5',year,month,'Data_'+month+day+'/')
		if not os.path.exists(tPath):
			os.makedirs(tPath)
		self.tPath = tPath


