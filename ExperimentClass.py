"""
This file contains useful python functions meant to simplify the Jupyter notebook.
ExperimentHandle
written by Mo Chen in Oct. 2023
"""
# not setting octave here anymore
# these are just for octave_calibration
# from set_octave import ElementsSettings, octave_settings
# from qm.QuantumMachinesManager import QuantumMachinesManager
# from configuration import *
# from qm.octave.octave_manager import ClockMode # for setting external clock
from quam import QuAM
import json
import datetime
import os

from ExperimentClass_1D import EH_exp1D
from ExperimentClass_2D import EH_exp2D
from ExperimentClass_Labber import EH_Labber
from ExperimentClass_Octave import EH_Octave

class EH_expsave:
	def __init__(self):
		pass

class ExperimentHandle:
	def __init__(self):
		self.exp1D = EH_exp1D(self.update_tPath,self.update_str_datetime,self.octave_calibration)
		self.exp2D = EH_exp2D(self.update_tPath,self.update_str_datetime,self.octave_calibration)
		#self.expsave = EH_expsave()
		self.set_Labber = EH_Labber(self.update_tPath,self.update_str_datetime)
		self.set_octave = EH_Octave(self,update_tPath,self.update_str_datetime)

		now = datetime.datetime.now()
		year = now.strftime("%Y")
		month = now.strftime("%m")
		day = now.strftime("%d")
		tPath = os.path.join(r'Z:\LabberData_DF5\QM_Data_DF5',year,month,'Data_'+month+day)
		if not os.path.exists(tPath):
			os.makedirs(tPath)
		self.tPath = tPath

	def update_tPath(self):
		now = datetime.datetime.now()
		year = now.strftime("%Y")
		month = now.strftime("%m")
		day = now.strftime("%d")
		tPath = os.path.join(r'Z:\LabberData_DF5\QM_Data_DF5',year,month,'Data_'+month+day)
		if not os.path.exists(tPath):
			os.makedirs(tPath)
		self.tPath = tPath
		return tPath

	def update_str_datetime(self):
		now = datetime.datetime.now()
		month = now.strftime("%m")
		day = now.strftime("%d")
		hour = now.strftime("%H")
		minute = now.strftime("%M")
		f_str_datetime = month + day + '-' + hour + minute
		return f_str_datetime

