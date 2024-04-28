"""
This file contains useful python functions meant to simplify the Jupyter notebook.
ExperimentHandle
written by Mo Chen in Oct. 2023
-------------------------------------------------------------
Update in April 2024
To acommodate the new QOP and python API, I have updated machine, configuration, and moved all octave functions into a Class.
Note that QM now prefers using configurations rather than API commands to change octave settings. This is reflected in the ExperimentClass_Octave.py methods. 
Only changes are made, and the idea is that once I run either calibration or experiments, these changes will be updated automatically.
"""
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
		self.exp1D = EH_exp1D(self.update_tPath,self.update_str_datetime,self.set_octave,self.set_Labber)
		self.exp2D = EH_exp2D(self.update_tPath,self.update_str_datetime,self.set_octave,self.set_Labber)
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

