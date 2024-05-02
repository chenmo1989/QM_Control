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

from ExperimentClass_1D import EH_exp1D
from ExperimentClass_2D import EH_exp2D
from ExperimentClass_Labber import EH_Labber
from ExperimentClass_Octave import EH_Octave
from DataLoggingClass import DataLoggingHandle

class EH_expsave:
	def __init__(self):
		pass

class ExperimentHandle:
	def __init__(self):
		self.exp1D = EH_exp1D(self.set_octave,self.set_Labber,self.datalogs)
		self.exp2D = EH_exp2D(self.set_octave,self.set_Labber,self.datalogs)
		self.set_Labber = EH_Labber()
		self.set_octave = EH_Octave()
		self.datalogs = DataLoggingHandle()
