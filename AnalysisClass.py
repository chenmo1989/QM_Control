"""
This file contains useful python functions meant to simplify the Jupyter notebook.
AnalysisHandle
written by Mo Chen in Oct. 2023
"""
from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig, LoopbackInterface
#from configuration import *
from scipy import signal
from qm import SimulationConfig
from qualang_tools.units import unit
from quam import QuAM
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
#from qutip import *
from typing import Union
import datetime
import os
import time
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np

from AnalysisClass_1D import AH_exp1D
from AnalysisClass_2D import AH_exp2D

class AnalysisHandle:
	def __init__(self, json_name):
		# for updated values
		self.ham_param = []
		self.poly_param = []
		self.json_name = json_name
		self.exp1D = AH_exp1D(self.ham_param, self.poly_param, self.json_name)
		self.exp2D = AH_exp2D(self.ham_param, self.poly_param, self.json_name, self.exp1D)
		

	def get_machine(self):
		machine = QuAM(self.json_name)
		#config = build_config(machine)
		return machine

	def set_machine(self,machine, json_name = None):
		if json_name is None:
			machine._save(machine.global_parameters.name)
		else:
			machine._save(json_name)
		return machine

	def update_machine_qubit_frequency(self,machine,qubit_index,new_freq):
		machine.qubits[qubit_index].f_01 = new_freq
		return machine

	def update_machine_qubit_frequency_rel(self,machine,qubit_index,new_freq):
		machine.qubits[qubit_index].f_01 += new_freq
		return machine

	def update_machine_res_frequency(self,machine,qubit_index,new_freq):
		machine.resonators[qubit_index].f_readout = new_freq + 0E6
		return machine

	def update_machine_res_frequency_rel(self,machine,qubit_index,new_freq):
		machine.resonators[qubit_index].f_readout += new_freq + 0E6
		return machine

	def update_machine_res_frequency_sweet_spot(self, machine, qubit_index, dc_flux_index):
		machine.resonators[qubit_index].f_readout = int(self.exp2D.ham([machine.dc_flux[dc_flux_index].max_frequency_point], *machine.resonators[qubit_index].tuning_curve, output_flag = 1).item()) * 1E6 + 0E6
		return machine

	def update_analysis_tuning_curve(self,qubit_index,ham_param = None, poly_param = None,is_DC_curve = False):
		if ham_param is None:
			self.ham_param = self.get_machine().resonators[qubit_index].tuning_curve
		if poly_param is None:
			if is_DC_curve:
				self.poly_param = self.get_machine().qubits[qubit_index].DC_tuning_curve
			else:
				self.poly_param = self.get_machine().qubits[qubit_index].AC_tuning_curve
		return
		
	def get_sweept_spot(self,poly_param = None):
		if poly_param is None:
			poly_param = self.poly_param
		if np.size(poly_param) > 3:
			print("polynomial order > 2")
			return None
		else:
			return -(poly_param[1]/2/poly_param[0])



