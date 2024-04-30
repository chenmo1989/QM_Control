"""
This file contains useful python functions meant to simplify the Jupyter notebook.
ExperimentHandle.exp1D
written by Mo Chen in Oct. 2023
"""
from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig, LoopbackInterface, generate_qua_script
from qm.octave import *
from qm.octave.octave_manager import ClockMode
from configuration import *
from scipy import signal
from qualang_tools.bakery import baking
from qualang_tools.units import unit
from qm.octave import QmOctaveConfig
#from set_octave import ElementsSettings, octave_settings
from quam import QuAM
from scipy.io import savemat, loadmat
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter
from qutip import *
from typing import Union
from macros import *
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# improt experiments
from ExperimentClass_1D_RR import EH_RR
from ExperimentClass_1D_Rabi import EH_Rabi
from ExperimentClass_1D_T1 import EH_T1
from ExperimentClass_1D_SWAP import EH_SWAP
from ExperimentClass_1D_Ramsey import EH_Ramsey
from ExperimentClass_1D_DD import EH_DD

class EH_exp1D:
	"""
	Class for running 1D experiments
	Attributes:

	Methods (useful ones):
		set_Labber:
		set_octave:
		RR: a class for running readout resonator related experiments
		Rabi: a class for running Rabi sequence based experiments
		T1: a class for running T1 sequence based experiments
		SWAP: a class for running SWAP sequence based experiments
		Ramsey: a class for running Ramsey sequence based experiments
		DD: a class for running Dynamical Decoupling sequence based experiments
	"""
	def __init__(self,ref_to_update_tPath, ref_to_update_str_datetime, ref_to_set_octave, ref_to_set_Labber):
		self.set_Labber = ref_to_set_Labber
		self.set_octave = ref_to_set_octave
		self.RR = EH_RR(ref_to_update_tPath,ref_to_update_str_datetime,ref_to_set_octave)
		self.Rabi = EH_Rabi(ref_to_update_tPath,ref_to_update_str_datetime,ref_to_set_octave)
		self.SWAP = EH_SWAP(ref_to_update_tPath, ref_to_update_str_datetime, ref_to_set_octave)
		self.DD = EH_DD(ref_to_update_tPath,ref_to_update_str_datetime, ref_to_set_octave)
		self.T1 = EH_T1(ref_to_update_tPath,ref_to_update_str_datetime, ref_to_set_octave)
		self.Ramsey = EH_Ramsey(ref_to_update_tPath, ref_to_update_str_datetime, ref_to_set_octave)

