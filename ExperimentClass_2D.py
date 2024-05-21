"""
This file contains useful python functions meant to simplify the Jupyter notebook.
ExperimentHandle.exp2D
written by Mo Chen in Oct. 2023
"""
from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig, LoopbackInterface, generate_qua_script
from qm.octave import *
from configuration import *
from scipy import signal
from qm import SimulationConfig
from qualang_tools.bakery import baking
from qm.octave import QmOctaveConfig
#from set_octave import ElementsSettings, octave_settings
from quam import QuAM
from scipy.io import savemat, loadmat
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
#from qutip import *
from typing import Union
from macros import ham, readout_rotated_macro, declare_vars, wait_until_job_is_paused
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
import Labber

#warnings.filterwarnings("ignore")

# improt experiments
from ExperimentClass_2D_RR import EH_RR
from ExperimentClass_2D_Rabi import EH_Rabi
from ExperimentClass_2D_SWAP import EH_SWAP
from ExperimentClass_2D_exp1D import EH_1D


class EH_exp2D:
	"""
	Class for running 2D experiments
	Attributes:

	Methods (useful ones):
		RR: a class for running readout resonator related experiments
	"""


	def __init__(self, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm):
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs
		self.qmm = ref_to_qmm
		self.exp1D = EH_1D(ref_to_qmm)
		self.RR = EH_RR(self.exp1D, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm)
		self.Rabi = EH_Rabi(self.exp1D, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm)
		self.SWAP = EH_SWAP(ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm)


