from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig, LoopbackInterface, generate_qua_script
from qm.octave import *
from configuration import *
from scipy import signal
from qualang_tools.bakery import baking
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qm.octave import QmOctaveConfig
from quam import QuAM
from typing import Union
from macros import ham, readout_rotated_macro, declare_vars, wait_until_job_is_paused
import warnings
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
import xarray as xr



class EH_Rabi:
	"""
	class in ExperimentHandle, for qubit Rabi related 2D experiments
	Methods:
		update_tPath
		update_str_datetime
		qubit_freq_vs_dc_flux(self, poly_param, ham_param, dc_flux_sweep, qubit_index, n_avg = 1E3, cd_time = 20E3, tPath = None, f_str_datetime = None, to_simulate = False, simulation_len = 3000)
	"""
	def __init__(self, ref_to_local_exp1D, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm):
		self.exp1D = ref_to_local_exp1D
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs
		self.qmm = ref_to_qmm


	def qubit_freq_vs_dc_flux(self, machine, dc_flux_sweep, qubit_if_sweep, qubit_index, n_avg = 1E3, cd_time = 20E3, ham_param = None, poly_param = None, to_simulate=False, simulation_len=3000, final_plot=True, live_plot=False, data_process_method = 'I'):
		"""Run qubit spectroscopy vs dc flux (2D). 

		Requires an initial qubit DC_tuning_curve (could come from educated guess based on ham_param), and resonator ham_param.
		
		The whole 2D sweep consists of 1D scans, calling `self.exp1D.res_freq` and `self.exp1D.qubit_freq`.
		At each dc flux, we first run a 1D resonator spectroscopy, to identify the readout pulse frequency (minimum of Amplitude),
		then set readout pulse frequency, and run 1D qubit spectroscopy. Then move on to the next dc flux.
		Two datasets are saved, one with the resonator spectroscopy (2D, vs dc flux), one with the qubit spectroscopy (2D, vs dc flux), 
		and another 1D array saving the readout pulse frequency used for each dc flux.
		To distinguish from pure 2D resonator spectroscopy (usually for a larger range, covering a few Phi_0), with filename `res_spec2D`, here the resonator file is called `res_spec_vs_dc_flux`.
		This sweep is not squared!!

		Args:
			machine ([type]): [description]
			dc_flux_sweep ([type]): [description]
			qubit_if_sweep ([type]): [description]
			qubit_index ([type]): [description]
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `20E3`)
			ham_param ([type]): [description] (default: `None`)
			poly_param ([type]): [description] (default: `None`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `3000`)
			final_plot (bool): [description] (default: `True`)
			live_plot (bool): [description] (default: `False`)
			data_process_method (str): [description] (default: `'I'`)
		
		Returns:
			[type]: [description]
		"""


		# this gives us the option of fitting elsewhere (e.g. manually) and pass the fitted value in
		if poly_param is None:
			poly_param = machine.qubits[qubit_index].DC_tuning_curve
		if ham_param is None:
			ham_param = machine.resonators[qubit_index].tuning_curve

		# Initialize empty vectors to store the global 'I' & 'Q' results
		I_qubit_tot = []
		Q_qubit_tot = []
		qubit_freq_sweep_tot = []
		I_res_tot = []
		Q_res_tot = []
		res_freq_sweep_tot = []
		res_freq_tot = []

		if live_plot:
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [8, 4]

		# start time
		timestamp_created = datetime.datetime.now()

		# 2D scan, RR frequency vs DC flux
		for dc_index, dc_value in enumerate(dc_flux_sweep): # sweep over all dc fluxes
			# qubit frequency estimate and sweep range
			qubit_freq_est = int(np.polyval(poly_param, dc_value) * 1E6) # in Hz
			qubit_freq_sweep = qubit_freq_est + qubit_if_sweep

			# check if octave LO frequency needs change
			qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
			if qubit_lo - (qubit_freq_est + min(qubit_if_sweep)) > 350E6: # the flux is bringing qubit freq down, need to decrease LO
				qubit_lo = qubit_freq_est + max(qubit_if_sweep) - 350E6
				machine.octaves[0].LO_sources[1].LO_frequency = int(qubit_lo) + 0E6
				machine.qubits[qubit_index].f_01 = int(qubit_freq_est) + 0E6
				machine = self.set_octave.calibration(machine, qubit_index, qubit_only = True) # since RR is changing, calibrate both

			if qubit_lo - (qubit_freq_est + max(qubit_if_sweep)) < -350E6: # the flux is bringing qubit freq up, need to increase LO
				qubit_lo = qubit_freq_est + min(qubit_if_sweep) + 350E6
				machine.octaves[0].LO_sources[1].LO_frequency = int(qubit_lo) + 0E6
				machine.qubits[qubit_index].f_01 = int(qubit_freq_est) + 0E6
				machine = self.set_octave.calibration(machine, qubit_index, qubit_only = True) # since RR is changing, calibrate both
		
			# set dc flux value
			machine = self.set_Labber.set_QDAC_single(machine, qubit_index, dc_value)
			time.sleep(0.1)
			
			# 1D RR experiment
			res_freq_est = ham([dc_value], *ham_param, output_flag = 1) * 1E6 # to Hz
			res_freq_sweep = int(res_freq_est[0]) + np.arange(-5E6,5E6 + 1.0,0.05E6)

			if live_plot:
				machine, I_tmp, Q_tmp = self.exp1D.res_freq(machine, res_freq_sweep, qubit_index, n_avg=n_avg, cd_time=cd_time, to_simulate=to_simulate, simulation_len=simulation_len, fig = fig)
			else:
				machine, I_tmp, Q_tmp = self.exp1D.res_freq(machine, res_freq_sweep, qubit_index, n_avg=n_avg, cd_time=cd_time, to_simulate=to_simulate, simulation_len=simulation_len)

			res_freq_tmp = self.exp1D.res_freq_analysis(res_freq_sweep, I_tmp, Q_tmp, data_process_method = 'Amplitude') # because Phase or anything else may not be well calibrated away from sweet spot. Amplitude is the most robust
			
			# save 1D RR data
			I_res_tot.append(I_tmp) # list of np array
			Q_res_tot.append(Q_tmp) # list of np array
			res_freq_tot.append(res_freq_tmp)
			res_freq_sweep_tot.append(res_freq_sweep) # list of np array

			# set resonator freq for qubit spectroscopy
			machine.resonators[qubit_index].f_readout = int(res_freq_tmp) + 0E6

			# 1D qubit experiment
			if live_plot:
				machine, I_tmp, Q_tmp = self.exp1D.qubit_freq(machine, qubit_freq_sweep, qubit_index, ff_amp=0.0, n_avg=n_avg, cd_time=cd_time, to_simulate=to_simulate, simulation_len=simulation_len, fig = fig)
			else:
				machine, I_tmp, Q_tmp = self.exp1D.qubit_freq(machine, qubit_freq_sweep, qubit_index, ff_amp=0.0, n_avg=n_avg, cd_time=cd_time, to_simulate=to_simulate, simulation_len=simulation_len)

			I_qubit_tot.append(I_tmp) # list of np array
			Q_qubit_tot.append(Q_tmp) # list of np array
			qubit_freq_sweep_tot.append(qubit_freq_sweep)

			progress_counter(dc_index, len(dc_flux_sweep), start_time=datetime.datetime.timestamp(timestamp_created))

		# Experiments finished
		timestamp_finished = datetime.datetime.now()

		# save
		I_qubit = np.array(I_qubit_tot)
		Q_qubit = np.array(Q_qubit_tot)
		qubit_freq_sweep = np.array(qubit_freq_sweep_tot)
		res_freq = np.array(res_freq_tot)

		I_res = np.array(I_res_tot)
		Q_res = np.array(Q_res_tot)
		res_freq_sweep = np.array(res_freq_sweep_tot)

		# save res and qubit data into separate files
		# generate xarray dataset for resonator
		expt_dataset = xr.Dataset(
			{
				"I": (["x", "y"], I_res),
				"Q": (["x", "y"], Q_res),
			},
			coords={
				"DC_Flux": (["x"], dc_flux_sweep),
				"Resonator_Frequency": (["x", "y"], res_freq_sweep),
			},
		)

		expt_name = r'res_spec_vs_dc_flux'
		expt_long_name = r'Resonator Spectroscopy vs DC Flux'
		expt_qubits = [machine.qubits[qubit_index].name]
		expt_TLS = []  # use t0, t1, t2, ...
		expt_sequence = """"""

		# save data
		expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
										  expt_long_name, expt_qubits, expt_TLS, expt_sequence)

		# generate xarray dataset for qubit
		expt_dataset = xr.Dataset(
			{
				"I": (["x", "y"], I_qubit),
				"Q": (["x", "y"], Q_qubit),
				"res_freq": (["x"], res_freq),
			},
			coords={
				"DC_Flux": (["x"], dc_flux_sweep),
				"Qubit_Frequency": (["x", "y"], qubit_freq_sweep),
			},
		)

		expt_name = r'qubit_spec_vs_dc_flux'
		expt_long_name = r'Qubit Spectroscopy vs DC Flux'
		expt_qubits = [machine.qubits[qubit_index].name]
		expt_TLS = []  # use t0, t1, t2, ...
		expt_sequence = """"""

		expt_extra = {
			'n_ave': str(n_avg),
			'Qubit CD [ns]': str(cd_time)
		}

		# save data
		expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
										  expt_long_name, expt_qubits, expt_TLS, expt_sequence, expt_extra)

		# plot qubit spectroscopy vs dc flux
		if final_plot:
			if live_plot is False:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
			plt.cla()
			expt_dataset[data_process_method].plot(x = list(expt_dataset.coords.keys())[0], y = list(expt_dataset.coords.keys())[1], cmap = "seismic")
			plt.title(expt_dataset.attrs['long_name'])
			plt.show()

		return machine, expt_dataset


	def qubit_freq_vs_fast_flux_slow(self, machine, ff_sweep_abs, qubit_if_sweep, qubit_index, n_avg = 1E3, cd_time = 20E3, ff_to_dc_ratio = 4.5, poly_param = None, to_simulate=False, simulation_len=3000, final_plot=True, live_plot=False, data_process_method = 'I'):
		"""Run qubit spectroscopy vs fast flux (2D), slow version: does NOT require an initial AC_tuning_curve.
		
		This is supposed to be an initial coarse sweep, to get an estimate of the first AC_tuning_curve. 
		At each fast flux, the qubit frequency (sweep range) is estimated based on qubit DC_tuning_curve from `qubit_freq_vs_dc_flux`. 
		The input argument `ff_to_dc_ratio` is a guess of the effective ratio between fast and dc flux. 
		Then the qubit frequency is estimated to be around dc_flux = ff_to_dc_ratio * ff_sweep_abs + machine.dc_flux[channel_index].max_frequency_point)
		The result can fit to give us qubit AC_tuning_curve. By adjusting ff_to_dc_ratio manually, we can have better coverage (qubit frequency in the scanned range), to give us a good initial estimate of AC_tuning_curve.
		Readout is performed back at the sweet spot.

		The old version could take AC_tuning_curve as well. But I have decided to remove that option, because it should be run with `qubit_freq_vs_fast_flux`, fast version.
		
		Args:
			machine ([type]): [description]
			ff_sweep_abs ([type]): [description]
			qubit_if_sweep ([type]): [description]
			qubit_index ([type]): [description]
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `20E3`)
			ff_to_dc_ratio ([type]): [description] (default: `None`)
			poly_param ([type]): [description] (default: `None`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `3000`)
			final_plot (bool): [description] (default: `True`)
			live_plot (bool): [description] (default: `False`)
			data_process_method (str): [description] (default: `'I'`)
		
		Returns:
			[type]: [description]
		"""

		# set up variables
		ff_sweep = ff_sweep_abs / machine.flux_lines[qubit_index].flux_pulse_amp # relative pulse amp
		channel_index = int(machine.qubits[qubit_index].name[1:])

		if ff_to_dc_ratio is None:
			if poly_param is None:
				poly_param = machine.qubits[qubit_index].AC_tuning_curve
			qubit_freq_est_sweep = np.polyval(poly_param, ff_sweep_abs) * 1E6  # Hz
		else:
			if poly_param is None:
				poly_param = machine.qubits[qubit_index].DC_tuning_curve
			qubit_freq_est_sweep = np.polyval(poly_param, (ff_to_dc_ratio * ff_sweep_abs) + machine.dc_flux[channel_index].max_frequency_point) * 1E6 # Hz

		qubit_freq_est_sweep = np.floor(qubit_freq_est_sweep)

		# Initialize empty vectors to store the global 'I' & 'Q' results
		I_qubit_tot = []
		Q_qubit_tot = []
		qubit_freq_sweep_tot = []

		if live_plot:
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [8, 4]

		config = build_config(machine)

		# start time
		timestamp_created = datetime.datetime.now()
		# 2D scan, RR frequency vs fast flux (ff)
		for ff_index, ff_value in enumerate(ff_sweep):  # sweep over all fast fluxes
			qubit_freq_est = qubit_freq_est_sweep[ff_index]
			qubit_freq_sweep = qubit_freq_est + qubit_if_sweep

			# check if octave LO frequency needs change
			qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
			if qubit_lo - (qubit_freq_est + min(qubit_if_sweep)) > 350E6: # the flux is bringing qubit freq down, need to decrease LO
				qubit_lo = qubit_freq_est + max(qubit_if_sweep) - 350E6
				machine.octaves[0].LO_sources[1].LO_frequency = int(qubit_lo) + 0E6
				machine.qubits[qubit_index].f_01 = int(qubit_freq_est) + 0E6
				machine = self.set_octave.calibration(machine, qubit_index, qubit_only = True) # since RR is changing, calibrate both

			if qubit_lo - (qubit_freq_est + max(qubit_if_sweep)) < -350E6: # the flux is bringing qubit freq up, need to increase LO
				qubit_lo = qubit_freq_est + min(qubit_if_sweep) + 350E6
				machine.octaves[0].LO_sources[1].LO_frequency = int(qubit_lo) + 0E6
				machine.qubits[qubit_index].f_01 = int(qubit_freq_est) + 0E6
				machine = self.set_octave.calibration(machine, qubit_index, qubit_only = True) # since RR is changing, calibrate both

			if live_plot:
				machine, I_tmp, Q_tmp = self.exp1D.qubit_freq(machine, qubit_freq_sweep, qubit_index, ff_amp=ff_value, n_avg=n_avg, cd_time=cd_time, to_simulate=to_simulate, simulation_len=simulation_len, fig=fig)
			else:
				machine, I_tmp, Q_tmp = self.exp1D.qubit_freq(machine, qubit_freq_sweep, qubit_index, ff_amp=ff_value, n_avg=n_avg, cd_time=cd_time, to_simulate=to_simulate, simulation_len=simulation_len)

			I_qubit_tot.append(I_tmp)
			Q_qubit_tot.append(Q_tmp)
			qubit_freq_sweep_tot.append(qubit_freq_sweep)
			progress_counter(ff_index, len(ff_sweep), start_time=datetime.datetime.timestamp(timestamp_created))

		# Experiments finished
		timestamp_finished = datetime.datetime.now()


		# save
		I_qubit = np.array(I_qubit_tot)
		Q_qubit = np.array(Q_qubit_tot)
		qubit_freq_sweep = np.array(qubit_freq_sweep_tot)
		
		# generate xarray dataset for qubit
		expt_dataset = xr.Dataset(
			{
				"I": (["x", "y"], I_qubit),
				"Q": (["x", "y"], Q_qubit),
			},
			coords={
				"Fast_Flux": (["x"], ff_sweep_abs),
				"Qubit_Frequency": (["x", "y"], qubit_freq_sweep),
			},
		)

		expt_name = r'qubit_spec_vs_fast_flux'
		expt_long_name = r'Qubit Spectroscopy vs Fast Flux'
		expt_qubits = [machine.qubits[qubit_index].name]
		expt_TLS = []  # use t0, t1, t2, ...
		expt_sequence = """"""
		expt_extra = {
			'n_ave': str(n_avg),
			'Qubit CD [ns]': str(cd_time),
			'ff_to_dc_ratio': str(ff_to_dc_ratio)
		}
		# save data
		expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
										  expt_long_name, expt_qubits, expt_TLS, expt_sequence, expt_extra)

		# plot qubit spectroscopy vs fast flux
		if final_plot:
			if live_plot is False:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
			plt.cla()
			expt_dataset[data_process_method].plot(x = list(expt_dataset.coords.keys())[0], y = list(expt_dataset.coords.keys())[1], cmap = "seismic")
			plt.title(expt_dataset.attrs['long_name'])
			plt.show()

		return machine, expt_dataset


	def qubit_freq_fast_flux_subroutine(self, machine, ff_sweep_rel, qubit_freq_est_sweep, qubit_if_sweep, qubit_index, n_avg = 1E3, cd_time = 20E3, to_simulate = False, simulation_len = 3000, fig = None):
		"""Run subroutine for 2D qubit freq spectroscopy vs fast flux (`qubit_freq_vs_fast_flux`)

		Input arguments should have been checked: no need to check/change LO.
		
		Args:
			machine ([type]): [description]
			ff_sweep_rel ([type]): [description]
			qubit_freq_est_sweep ([type]): [description]
			qubit_if_sweep ([type]): [description]
			qubit_index ([type]): [description]
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `20E3`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `3000`)
			fig ([type]): [description] (default: `None`)
		
		Returns:
			[type]: [description]
		"""


		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
		qubit_if_est_sweep = np.floor(qubit_freq_est_sweep - qubit_lo)
		qubit_if_est_sweep = qubit_if_est_sweep.astype(int)

		if max(abs(qubit_if_est_sweep)) + max(qubit_if_sweep) > 400E6:
			print("max IF freq > 400 MHz, abort. Check the input!")
			return machine, None, None, None, None

		ff_duration = machine.qubits[qubit_index].pi_length + 40

		# construct qubit freq_sweep_tot
		qubit_freq_sweep_tot = []
		for qubit_freq_i in qubit_freq_est_sweep:
			qubit_freq_sweep_tot.append(qubit_freq_i + qubit_if_sweep)

		with program() as qubit_freq_2D_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			df = declare(int)
			q_freq_est = declare(int) # estimated qubit freq
			da = declare(fixed) # fast flux amplitude

			with for_(n, 0, n < n_avg, n+1):
				with for_each_((da, q_freq_est), (ff_sweep_rel, qubit_if_est_sweep)):
					with for_(*from_array(df,qubit_if_sweep)):
						update_frequency(machine.qubits[qubit_index].name, df + q_freq_est)
						play("const" * amp(da), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
						wait(5, machine.qubits[qubit_index].name)
						play('pi', machine.qubits[qubit_index].name)
						wait(5, machine.qubits[qubit_index].name)
						align(machine.qubits[qubit_index].name, machine.flux_lines[qubit_index].name,
							  machine.resonators[qubit_index].name)
						#wait(4)  # avoid overlap between Z and RO
						readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
						align()
						# eliminate charge accumulation
						play("const" * amp(-1 * da), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
						wait(cd_time * u.ns, machine.resonators[qubit_index].name)
						save(I, I_st)
						save(Q, Q_st)
				save(n, n_st)
			with stream_processing():
				I_st.buffer(len(qubit_if_sweep)).buffer(len(ff_sweep_rel)).average().save("I")
				Q_st.buffer(len(qubit_if_sweep)).buffer(len(ff_sweep_rel)).average().save("Q")
				n_st.save("iteration")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
				# Simulate or execute #
		if to_simulate: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, qubit_freq_2D_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None, None, None
		else:
			qm = self.qmm.open_qm(config)
			job = qm.execute(qubit_freq_2D_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
		    #%matplotlib qt
			if fig is not None:
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				# progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				plt.pause(0.5)

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)

			return machine, qubit_freq_sweep_tot, I, Q, ff_sweep_rel * machine.flux_lines[qubit_index].flux_pulse_amp


	def qubit_freq_vs_fast_flux(self, machine, qubit_freq_sweep, qubit_if_sweep, qubit_index, n_avg = 1E3, cd_time = 20E3, poly_param = None, to_simulate=False, simulation_len=3000, final_plot=True, live_plot=False, data_process_method = 'I'):
		"""Run qubit spectroscopy vs fast flux (2D), fast version: requires an initial AC_tuning_curve.
		
		The whole 2D sweep consists of block-wise 2D scans, calling the `qubit_freq_fast_flux_subroutine`. 
		Each block consists of 4 calls to the subroutine, with 2 LO frequencies (octave calibration only for the qubit). 
		Readout is performed back at the sweet spot.
		With a good AC_tuning_curve, this method is used to run fine scans of the qubit spectroscopy, for identification of avoided crossings
		
		Args:
			machine ([type]): [description]
			qubit_freq_sweep ([type]): [description]
			qubit_if_sweep ([type]): [description]
			qubit_index ([type]): [description]
			n_avg (number): [description] (default: `1E3`)
			cd_time (number): [description] (default: `20E3`)
			poly_param ([type]): [description] (default: `None`)
			to_simulate (bool): [description] (default: `False`)
			simulation_len (number): [description] (default: `3000`)
			final_plot (bool): [description] (default: `True`)
			live_plot (bool): [description] (default: `False`)
			data_process_method (str): [description] (default: `'I'`)
		
		Returns:
			[type]: [description]
		"""


		# set up variables		
		if poly_param is None:
			poly_param = np.array(machine.qubits[qubit_index].AC_tuning_curve[:])

		# sort qubit_freq_sweep from small to large, for later use of searchsorted
		qubit_freq_sweep = np.sort(np.floor(qubit_freq_sweep)) # floor, to avoid larger than max freq situation

		# find the corresponding fast flux values for qubit_freq_sweep
		ff_sweep_abs = []
		for freq_tmp in qubit_freq_sweep:
			sol_tmp = np.roots(poly_param - np.array([0, 0, 0, 0, freq_tmp/1E6]))
			if np.sum(np.isreal(sol_tmp))==4: # four real solutions, take the smaller positive one
				sol_tmp = min(np.real(sol_tmp[sol_tmp>0]))
			else: # two real and two complex solutions, take the smaller positive real value
				sol_tmp = sol_tmp[np.isreal(sol_tmp)]
				sol_tmp = min(np.real(sol_tmp[sol_tmp>0]))
			ff_sweep_abs.append(sol_tmp)
		ff_sweep_abs = np.array(ff_sweep_abs)
		if max(abs(ff_sweep_abs)) > 0.5:
			print("-------------------------------------some fast flux > 0.5V, removed from experiment run")
			qubit_freq_sweep = qubit_freq_sweep[abs(ff_sweep_abs) < 0.5]
			ff_sweep_abs = ff_sweep_abs[abs(ff_sweep_abs) < 0.5]

		ff_sweep_rel = ff_sweep_abs / machine.flux_lines[qubit_index].flux_pulse_amp

		if live_plot:
			fig = plt.figure()
			plt.rcParams['figure.figsize'] = [8, 4]

		# Initialize empty vectors to store the global 'I' & 'Q' results
		I_tot = []
		Q_tot = []
		qubit_freq_sweep_tot = []
		ff_sweep_tot = []

		# start time
		timestamp_created = datetime.datetime.now()

		# divide and conquer!
		# use +- (100 - 300) MHz IF freq range
		# each period is 800 MHz, with two LOs, separated by 200 MHz
		qubit_freq_sweep_head = max(qubit_freq_sweep) # the start of the current analysis
		while qubit_freq_sweep_head + 1 > min(qubit_freq_sweep): # full period (with 2x LOs).
			# find the index of frequencies in qubit_freq_sweep for this sweep, with IF freq > 0
			# LO1
			freq_seg_index_pos_IF_LO1 = [np.searchsorted(qubit_freq_sweep, qubit_freq_sweep_head - 200E6, side='right'),
										 np.searchsorted(
											 qubit_freq_sweep, qubit_freq_sweep_head,
											 side='right')]  # -1 s.t. head will be included. sweep range is IF = (100, 300] MHz
			freq_seg_index_neg_IF_LO1 = [np.searchsorted(qubit_freq_sweep, qubit_freq_sweep_head - 600E6, side='right'),
										 np.searchsorted(
											 qubit_freq_sweep, qubit_freq_sweep_head - 400E6,
											 side='right')]  # -1 s.t. head will be included. sweep range is IF = (-300, -100] MHz
			# LO2
			freq_seg_index_pos_IF_LO2 = [np.searchsorted(qubit_freq_sweep, qubit_freq_sweep_head - 400E6, side='right'),
										 np.searchsorted(
											 qubit_freq_sweep, qubit_freq_sweep_head - 200E6,
											 side='right')]  # -1 s.t. head will be included. sweep range is IF = (100, 300] MHz
			freq_seg_index_neg_IF_LO2 = [np.searchsorted(qubit_freq_sweep, qubit_freq_sweep_head - 800E6, side='right'),
										 np.searchsorted(
											 qubit_freq_sweep, qubit_freq_sweep_head - 600E6,
											 side='right')]  # -1 s.t. head will be included. sweep range is IF = (-300, -100] MHz

			# LO1
			qubit_lo = qubit_freq_sweep_head - 300E6
			machine.octaves[0].LO_sources[1].LO_frequency = int(qubit_lo) + 0E6
			machine.qubits[qubit_index].f_01 = int(qubit_lo) + 200E6  # calibrate in the center of the +ive sweep range
			machine = self.set_octave.calibration(machine, qubit_index, qubit_only = True)
			# qubit freq vs fast flux, sweep over +ive and -ive IF freq
			for freq_seg_index in [freq_seg_index_pos_IF_LO1,freq_seg_index_neg_IF_LO1]:
				ff_sweep_rel_seg = ff_sweep_rel[freq_seg_index[0]:freq_seg_index[1]]
				qubit_freq_sweep_seg = qubit_freq_sweep[freq_seg_index[0]:freq_seg_index[1]]

				if len(ff_sweep_rel_seg) > 0: # still need to sweep
					if live_plot:
						machine, qubit_freq_sweep_tmp, I_tmp, Q_tmp, ff_sweep_tmp = self.qubit_freq_fast_flux_subroutine(machine, ff_sweep_rel_seg, qubit_freq_sweep_seg, qubit_if_sweep, qubit_index, n_avg=n_avg, cd_time=cd_time, fig=fig)
					else:
						machine, qubit_freq_sweep_tmp, I_tmp, Q_tmp, ff_sweep_tmp = self.qubit_freq_fast_flux_subroutine(machine, ff_sweep_rel_seg, qubit_freq_sweep_seg, qubit_if_sweep, qubit_index, n_avg=n_avg, cd_time=cd_time)

					I_tot.append(I_tmp)
					Q_tot.append(Q_tmp)
					qubit_freq_sweep_tot.append(qubit_freq_sweep_tmp)
					ff_sweep_tot.append(ff_sweep_tmp)
				else: # no need to scan. For the last period
					pass

			# LO2
			qubit_lo = qubit_freq_sweep_head - 500E6
			machine.octaves[0].LO_sources[1].LO_frequency = int(qubit_lo) + 0E6
			machine.qubits[qubit_index].f_01 = int(qubit_lo) + 200E6  # calibrate in the center of the +ive sweep range
			machine = self.set_octave.calibration(machine, qubit_index, qubit_only = True)
			# qubit freq vs fast flux, sweep over +ive and -ive IF freq
			for freq_seg_index in [freq_seg_index_pos_IF_LO2, freq_seg_index_neg_IF_LO2]:
				ff_sweep_rel_seg = ff_sweep_rel[freq_seg_index[0]:freq_seg_index[1]]
				qubit_freq_sweep_seg = qubit_freq_sweep[freq_seg_index[0]:freq_seg_index[1]]

				if len(ff_sweep_rel_seg) > 0:  # still need to sweep
					if live_plot:
						machine, qubit_freq_sweep_tmp, I_tmp, Q_tmp, ff_sweep_tmp = self.qubit_freq_fast_flux_subroutine(machine, ff_sweep_rel_seg, qubit_freq_sweep_seg, qubit_if_sweep, qubit_index, n_avg=n_avg, cd_time=cd_time, fig=fig)
					else:
						machine, qubit_freq_sweep_tmp, I_tmp, Q_tmp, ff_sweep_tmp = self.qubit_freq_fast_flux_subroutine(machine, ff_sweep_rel_seg, qubit_freq_sweep_seg, qubit_if_sweep, qubit_index, n_avg=n_avg, cd_time=cd_time)

					I_tot.append(I_tmp)
					Q_tot.append(Q_tmp)
					qubit_freq_sweep_tot.append(qubit_freq_sweep_tmp)
					ff_sweep_tot.append(ff_sweep_tmp)
				else: # no need to scan. For the last period
					pass

			qubit_freq_sweep_head -= 800E6 # move to the next full period

		# Experiments finished
		timestamp_finished = datetime.datetime.now()

		# save
		I_qubit = np.concatenate(I_tot)
		Q_qubit = np.concatenate(Q_tot)
		qubit_freq_sweep = np.concatenate(qubit_freq_sweep_tot)
		ff_sweep_abs = np.concatenate(ff_sweep_tot)
		
		# sort according to fast flux sweep
		qubit_freq_sweep = qubit_freq_sweep.reshape(np.size(ff_sweep_abs), np.size(qubit_freq_sweep) // np.size(ff_sweep_abs))
		I_qubit = I_qubit.reshape(np.size(ff_sweep_abs), np.size(I_qubit) // np.size(ff_sweep_abs))
		Q_qubit = Q_qubit.reshape(np.size(ff_sweep_abs), np.size(Q_qubit) // np.size(ff_sweep_abs))

		sort_index = np.argsort(ff_sweep_abs)
		qubit_freq_sweep = qubit_freq_sweep[sort_index,:]
		I_qubit = I_qubit[sort_index,:]
		Q_qubit = Q_qubit[sort_index,:]
		ff_sweep_abs = ff_sweep_abs[sort_index]

		# generate xarray dataset for qubit
		expt_dataset = xr.Dataset(
			{
				"I": (["x", "y"], I_qubit),
				"Q": (["x", "y"], Q_qubit),
			},
			coords={
				"Fast_Flux": (["x"], ff_sweep_abs),
				"Qubit_Frequency": (["x", "y"], qubit_freq_sweep),
			},
		)

		expt_name = r'qubit_spec_vs_fast_flux'
		expt_long_name = r'Qubit Spectroscopy vs Fast Flux'
		expt_qubits = [machine.qubits[qubit_index].name]
		expt_TLS = []  # use t0, t1, t2, ...
		expt_sequence = """"""
		expt_extra = {
			'n_ave': str(n_avg),
			'Qubit CD [ns]': str(cd_time)
		}

		# save data
		expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
										  expt_long_name, expt_qubits, expt_TLS, expt_sequence, expt_extra)

		# plot qubit spectroscopy vs fast flux
		if final_plot:
			if live_plot is False:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
			plt.cla()
			expt_dataset[data_process_method].plot(x = list(expt_dataset.coords.keys())[0], y = list(expt_dataset.coords.keys())[1], cmap = "seismic")
			plt.title(expt_dataset.attrs['long_name'])
			plt.show()

		return machine, expt_dataset

	def qubit_freq_time(self, machine, qubit_freq_sweep, qubit_index, N_rounds = 1, pi_amp_rel = 1.0, ff_amp = 0.0, n_avg = 1E3, cd_time = 10E3, to_simulate = False, simulation_len = 3000, final_plot=True, live_plot=False, data_process_method = 'I'):
		"""
		Qubit spectroscopy experiment in 1D as a function of time

		Args:
			machine ([type]): 1D array of qubit frequency sweep
			qubit_freq_sweep ([type]): [description]
			qubit_index ([type]): [description]
			pi_amp_rel (number): [number of times the spectrosocpy is run - this will yield a final time] (default: `1.0`)
			pi_amp_rel (number): [description] (default: `1.0`)
			ff_amp (number): Fast flux amplitude that overlaps with the Rabi pulse. The ff pulse is 40ns longer than Rabi pulse, and share the pulse midpoint. (default: `0.0`)
			n_avg (number): Repetitions of the experiments (default: `1E3`)
			cd_time (number): Cooldown time between subsequent experiments (default: `10E3`)
			to_simulate (bool): True: run simulation; False: run experiment (default: `False`)
			simulation_len (number): Length of the sequence to simulate. In clock cycles (4ns) (default: `1000`)
			final_plot (bool): [description] (default: `True`)
			live_plot (bool): [description] (default: `False`)
			data_process_method (str): [description] (default: `'I'`)

		Returns:
			machine
			expt_dataset
		"""

		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
		qubit_if_sweep = qubit_freq_sweep - qubit_lo
		qubit_if_sweep = np.floor(qubit_if_sweep)
		ff_duration = machine.qubits[qubit_index].pi_length + 40

		# Initialize empty vectors to store the global 'I', 'Q', and 'timestamp' results
		I_tot = []
		Q_tot = []
		timestamp = []

		if np.max(abs(qubit_if_sweep)) > 400E6: # check if parameters are within hardware limit
			print("qubit if range > 400MHz")
			return machine, None

		start_time = time.time()
		timestamp.append(0)
		t = datetime.datetime.now()
		timestamp_created = datetime.datetime.now()

		for N_index in np.arange(N_rounds):
			progress_counter(N_index, N_rounds, start_time=start_time)

			with program() as qubit_freq_prog:
				[I,Q,n,I_st,Q_st,n_st] = declare_vars()
				df = declare(int)

				with for_(n, 0, n < n_avg, n+1):
					with for_(*from_array(df,qubit_if_sweep)):
						update_frequency(machine.qubits[qubit_index].name, df)
						play("const" * amp(ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
						wait(5, machine.qubits[qubit_index].name)
						play('pi'*amp(pi_amp_rel), machine.qubits[qubit_index].name)
						wait(5, machine.qubits[qubit_index].name)
						align(machine.qubits[qubit_index].name, machine.flux_lines[qubit_index].name,
							  machine.resonators[qubit_index].name)
						readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
						align()
						# eliminate charge accumulation
						play("const" * amp(-1 * ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
						wait(cd_time * u.ns, machine.flux_lines[qubit_index].name)
						save(I, I_st)
						save(Q, Q_st)
					save(n, n_st)
				with stream_processing():
					n_st.save('iteration')
					I_st.buffer(len(qubit_if_sweep)).average().save("I")
					Q_st.buffer(len(qubit_if_sweep)).average().save("Q")

			#####################################
			#  Open Communication with the QOP  #
			#####################################
			config = build_config(machine)
			# Simulate or execute #
			if to_simulate:  # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
				simulation_config = SimulationConfig(duration=simulation_len)
				job = self.qmm.simulate(config, qubit_freq_prog, simulation_config)
				job.get_simulated_samples().con1.plot()
				return machine, None
			else:
				qm = self.qmm.open_qm(config)
				job = qm.execute(qubit_freq_prog)
				# Get results from QUA program
				results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
				# Live plotting
				if live_plot:
					fig = plt.figure()
					plt.rcParams['figure.figsize'] = [8, 4]
					interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

					while results.is_processing():
						# Fetch results
						I_tmp, Q_tmp, iteration = results.fetch_all()
						I_tmp = u.demod2volts(I_tmp, machine.resonators[qubit_index].readout_pulse_length)
						Q_tmp = u.demod2volts(Q_tmp, machine.resonators[qubit_index].readout_pulse_length)

						# Update the live plot!
						plt.cla()
						plt.title("Qubit Spectroscopy")
						if data_process_method == 'Phase':
							plt.plot((qubit_freq_sweep) / u.MHz, np.unwrap(np.angle(I_tmp + 1j * Q_tmp)), ".")
							plt.xlabel("Qubit Frequency [MHz]")
							plt.ylabel("Signal Phase [rad]")
						elif data_process_method == 'Amplitude':
							plt.plot((qubit_freq_sweep) / u.MHz, np.abs(I_tmp + 1j * Q_tmp), ".")
							plt.xlabel("Qubit Frequency [MHz]")
							plt.ylabel("Signal Amplitude [V]")
						elif data_process_method == 'I':
							plt.plot((qubit_freq_sweep) / u.MHz, I_tmp, ".")
							plt.xlabel("Qubit Frequency [MHz]")
							plt.ylabel("Signal I Quadrature [V]")
						plt.pause(0.5)
				else:
					while results.is_processing():
						# Fetch results
						I_tmp, Q_tmp, iteration = results.fetch_all()

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)

			elapsed = datetime.datetime.now() - t
			timestamp.append(elapsed.total_seconds())
			I_tot.append(I)
			Q_tot.append(Q)

		timestamp_finished = datetime.datetime.now()
		# save
		I_final = np.concatenate([I_tot])
		Q_final = np.concatenate([Q_tot])
		tau = np.concatenate([timestamp])
		# generate xarray dataset
		expt_dataset = xr.Dataset(
			{
				"I": (["x", "y"], I_final),
				"Q": (["x", "y"], Q_final),
			},
			coords={
				"Elapsed": (["x"], tau[1:]),
				"Qubit_Frequency": (["y"], qubit_freq_sweep)})

		expt_name = 'time_spec'
		expt_long_name = 'Qubit Time Spectroscopy'
		expt_qubits = [machine.qubits[qubit_index].name]
		expt_TLS = []  # use t0, t1, t2, ...
		expt_sequence = """with for_(n, 0, n < n_avg, n+1):
		with for_(*from_array(df,qubit_if_sweep)):
			update_frequency(machine.qubits[qubit_index].name, df)
			play("const" * amp(ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
			wait(5, machine.qubits[qubit_index].name)
			play('pi'*amp(pi_amp_rel), machine.qubits[qubit_index].name)
			wait(5, machine.qubits[qubit_index].name)
			align(machine.qubits[qubit_index].name, machine.flux_lines[qubit_index].name,
				  machine.resonators[qubit_index].name)
			readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
			align()
			# eliminate charge accumulation
			play("const" * amp(-1 * ff_amp), machine.flux_lines[qubit_index].name, duration=ff_duration * u.ns)
			wait(cd_time * u.ns, machine.resonators[qubit_index].name)
			save(I, I_st)
			save(Q, Q_st)
		save(n, n_st)"""
		expt_extra = {
			'N rounds': str(N_rounds),
			'pi_amp_rel': str(pi_amp_rel),
			'ff amp [V]': str(ff_amp),
			'n_ave': str(n_avg),
			'Qubit CD [ns]': str(cd_time)
		}

		# save data
		expt_dataset = self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name,
										  expt_long_name, expt_qubits, expt_TLS, expt_sequence, expt_extra)


		if final_plot:
			if live_plot is False:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
			plt.cla()
			expt_dataset[data_process_method].plot(x=list(expt_dataset.coords.keys())[0],
												   y=list(expt_dataset.coords.keys())[1], cmap="seismic")
			plt.title(expt_dataset.attrs['long_name'])
			plt.show()


		return machine, expt_dataset

