import datetime
from configuration import datetime_format_string

class EH_SWAP:
	"""
	class in ExperimentHandle, for SWAP related 2D experiments
	Methods:
		update_tPath
		update_str_datetime
	"""

	def __init__(self, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs, ref_to_qmm):
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs
		self.qmm = ref_to_qmm

	def swap_coarse(self, machine, tau_sweep_abs, ff_sweep_abs, qubit_index, n_avg = 1E3, cd_time = 20E3, to_simulate=False, simulation_len=3000, final_plot=True):
		"""
		runs 2D SWAP spectroscopy experiment
		Note time resolution is 4ns!

		Args:
			machine
			tau_sweep_abs (): interaction time sweep, in ns. Will be regulated to multiples of 4ns, starting from 16ns
			ff_sweep_abs (): fast flux sweep, in V
			qubit_index ():
			n_avg ():
			cd_time ():
			to_simulate ():
			simulation_len ():
			final_plot ():
			
		Returns:
			machine
			ff_sweep_abs
			tau_sweep_abs
			sig_amp

		"""
		
		ff_sweep_rel = ff_sweep_abs / machine.flux_lines[qubit_index].flux_pulse_amp
		tau_sweep_cc = tau_sweep_abs // 4  # in clock cycles
		tau_sweep_cc = np.unique(tau_sweep_cc)
		tau_sweep = tau_sweep_cc.astype(int)  # clock cycles
		tau_sweep_abs = tau_sweep * 4  # time in ns

		with program() as iswap:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)
			da = declare(fixed)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, tau_sweep)):
					with for_(*from_array(da, ff_sweep_rel)):
						play("pi", machine.qubits[qubit_index].name)
						align()
						play("const" * amp(da), machine.flux_lines[qubit_index].name, duration=t)
						align()
						readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
						align()
						wait(50)
						play("const" * amp(-da), machine.flux_lines[qubit_index].name, duration=t)
						save(I, I_st)
						save(Q, Q_st)
						wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				# for the progress counter
				n_st.save("iteration")
				I_st.buffer(len(ff_sweep_rel)).buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(ff_sweep_rel)).buffer(len(tau_sweep)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, iswap, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(iswap)
			results = fetching_tool(job, ["I", "Q", "iteration"], mode="live")

			if final_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8,4]
				interrupt_on_close(fig, job)

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				# progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				time.sleep(0.1)

				if final_plot:
					plt.cla()
					plt.pcolor(ff_sweep_abs, tau_sweep_abs, np.sqrt(I**2 + Q**2), cmap="seismic")
					plt.colorbar()
					plt.title("SWAP Spectroscopy")
					plt.xlabel("Fast Flux(V)")
					plt.ylabel("Interaction Time (ns)")

			# fetch all data after live-updating
			timestamp_finished = datetime.datetime.now()
			I, Q, _ = results.fetch_all()
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
			# sig_amp = np.sqrt(I ** 2 + Q ** 2)
			# sig_phase = np.angle(I + 1j * Q)

			# generate xarray dataset
			expt_dataset = xr.Dataset(
			    {
			        "I": (["y", "x"], I),
			        "Q": (["y", "x"], Q),
			    },
			    coords={
			        "Fast_Flux": (["x"], ff_sweep_abs),
			        "Time": (["y"], tau_sweep_abs),
			    },
			)
			
			expt_name = 'SWAP2D'
			expt_long_name = 'SWAP Spectroscopy'
			expt_qubits = [machine.qubits[qubit_index].name]
			expt_TLS = [] # use t0, t1, t2, ...
			expt_sequence = """with for_(n, 0, n < n_avg, n + 1):
	with for_(*from_array(t, tau_sweep)):
		with for_(*from_array(da, ff_sweep_rel)):
			play("pi", machine.qubits[qubit_index].name)
			align()
			play("const" * amp(da), machine.flux_lines[qubit_index].name, duration=t)
			align()
			readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
			align()
			wait(50)
			play("const" * amp(-da), machine.flux_lines[qubit_index].name, duration=t)
			save(I, I_st)
			save(Q, Q_st)
			wait(cd_time * u.ns, machine.resonators[qubit_index].name)
	save(n, n_st)"""

			# save data
			self.datalogs.save(expt_dataset, machine, timestamp_created, timestamp_finished, expt_name, expt_long_name, expt_qubits, expt_TLS, expt_sequence)

			if final_plot:
				plt.cla()
				sig_amp = np.sqrt(expt_dataset.I**2 + expt_dataset.Q**2)
				sig_amp.plot()

		return machine, expt_dataset

	def swap_fine(self, machine, tau_sweep_abs, ff_sweep_abs, qubit_index, n_avg = 1E3, cd_time = 20E3, to_simulate=False, simulation_len=3000, final_plot=True):
		"""
		runs 2D SWAP spectroscopy
		allows 1ns time resolution, and start at t < 16ns
		Args:
			machine
			tau_sweep_abs (): interaction time sweep, in ns.
			ff_sweep_abs (): fast flux sweep in V
			qubit_index ():
			n_avg ():
			cd_time ():
			to_simulate ():
			simulation_len ():
			final_plot ():
			
		Returns:
			machine
			ff_sweep_abs: 1D array of fast flux amplitude in V
			tau_sweep_abs: 1D array of tau in ns
			sig_amp.T: such that x is flux, y is time

		"""
		

		ff_sweep_rel = ff_sweep_abs / machine.flux_lines[qubit_index].flux_pulse_amp
		tau_sweep = tau_sweep_abs.astype(int)  # clock cycles

		# set up variables
		max_pulse_duration = max(tau_sweep_abs)
		max_pulse_duration = int(max_pulse_duration)
		min_pulse_duration = min(tau_sweep_abs)
		min_pulse_duration = int(min_pulse_duration)
		dt_pulse_duration = tau_sweep_abs[2] - tau_sweep_abs[1]
		dt_pulse_duration = int(dt_pulse_duration)

		# fLux pulse waveform generation, 250 ns/points
		flux_waveform = np.array([machine.flux_lines[qubit_index].flux_pulse_amp] * max_pulse_duration)

		def baked_ff_waveform(waveform, pulse_duration):
			pulse_segments = []  # Stores the baking objects
			# Create the different baked sequences, each one corresponding to a different truncated duration
			for i in range(0, pulse_duration + 1):
				with baking(config, padding_method="right") as b:
					if i == 0:  # Otherwise, the baking will be empty and will not be created
						wf = [0.0] * 16
					else:
						wf = waveform[:i].tolist()
					b.add_op("flux_pulse", machine.flux_lines[qubit_index].name, wf)
					b.play("flux_pulse", machine.flux_lines[qubit_index].name)
				# Append the baking object in the list to call it from the QUA program
				pulse_segments.append(b)
			return pulse_segments

		# Baked flux pulse segments
		square_pulse_segments = baked_ff_waveform(flux_waveform, max_pulse_duration)

		with program() as iswap:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)
			da = declare(fixed)
			segment = declare(int)  # Flux pulse segment

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(da, ff_sweep_rel)):
					with for_(segment, min_pulse_duration, segment <= max_pulse_duration, segment + dt_pulse_duration):
						play("pi", machine.qubits[qubit_index].name)
						align()
						with switch_(segment):
							for j in range(min_pulse_duration,max_pulse_duration+1,dt_pulse_duration):
								with case_(j):
									square_pulse_segments[j].run(amp_array=[(machine.flux_lines[qubit_index].name, da)])
						align()
						readout_rotated_macro(machine.resonators[qubit_index].name,I,Q)
						save(I, I_st)
						save(Q, Q_st)
						align()
						wait(cd_time * u.ns, machine.resonators[qubit_index].name)
						align()
						with switch_(segment):
							for j in range(min_pulse_duration,max_pulse_duration+1,dt_pulse_duration):
								with case_(j):
									square_pulse_segments[j].run(amp_array=[(machine.flux_lines[qubit_index].name, -da)])
						wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				# for the progress counter
				n_st.save("iteration")
				I_st.buffer(len(tau_sweep)).buffer(len(ff_sweep_rel)).average().save("I")
				Q_st.buffer(len(tau_sweep)).buffer(len(ff_sweep_rel)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		
		if to_simulate:
			simulation_config = SimulationConfig(duration = simulation_len)
			job = self.qmm.simulate(config, iswap, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = self.qmm.open_qm(config)
			timestamp_created = datetime.datetime.now()
			job = qm.execute(iswap)
			results = fetching_tool(job, ["I", "Q", "iteration"], mode="live")

			if final_plot:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8,4]
				interrupt_on_close(fig, job)

			while results.is_processing():
				I, Q, iteration = results.fetch_all()
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				time.sleep(0.1)

			I, Q, _ = results.fetch_all()
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
			sig_amp = np.sqrt(I ** 2 + Q ** 2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'SWAP'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"ff_sweep": ff_sweep_abs, "sig_amp": sig_amp.T, "sig_phase": sig_phase.T,
					 "tau_sweep": tau_sweep_abs})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			if final_plot:
				plt.cla()
				plt.pcolor(ff_sweep_abs, tau_sweep_abs, sig_amp.T, cmap="seismic")
				plt.colorbar()
				plt.xlabel("fast flux amp (V)")
				plt.ylabel("interaction time (ns)")

		return machine, expt_dataset
