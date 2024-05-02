class EH_SWAP:
	"""
	class in ExperimentHandle, for SWAP sequence related 1D experiments
	Methods:
		update_tPath
		update_str_datetime
		qubit_freq(self, qubit_freq_sweep, qubit_index, n_avg, cd_time, ff_amp = 1.0, simulate_flag = False, simulation_len = 1000)
	"""

	def __init__(self, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs):
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs

	def rabi_SWAP(self, machine, rabi_duration_sweep, qubit_index, TLS_index, pi_amp_rel = 1.0, n_avg = 1E3, cd_time = 10E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		1D experiment: qubit rabi (length sweep) - SWAP - measure
		qubit rabi duration in clock cycle

		Args:
			machine
			rabi_duration_sweep (): in clock cycle!
			qubit_index ():
			TLS_index ():
			pi_amp_rel ():
			n_avg ():
			cd_time ():
			simulate_flag ():
			simulation_len ():
			plot_flag ():

		Returns:
			machine
			rabi_duration_sweep * 4: in ns
			sig_amp
		"""
		

		if min(rabi_duration_sweep) < 4:
			print("some rabi lengths shorter than 4 clock cycles, removed from run")
			rabi_duration_sweep = rabi_duration_sweep[rabi_duration_sweep>3]

		rabi_duration_sweep = rabi_duration_sweep.astype(int)

		# fLux pulse baking for SWAP
		swap_length = machine.flux_lines[qubit_index].iswap.length[TLS_index]
		swap_amp = machine.flux_lines[qubit_index].iswap.level[TLS_index]
		flux_waveform = np.array([swap_amp] * swap_length)

		def baked_swap_waveform(waveform):
			pulse_segments = []  # Stores the baking objects
			# Create the different baked sequences, each one corresponding to a different truncated duration
			with baking(config, padding_method="right") as b:
				b.add_op("flux_pulse", machine.flux_lines[qubit_index].name, waveform.tolist())
				b.play("flux_pulse", machine.flux_lines[qubit_index].name)
				pulse_segments.append(b)
			return pulse_segments
		square_TLS_swap = baked_swap_waveform(flux_waveform)

		with program() as time_rabi:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, rabi_duration_sweep)):
					wait(5, machine.qubits[qubit_index].name)
					play("pi" * amp(pi_amp_rel), machine.qubits[qubit_index].name, duration=t)
					wait(5, machine.qubits[qubit_index].name)
					align()
					square_TLS_swap[0].run()
					align()
					readout_avg_macro(machine.resonators[qubit_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					align()
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(rabi_duration_sweep)).average().save("I")
				Q_st.buffer(len(rabi_duration_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")

		if simulate_flag:
			simulation_config = SimulationConfig(duration=simulation_len)  # in clock cycles
			job = qmm.simulate(config, time_rabi, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(time_rabi)
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

			# Live plotting
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				sig_amp = np.sqrt(I ** 2 + Q ** 2)
				sig_phase = np.angle(I + 1j * Q)
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				if plot_flag == True:
					plt.cla()
					plt.title("Time Rabi")
					plt.plot(rabi_duration_sweep * 4, sig_amp, "b.")
					plt.xlabel("tau [ns]")
					plt.ylabel("Signal Amplitude [V]")
					plt.pause(0.01)

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
			sig_amp = np.sqrt(I ** 2 + Q ** 2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'time_rabi'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"Q_rabi_duration": rabi_duration_sweep * 4, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset

	def swap_coarse(self,machine, tau_sweep_abs, qubit_index, TLS_index, n_avg, cd_time, simulate_flag=False, simulation_len=1000, plot_flag=True):
		"""
		1D SWAP spectroscopy. qubit pi - SWAP (sweep Z duration) - measure
		tau_sweep in ns, only takes multiples of 4ns

		Args:
			machine
			tau_sweep_abs ():
			qubit_index ():
			TLS_index ():
			n_avg ():
			cd_time ():
			simulate_flag ():
			simulation_len ():
			plot_flag ():

		Returns:
			machine
			tau_sweep_abs
			sig_amp
		"""

		

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
					play("pi", machine.qubits[qubit_index].name)
					align()
					play("iswap", machine.flux_lines[qubit_index].name, duration=t)
					align()
					readout_avg_macro(machine.resonators[qubit_index].name,I,Q)
					align()
					wait(50)
					play("iswap" * amp(-1), machine.flux_lines[qubit_index].name, duration=t)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				# for the progress counter
				n_st.save("iteration")
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")

		if simulate_flag:
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, iswap, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(iswap)
			results = fetching_tool(job, ["I", "Q", "iteration"], mode="live")

			if plot_flag:
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
			exp_name = 'SWAP1D'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"ff_amp": machine.flux_lines[qubit_index].iswap.level[TLS_index], "sig_amp": sig_amp, "sig_phase": sig_phase,
					 "tau_sweep": tau_sweep_abs})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			if plot_flag:
				plt.cla()
				plt.plot(tau_sweep_abs, sig_amp)
				plt.ylabel("Signal Amplitude (V)")
				plt.xlabel("interaction time (ns)")

		return machine, expt_dataset

	def SWAP_rabi(self, machine, rabi_duration_sweep, qubit_index, TLS_index, pi_amp_rel = 1.0, n_avg = 1E3, cd_time = 10E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		1D experiment for debug: SWAP - qubit rabi (sweep duration) - measure

		Args:
			machine
			rabi_duration_sweep (): in clock cycle
			qubit_index ():
			TLS_index ():
			pi_amp_rel ():
			n_avg ():
			cd_time ():
			simulate_flag ():
			simulation_len ():
			plot_flag ():
			
		Returns:
			machine
			rabi_duration_sweep * 4
			sig_amp
		"""

		

		if min(rabi_duration_sweep) < 4:
			print("some rabi lengths shorter than 4 clock cycles, removed from run")
			rabi_duration_sweep = rabi_duration_sweep[rabi_duration_sweep>3]
		
		rabi_duration_sweep = rabi_duration_sweep.astype(int)

		# fLux pulse baking for SWAP
		swap_length = machine.flux_lines[qubit_index].iswap.length[TLS_index]
		swap_amp = machine.flux_lines[qubit_index].iswap.level[TLS_index]
		flux_waveform = np.array([swap_amp] * swap_length)

		def baked_swap_waveform(waveform):
			pulse_segments = []  # Stores the baking objects
			# Create the different baked sequences, each one corresponding to a different truncated duration
			with baking(config, padding_method="right") as b:
				b.add_op("flux_pulse", machine.flux_lines[qubit_index].name, waveform.tolist())
				b.play("flux_pulse", machine.flux_lines[qubit_index].name)
				pulse_segments.append(b)
			return pulse_segments
		square_TLS_swap = baked_swap_waveform(flux_waveform)

		with program() as time_rabi:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, rabi_duration_sweep)):
					square_TLS_swap[0].run()
					align()
					wait(5, machine.qubits[qubit_index].name)
					play("pi" * amp(pi_amp_rel), machine.qubits[qubit_index].name, duration=t)
					wait(5, machine.qubits[qubit_index].name)
					align()
					readout_avg_macro(machine.resonators[qubit_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					align()
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(rabi_duration_sweep)).average().save("I")
				Q_st.buffer(len(rabi_duration_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")

		if simulate_flag:
			simulation_config = SimulationConfig(duration=simulation_len)  # in clock cycles
			job = qmm.simulate(config, time_rabi, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(time_rabi)
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

			# Live plotting
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				sig_amp = np.sqrt(I ** 2 + Q ** 2)
				sig_phase = np.angle(I + 1j * Q)
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				if plot_flag == True:
					plt.cla()
					plt.title("Time Rabi")
					plt.plot(rabi_duration_sweep * 4, sig_amp, "b.")
					plt.xlabel("tau [ns]")
					plt.ylabel("Signal Amplitude [V]")
					plt.pause(0.01)

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
			sig_amp = np.sqrt(I ** 2 + Q ** 2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'time_rabi'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"Q_rabi_duration": rabi_duration_sweep * 4, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset

	def rabi_SWAP2(self, machine rabi_duration_sweep, qubit_index, TLS_index, pi_amp_rel = 1.0, n_avg = 1E3, cd_time = 10E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		1D experiment: qubit rabi (sweep duration) - SWAP - SWAP, to see if the state comes back

		Args:
			machine
			rabi_duration_sweep (): in clock cycle
			qubit_index ():
			TLS_index ():
			pi_amp_rel ():
			n_avg ():
			cd_time ():
			simulate_flag ():
			simulation_len ():
			plot_flag ():
			
		Returns:
			machine
			rabi_duration_sweep * 4
			sig_amp
		"""
		

		if min(rabi_duration_sweep) < 4:
			print("some rabi lengths shorter than 4 clock cycles, removed from run")
			rabi_duration_sweep = rabi_duration_sweep[rabi_duration_sweep>3]

		rabi_duration_sweep = rabi_duration_sweep.astype(int)

		# fLux pulse baking for SWAP
		swap_length = machine.flux_lines[qubit_index].iswap.length[TLS_index]
		swap_amp = machine.flux_lines[qubit_index].iswap.level[TLS_index]
		flux_waveform = np.array([swap_amp] * swap_length)

		def baked_swap_waveform(waveform):
			pulse_segments = []  # Stores the baking objects
			# Create the different baked sequences, each one corresponding to a different truncated duration
			with baking(config, padding_method="right") as b:
				b.add_op("flux_pulse", machine.flux_lines[qubit_index].name, waveform.tolist())
				b.play("flux_pulse", machine.flux_lines[qubit_index].name)
				pulse_segments.append(b)
			return pulse_segments
		square_TLS_swap = baked_swap_waveform(flux_waveform)

		with program() as time_rabi:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, rabi_duration_sweep)):
					wait(5, machine.qubits[qubit_index].name)
					play("pi" * amp(pi_amp_rel), machine.qubits[qubit_index].name, duration=t)
					wait(5, machine.qubits[qubit_index].name)
					align()
					square_TLS_swap[0].run()
					wait(5)
					square_TLS_swap[0].run()
					align()
					readout_avg_macro(machine.resonators[qubit_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					wait(5)
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[qubit_index].name, -1)])
					align()
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(rabi_duration_sweep)).average().save("I")
				Q_st.buffer(len(rabi_duration_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")

		if simulate_flag:
			simulation_config = SimulationConfig(duration=simulation_len)  # in clock cycles
			job = qmm.simulate(config, time_rabi, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(time_rabi)
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

			# Live plotting
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				sig_amp = np.sqrt(I ** 2 + Q ** 2)
				sig_phase = np.angle(I + 1j * Q)
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				if plot_flag == True:
					plt.cla()
					plt.title("Time Rabi")
					plt.plot(rabi_duration_sweep * 4, sig_amp, "b.")
					plt.xlabel("tau [ns]")
					plt.ylabel("Signal Amplitude [V]")
					plt.pause(0.01)

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
			sig_amp = np.sqrt(I ** 2 + Q ** 2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'time_rabi'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"Q_rabi_duration": rabi_duration_sweep * 4, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset
