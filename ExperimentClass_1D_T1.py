class EH_T1:
	def __init__(self, ref_to_update_tPath, ref_to_update_str_datetime, ref_to_set_octave):
		self.update_tPath = ref_to_update_tPath
		self.update_str_datetime = ref_to_update_str_datetime
		self.set_octave = ref_to_set_octave

	def qubit_T1(self, machine, tau_sweep_abs, qubit_index, res_index, n_avg = 1E3, cd_time = 10E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		runs qubit T1. Designed to be at fixed 0 fast flux.

		Args:
			machine
			tau_sweep_abs (): in ns. Will be regulated to integer clock cycles
			qubit_index ():
			res_index ():
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
		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		tau_sweep_cc = tau_sweep_abs//4 # in clock cycles
		tau_sweep_cc = np.unique(tau_sweep_cc)
		tau_sweep = tau_sweep_cc.astype(int) # clock cycles
		tau_sweep_abs = tau_sweep * 4 # time in ns

		with program() as t1_prog:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			tau = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(tau, tau_sweep)):
					play("pi", machine.qubits[qubit_index].name)
					wait(tau, machine.qubits[qubit_index].name)
					align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[res_index].name)
				save(n, n_st)
			with stream_processing():
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port='9510', octave=octave_config, log_level = "ERROR")

		# Simulate or execute #
		if simulate_flag:
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, t1_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(t1_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
			if plot_flag is True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [12, 8]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure    while results.is_processing():
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
					plt.title("T1")
					plt.plot(tau_sweep_abs, sig_amp, "b.")
					plt.xlabel("tau [ns]")
					plt.ylabel("Signal Amplitude [V]")
					plt.pause(0.01)

		# save data
		exp_name = 'T1'
		qubit_name = 'Q' + str(qubit_index + 1)
		f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
		file_name = f_str + '.mat'
		json_name = f_str + '_state.json'
		savemat(os.path.join(tPath, file_name),
				{"Q_tau": tau_sweep_abs, "sig_amp": sig_amp, "sig_phase": sig_phase})
		machine._save(os.path.join(tPath, json_name), flat_data=False)

		return machine, tau_sweep_abs, sig_amp

	def TLS_T1(self, machine, tau_sweep_abs, qubit_index, res_index, flux_index, TLS_index, n_avg = 1E3, cd_time_qubit = 10E3, cd_time_TLS = None, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		TLS T1 using SWAP with the transmon to prepare the excited state
		sequence is qubit pi - SWAP - wait - SWAP - qubit readout
		
		:param machine
		:param tau_sweep_abs: in ns!
		:param qubit_index:
		:param res_index:
		:param flux_index:
		:param TLS_index:
		:param n_avg:
		:param cd_time:
		:param cd_time_TLS:
		:param simulate_flag:
		:param simulation_len:
		:param plot_flag:
		:return:
			machine
			tau_sweep_abs
			sig_amp
		"""
		if cd_time_TLS is None:
			cd_time_TLS = cd_time_qubit

		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		swap_length = machine.flux_lines[flux_index].iswap.length[TLS_index]
		swap_amp = machine.flux_lines[flux_index].iswap.level[TLS_index]
		tau_sweep_cc = tau_sweep_abs//4 # in clock cycles
		tau_sweep_cc = np.unique(tau_sweep_cc)
		tau_sweep = tau_sweep_cc.astype(int) # clock cycles
		tau_sweep_abs = tau_sweep * 4 # time in ns

		# fLux pulse baking for SWAP
		flux_waveform = np.array([swap_amp] * swap_length)
		def baked_swap_waveform(waveform):
			pulse_segments = []  # Stores the baking objects
			# Create the different baked sequences, each one corresponding to a different truncated duration
			with baking(config, padding_method="right") as b:
				b.add_op("flux_pulse", machine.flux_lines[flux_index].name, waveform.tolist())
				b.play("flux_pulse", machine.flux_lines[flux_index].name)
				pulse_segments.append(b)
			return pulse_segments
		square_TLS_swap = baked_swap_waveform(flux_waveform)

		with program() as t1_prog:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			tau = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(tau, tau_sweep)):
					play("pi", machine.qubits[qubit_index].name)
					align()
					square_TLS_swap[0].run()
					wait(tau, machine.flux_lines[flux_index].name)
					square_TLS_swap[0].run()
					align()
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					align()
					wait(cd_time_qubit * u.ns, machine.flux_lines[flux_index].name)
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[flux_index].name, -1)])
					wait(cd_time_qubit * u.ns, machine.flux_lines[flux_index].name)
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[flux_index].name, -1)])
					wait(cd_time_TLS * u.ns, machine.flux_lines[flux_index].name)
				save(n, n_st)
			with stream_processing():
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port='9510', octave=octave_config, log_level = "ERROR")

		# Simulate or execute #
		if simulate_flag:
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, t1_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(t1_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
			if plot_flag is True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [12, 8]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure    while results.is_processing():
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
					plt.title("TLS T1")
					plt.plot(tau_sweep_abs, np.sqrt(I**2 + Q**2), "b.")
					plt.xlabel("tau [ns]")
					plt.ylabel("Signal Amplitude [V]")
					plt.pause(0.01)

		# save data
		exp_name = 'T1'
		qubit_name = 'Q' + str(qubit_index + 1) + "_TLS" + str(TLS_index + 1)
		f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
		file_name = f_str + '.mat'
		json_name = f_str + '_state.json'
		savemat(os.path.join(tPath, file_name),
				{"TLS_tau": tau_sweep_abs, "sig_amp": sig_amp, "sig_phase": sig_phase})
		machine._save(os.path.join(tPath, json_name), flat_data=False)

		return machine, tau_sweep_abs, sig_amp

	def TLS_T1_driving(self, machine, tau_sweep_abs, qubit_index, res_index, flux_index, TLS_index, n_avg = 1E3, cd_time_qubit = 10E3, cd_time_TLS = None, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		TLS T1 using direct TLS driving to prepare the excited state
		sequence is TLS pi - wait - SWAP - qubit readout
		
		:param machine
		:param tau_sweep_abs:
		:param qubit_index:
		:param res_index:
		:param flux_index:
		:param TLS_index:
		:param n_avg:
		:param cd_time_qubit:
		:param cd_time_TLS:
		:param tPath:
		:param f_str_datetime:
		:param simulate_flag:
		:param simulation_len:
		:param plot_flag:
		:param machine:
		:return:
			machine
			tau_sweep_abs
			sig_amp
		"""
		if cd_time_TLS is None:
			cd_time_TLS = cd_time_qubit

		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		swap_length = machine.flux_lines[flux_index].iswap.length[TLS_index]
		swap_amp = machine.flux_lines[flux_index].iswap.level[TLS_index]
		tau_sweep_cc = tau_sweep_abs//4 # in clock cycles
		tau_sweep_cc = np.unique(tau_sweep_cc)
		tau_sweep = tau_sweep_cc.astype(int) # clock cycles
		tau_sweep_abs = tau_sweep * 4 # time in ns

		# important, need to update if for qua program
		TLS_if = machine.qubits[qubit_index].f_tls[TLS_index] - machine.octaves[0].LO_sources[1].

		# Update the hardware parameters to TLS of interest
		machine.qubits[qubit_index].hardware_parameters.pi_length_tls = machine.qubits[qubit_index].pi_length_tls[TLS_index]
		machine.qubits[qubit_index].hardware_parameters.pi_amp_tls = machine.qubits[qubit_index].pi_amp_tls[TLS_index]

		# fLux pulse baking for SWAP
		flux_waveform = np.array([swap_amp] * swap_length)
		def baked_swap_waveform(waveform):
			pulse_segments = []  # Stores the baking objects
			# Create the different baked sequences, each one corresponding to a different truncated duration
			with baking(config, padding_method="right") as b:
				b.add_op("flux_pulse", machine.flux_lines[flux_index].name, waveform.tolist())
				b.play("flux_pulse", machine.flux_lines[flux_index].name)
				pulse_segments.append(b)
			return pulse_segments
		square_TLS_swap = baked_swap_waveform(flux_waveform)

		with program() as t1_prog:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			tau = declare(int)

			update_frequency(machine.qubits[qubit_index].name, TLS_if) # important, otherwise will use the if in configuration, calculated from f_01
			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(tau, tau_sweep)):
					play("pi_tls", machine.qubits[qubit_index].name)
					wait(tau, machine.qubits[qubit_index].name)
					align()
					square_TLS_swap[0].run()
					align()
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					align()
					wait(cd_time_qubit * u.ns)
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[flux_index].name, -1)])
					wait(cd_time_TLS * u.ns)
				save(n, n_st)
			with stream_processing():
				I_st.buffer(len(tau_sweep)).average().save("I")
				Q_st.buffer(len(tau_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port='9510', octave=octave_config, log_level = "ERROR")

		# Simulate or execute #
		if simulate_flag:
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, t1_prog, simulation_config)
			job.get_simulated_samples().con1.plot()

			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(t1_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
			if plot_flag is True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [12, 8]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure    while results.is_processing():
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
					plt.title("TLS T1 (driving)")
					plt.plot(tau_sweep_abs, sig_amp, "b.")
					plt.xlabel("tau [ns]")
					plt.ylabel("Signal Amplitude [V]")
					plt.pause(0.01)

		# save data
		exp_name = 'T1_driving'
		qubit_name = 'Q' + str(qubit_index + 1) + "_TLS" + str(TLS_index + 1)
		f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
		file_name = f_str + '.mat'
		json_name = f_str + '_state.json'
		savemat(os.path.join(tPath, file_name),
				{"TLS_tau": tau_sweep_abs, "sig_amp": sig_amp, "sig_phase": sig_phase})
		machine._save(os.path.join(tPath, json_name), flat_data=False)

		return machine, tau_sweep_abs, sig_amp
