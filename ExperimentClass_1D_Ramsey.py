class EH_Ramsey:
	def __init__(self, ref_to_update_tPath, ref_to_update_str_datetime,ref_to_set_octave):
		self.update_tPath = ref_to_update_tPath
		self.update_str_datetime = ref_to_update_str_datetime
		self.set_octave = ref_to_set_octave

	def ramsey(self, machine, ramsey_duration_sweep, qubit_index, res_index, n_avg = 1E3, detuning = 1E6, cd_time = 10E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		qubit Ramsey in 1D. Detuning realized by tuning the phase of second pi/2 pulse
		sequence given by pi/2 - wait - pi/2 for various wait times
		the frame of the last pi/2 pulse is rotated rather than using qubit detuning

		:param machine
		:param ramsey_duration_sweep: in clock cycles!
		:param qubit_index:
		:param res_index:
		:param n_avg:
		:param detuning: effective detuning, in unit of Hz!
		:param cd_time:
		:param simulate_flag:
		:param simulation_len:
		:param plot_flag:
		Return:
			machine
			rabi_duration_sweep: in ns!
			sig_amp
		"""
		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		if min(ramsey_duration_sweep) < 4:
			print("some ramsey lengths shorter than 4 clock cycles, removed from run")
			ramsey_duration_sweep = ramsey_duration_sweep[ramsey_duration_sweep>3]

		ramsey_duration_sweep = ramsey_duration_sweep.astype(int)
		
		with program() as ramsey_vr:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)
			phase = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, ramsey_duration_sweep)):
					assign(phase, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
					with strict_timing_():
						play("pi2", machine.qubits[qubit_index].name)
						wait(t, machine.qubits[qubit_index].name)
						frame_rotation_2pi(phase, machine.qubits[qubit_index].name)
						play("pi2", machine.qubits[qubit_index].name)
					align(machine.qubits[qubit_index].name, machine.resonators[res_index].name)
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					reset_frame(machine.qubits[qubit_index].name) # to avoid phase accumulation
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(ramsey_duration_sweep)).average().save("I")
				Q_st.buffer(len(ramsey_duration_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")

		if simulate_flag:
			simulation_config = SimulationConfig(duration=simulation_len)  # in clock cycles
			job = qmm.simulate(config, ramsey_vr, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(ramsey_vr)
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
					plt.title("Ramsey with detuning = %i MHz" % (detuning/1E6))
					plt.plot(ramsey_duration_sweep * 4, sig_amp, "b.")
					plt.xlabel("tau [ns]")
					plt.ylabel("Signal Amplitude [V]")
					plt.pause(0.01)

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
			sig_amp = np.sqrt(I ** 2 + Q ** 2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'ramsey'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"Q_ramsey_duration": ramsey_duration_sweep * 4, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, ramsey_duration_sweep * 4, sig_amp

	def TLS_ramsey(self, machine, ramsey_duration_sweep, qubit_index, res_index, flux_index, TLS_index, n_avg = 1E3, detuning = 1E6, cd_time_qubit = 20E3, cd_time_TLS = None, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		TLS Ramsey in 1D. Detuning realized by tuning the phase of second pi/2 pulse
		sequence given by pi/2 - wait - pi/2 for various wait times
		the frame of the last pi/2 pulse is rotated rather than using actual driving freq. detuning

		:param machine
		:param ramsey_duration_sweep: in clock cycles!
		:param qubit_index:
		:param res_index:
		:param flux_index:
		:param TLS_index:
		:param n_avg:
		:param detuning:
		:param cd_time_qubit:
		:param cd_time_TLS:
		:param simulate_flag:
		:param simulation_len:
		:param plot_flag:
		
		:return:
			machine
			ramsey_duration_sweep * 4: in ns!
			sig_amp
		"""
		if cd_time_TLS is None:
			cd_time_TLS = cd_time_qubit

		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		# important, need to update if for qua program
		TLS_if = machine.qubits[qubit_index].f_tls[TLS_index] - machine.octaves[0].LO_sources[1].

		# Update the hardware parameters to TLS of interest
		machine.qubits[qubit_index].hardware_parameters.pi_length_tls = machine.qubits[qubit_index].pi_length_tls[TLS_index]
		machine.qubits[qubit_index].hardware_parameters.pi_amp_tls = machine.qubits[qubit_index].pi_amp_tls[TLS_index]

		if min(ramsey_duration_sweep) < 4:
			print("some ramsey lengths shorter than 4 clock cycles, removed from run")
			ramsey_duration_sweep = ramsey_duration_sweep[ramsey_duration_sweep>3]

		ramsey_duration_sweep = ramsey_duration_sweep.astype(int)
		
		# fLux pulse baking for SWAP
		swap_length = machine.flux_lines[flux_index].iswap.length[TLS_index]
		swap_amp = machine.flux_lines[flux_index].iswap.level[TLS_index]
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

		with program() as tls_ramsey:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)
			phase = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)

			update_frequency(machine.qubits[qubit_index].name, TLS_if) # important, otherwise will use the if in configuration, calculated from f_01
			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, ramsey_duration_sweep)):
					assign(phase, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
					with strict_timing_():
						play("pi2_tls", machine.qubits[qubit_index].name)
						wait(t, machine.qubits[qubit_index].name)
						frame_rotation_2pi(phase, machine.qubits[qubit_index].name)
						play("pi2_tls", machine.qubits[qubit_index].name)
					align()
					square_TLS_swap[0].run()
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					align()
					wait(cd_time_qubit * u.ns, machine.resonators[qubit_index].name)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[flux_index].name, -1)])
					wait(cd_time_TLS * u.ns, machine.resonators[qubit_index].name)
					reset_frame(machine.qubits[qubit_index].name) # to avoid phase accumulation
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(ramsey_duration_sweep)).average().save("I")
				Q_st.buffer(len(ramsey_duration_sweep)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")

		if simulate_flag:
			simulation_config = SimulationConfig(duration=simulation_len)  # in clock cycles
			job = qmm.simulate(config, tls_ramsey, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(tls_ramsey)
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
					plt.title("Ramsey with detuning = %i MHz" % (detuning/1E6))
					#plt.plot(ramsey_duration_sweep * 4, sig_amp, "b.")
					plt.plot(ramsey_duration_sweep * 4, sig_amp, "b.")
					plt.xlabel("tau [ns]")
					#plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [V]")
					plt.ylabel("Signal Amplitude [V]")
					plt.pause(0.01)

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
			sig_amp = np.sqrt(I ** 2 + Q ** 2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'ramsey'
			qubit_name = 'Q' + str(qubit_index + 1) + 'TLS' + str(TLS_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"TLS_ramsey_duration": ramsey_duration_sweep * 4, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, ramsey_duration_sweep * 4, sig_amp
