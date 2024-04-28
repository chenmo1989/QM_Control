class EH_Rabi:
	"""
	class in ExperimentHandle, for Rabi sequence related 1D experiments
	Methods:
		update_tPath
		update_str_datetime
		qubit_freq(self, qubit_freq_sweep, qubit_index, res_index, flux_index, n_avg, cd_time, ff_amp = 1.0, simulate_flag = False, simulation_len = 1000)
	"""
	def __init__(self, ref_to_update_tPath, ref_to_update_str_datetime, ref_to_set_octave):
		self.update_tPath = ref_to_update_tPath
		self.update_str_datetime = ref_to_update_str_datetime
		self.set_octave = ref_to_set_octave

	def qubit_freq(self, machine, qubit_freq_sweep, qubit_index, res_index, flux_index, pi_amp_rel = 1.0, ff_amp = 0.0, n_avg = 1E3, cd_time = 10E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		qubit spectroscopy experiment in 1D (equivalent of ESR for spin qubit)

		Args:
		:param machine:
		:param qubit_freq_sweep: 1D array of qubit frequency sweep
		:param qubit_index:
		:param res_index:
		:param flux_index:
		:param n_avg: repetition of the experiments
		:param cd_time: cooldown time between subsequent experiments
		:param ff_amp: fast flux amplitude the overlaps with the Rabi pulse. The ff pulse is 40ns longer than Rabi pulse, and share the same center time.
		:param simulate_flag: True-run simulation; False (default)-run experiment.
		:param simulation_len: Length of the sequence to simulate. In clock cycles (4ns).
		:param plot_flag: True (default) plot the experiment. False, do not plot.
		Return:
			machine
			qubit_freq_sweep
			sig_amp
		"""
		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
		qubit_if_sweep = qubit_freq_sweep - qubit_lo
		qubit_if_sweep = np.round(qubit_if_sweep)
		ff_duration = machine.qubits[qubit_index].pi_length + 40

		if np.max(abs(qubit_if_sweep)) > 400E6: # check if parameters are within hardware limit
			print("qubit if range > 400MHz")
			return machine, None, None

		with program() as qubit_freq_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			df = declare(int)

			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(df,qubit_if_sweep)):
					update_frequency(machine.qubits[qubit_index].name, df)
					play("const" * amp(ff_amp), machine.flux_lines[flux_index].name, duration=ff_duration * u.ns)
					wait(5, machine.qubits[qubit_index].name)
					play('pi'*amp(pi_amp_rel), machine.qubits[qubit_index].name)
					align(machine.qubits[qubit_index].name, machine.flux_lines[flux_index].name,
						  machine.resonators[res_index].name)
					#wait(4) # avoid overlap between Z and RO
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					align()
					wait(50)
					# eliminate charge accumulation
					play("const" * amp(-1 * ff_amp), machine.flux_lines[flux_index].name, duration=ff_duration * u.ns)
					wait(cd_time * u.ns, machine.resonators[res_index].name)
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
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		# Simulate or execute #
		if simulate_flag: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, qubit_freq_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(qubit_freq_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
		    #%matplotlib qt
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
				# progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())

				if plot_flag == True:
					plt.cla()
					plt.title("qubit spectroscopy")
					plt.plot((qubit_freq_sweep) / u.MHz, np.sqrt(I**2 +  Q**2), ".")
					#plt.plot((qubit_freq_sweep) / u.MHz, I, "o")
					#plt.plot((qubit_freq_sweep) / u.MHz, Q, "x")
					plt.xlabel("Frequency [MHz]")
					plt.ylabel("Signal Amplitude [V]")

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
			sig_amp = np.sqrt(I**2 + Q**2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'freq'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"Q_freq": qubit_freq_sweep, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, qubit_freq_sweep, sig_amp

	def rabi_length(self, machine, rabi_duration_sweep, qubit_index, res_index, pi_amp_rel = 1.0, n_avg = 1E3, cd_time = 10E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		qubit rabi experiment in 1D (sweeps length of rabi pulse)
		
		:param machine:
		:param rabi_duration_sweep: in clock cycles!
		:param qubit_index:
		:param res_index:
		:param flux_index:
		:param pi_amp_rel:
		:param n_avg:
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

		if min(rabi_duration_sweep) < 4:
			print("some rabi lengths shorter than 4 clock cycles, removed from run")
			rabi_duration_sweep = rabi_duration_sweep[rabi_duration_sweep>3]

		rabi_duration_sweep = rabi_duration_sweep.astype(int)

		with program() as time_rabi:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, rabi_duration_sweep)):
					wait(5, machine.qubits[qubit_index].name)
					play("pi" * amp(pi_amp_rel), machine.qubits[qubit_index].name, duration=t)
					wait(5, machine.qubits[qubit_index].name)
					align(machine.qubits[qubit_index].name, machine.resonators[res_index].name)
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
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
			return machine, None, None
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
			I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
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

			return machine, rabi_duration_sweep * 4, sig_amp

	def rabi_amp(self, machine, rabi_amp_sweep_rel, qubit_index, res_index, n_avg = 1E3, cd_time = 10E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		qubit rabi experiment in 1D (sweeps amplitude of rabi pulse)
		note that the input argument is in relative amplitude, the return argument is in absolute amplitude
		
		:param machine
		:param rabi_amp_sweep: relative amplitude, based on pi_amp
		:param qubit_index:
		:param res_index:
		:param n_avg:
		:param cd_time:
		:param simulate_flag:
		:param simulation_len:
		:param plot_flag:
		Return:
			machine
			rabi_amp_sweep_abs
			sig_amp
		"""
		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		qubit_if = machine.qubits[qubit_index].f_01 - machine.octaves[0].LO_sources[1].LO_frequency
		if max(abs(rabi_amp_sweep_rel)) > 2:
			print("some relative amps > 2, removed from experiment run")
			rabi_amp_sweep_rel = rabi_amp_sweep_rel[abs(rabi_amp_sweep_rel) < 2]
		rabi_amp_sweep_abs = rabi_amp_sweep_rel * machine.qubits[qubit_index].pi_amp # actual rabi amplitude

		with program() as power_rabi:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			a = declare(fixed)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(a, rabi_amp_sweep_rel)):
					update_frequency(machine.qubits[qubit_index].name, qubit_if)
					wait(5, machine.qubits[qubit_index].name)
					play("pi" * amp(a), machine.qubits[qubit_index].name)
					wait(5, machine.qubits[qubit_index].name)
					align(machine.qubits[qubit_index].name, machine.resonators[res_index].name)
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(rabi_amp_sweep_rel)).average().save("I")
				Q_st.buffer(len(rabi_amp_sweep_rel)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")

		if simulate_flag:
			simulation_config = SimulationConfig(duration=1000)  # in clock cycles
			job = qmm.simulate(config, power_rabi, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(power_rabi)
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
					plt.title("Power Rabi")
					plt.plot(rabi_amp_sweep_abs, sig_amp, "b.")
					plt.xlabel("rabi amplitude [V]")
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
			exp_name = 'power_rabi'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"Q_rabi_amplitude": rabi_amp_sweep_abs, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, rabi_amp_sweep_abs, sig_amp

	def qubit_switch_delay(self, machine, qubit_switch_delay_sweep, qubit_index, res_index, n_avg, cd_time, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		1D experiment to calibrate switch delay for the qubit.

		Args:
			machine:
			qubit_switch_delay_sweep (): in ns
			qubit_index ():
			res_index ():
			n_avg ():
			cd_time ():
			simulate_flag ():
			simulation_len ():
			plot_flag ():
			
		Returns:
			machine
			qubit_switch_delay_sweep
			sig_amp
		"""
		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
		qubit_if = machine.qubits[qubit_index].f_01 - qubit_lo

		if abs(qubit_if) > 400E6: # check if parameters are within hardware limit
			print("qubit if > 400MHz")
			return machine, None, None

		with program() as qubit_switch_delay_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()

			with for_(n, 0, n < n_avg, n+1):
				update_frequency(machine.qubits[qubit_index].name, qubit_if)
				play('pi2', machine.qubits[qubit_index].name)
				align()
				readout_avg_macro(machine.resonators[res_index].name,I,Q)
				wait(cd_time * u.ns, machine.resonators[res_index].name)
				save(I, I_st)
				save(Q, Q_st)
			with stream_processing():
				I_st.average().save("I")
				Q_st.average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		# Simulate or execute #
		if simulate_flag: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, qubit_switch_delay_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			start_time = time.time()
			I_tot = []
			Q_tot = []
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]

			for delay_index, delay_value in enumerate(qubit_switch_delay_sweep):
				machine = self.set_digital_delay(machine, "qubits", int(delay_value))
				config = build_config(machine)

				qm = qmm.open_qm(config)
				job = qm.execute(qubit_switch_delay_prog)
				# Get results from QUA program
				results = fetching_tool(job, data_list=["I", "Q"], mode = "live")

				if plot_flag:
					interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
				while results.is_processing():
					# Fetch results
					time.sleep(0.1)

				I, Q = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
				I_tot.append(I)
				Q_tot.append(Q)

				# progress bar
				progress_counter(delay_index, len(qubit_switch_delay_sweep), start_time=start_time)

			I_tot = np.array(I_tot)
			Q_tot = np.array(Q_tot)
			sigs_qubit = I_tot + 1j * Q_tot
			sig_amp = np.abs(sigs_qubit)  # Amplitude
			sig_phase = np.angle(sigs_qubit)  # Phase

			if plot_flag == True:
				plt.cla()
				plt.title("qubit switch delay")
				plt.plot(qubit_switch_delay_sweep, sig_amp, ".")
				plt.xlabel("switch delay [ns]")
				plt.ylabel("Signal Amplitude [V]")

			# save data
			exp_name = 'qubit_switch_delay'
			qubit_name = 'Q' + str(res_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"qubit_delay": qubit_switch_delay_sweep, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, qubit_switch_delay_sweep, sig_amp

	def qubit_switch_buffer(self, machine, qubit_switch_buffer_sweep, qubit_index, res_index, n_avg, cd_time, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		1D experiment to calibrate switch buffer for the qubit.

		Args:
			machine
			qubit_switch_buffer_sweep (): in ns, this will be added to both sides of the switch (x2), to account for the rise and fall
			qubit_index ():
			res_index ():
			n_avg ():
			cd_time ():
			tPath ():
			f_str_datetime ():
			simulate_flag ():
			simulation_len ():
			plot_flag ():
			
		Returns:
			machine
			qubit_switch_buffer_sweep
			sig_amp
		"""
		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
		qubit_if = machine.qubits[qubit_index].f_01 - qubit_lo

		if abs(qubit_if) > 400E6: # check if parameters are within hardware limit
			print("qubit if > 400MHz")
			return machine, None, None

		with program() as qubit_switch_buffer_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()

			with for_(n, 0, n < n_avg, n+1):
				update_frequency(machine.qubits[qubit_index].name, qubit_if)
				play('pi2', machine.qubits[qubit_index].name)
				align()
				readout_avg_macro(machine.resonators[res_index].name,I,Q)
				wait(cd_time * u.ns, machine.resonators[res_index].name)
				save(I, I_st)
				save(Q, Q_st)
			with stream_processing():
				I_st.average().save("I")
				Q_st.average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		# Simulate or execute #
		if simulate_flag: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, qubit_switch_buffer_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			start_time = time.time()
			I_tot = []
			Q_tot = []
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]

			for buffer_index, buffer_value in enumerate(qubit_switch_buffer_sweep):
				machine = self.set_digital_buffer(machine, "qubits", int(buffer_value))
				config = build_config(machine)

				qm = qmm.open_qm(config)
				job = qm.execute(qubit_switch_buffer_prog)
				# Get results from QUA program
				results = fetching_tool(job, data_list=["I", "Q"], mode = "live")

				if plot_flag:
					interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
				while results.is_processing():
					# Fetch results
					time.sleep(0.1)

				I, Q = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
				I_tot.append(I)
				Q_tot.append(Q)

				# progress bar
				progress_counter(buffer_index, len(qubit_switch_buffer_sweep), start_time=start_time)

			I_tot = np.array(I_tot)
			Q_tot = np.array(Q_tot)
			sigs_qubit = I_tot + 1j * Q_tot
			sig_amp = np.abs(sigs_qubit)  # Amplitude
			sig_phase = np.angle(sigs_qubit)  # Phase

			if plot_flag == True:
				plt.cla()
				plt.title("qubit switch buffer")
				plt.plot(qubit_switch_buffer_sweep, sig_amp, ".")
				plt.xlabel("switch buffer [ns]")
				plt.ylabel("Signal Amplitude [V]")

			# save data
			exp_name = 'qubit_switch_buffer'
			qubit_name = 'Q' + str(res_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"qubit_buffer": qubit_switch_buffer_sweep, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, qubit_switch_buffer_sweep, sig_amp

	def TLS_freq(self, machine, TLS_freq_sweep, qubit_index, res_index, flux_index, TLS_index, pi_amp_rel = 1.0, n_avg = 1E3, cd_time_qubit = 10E3, cd_time_TLS = None, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		1D experiment of TLS spectroscopy
		a strong MW pulse - SWAP - readout

		uses the iswap defined in machine.flux_lines[flux_index].iswap.length/level[TLS_index]
		the TLS driving pulse is a square wave, with duration = machine.qubits[qubit_index].pi_length_tls[TLS_index],
		 amplitude = machine.qubits[qubit_index].pi_amp_tls[TLS_index]

		Args:
			machine
			TLS_freq_sweep ():
			qubit_index ():
			res_index ():
			flux_index ():
			TLS_index ():
			pi_amp_rel ():
			n_avg ():
			cd_time ():
			simulate_flag ():
			simulation_len ():
			plot_flag ():

		Returns:
			machine
			TLS_freq_sweep
			sig_amp
		"""
		calibrate_octave = False # flag for calibrating octave. So that I can move this to the real run, avoiding it for simulation

		if cd_time_TLS is None:
			cd_time_TLS = cd_time_qubit

		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
		TLS_if_sweep = TLS_freq_sweep - qubit_lo
		TLS_if_sweep = np.round(TLS_if_sweep)
		# Update the hardware parameters to TLS of interest
		machine.qubits[qubit_index].hardware_parameters.pi_length_tls = machine.qubits[qubit_index].pi_length_tls[TLS_index]
		machine.qubits[qubit_index].hardware_parameters.pi_amp_tls = machine.qubits[qubit_index].pi_amp_tls[TLS_index]

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

		if np.max(abs(TLS_if_sweep)) > 350E6: # check if parameters are within hardware limit
			print("TLS if range > 350MHz, changing LO...")
			qubit_lo = np.mean(TLS_freq_sweep) - 200E6
			qubit_lo = int(qubit_lo.tolist())
			machine.octaves[0].LO_sources[1].LO_frequency = qubit_lo + 0E6
			calibrate_octave = True
			# reassign values
			TLS_if_sweep = TLS_freq_sweep - qubit_lo
			TLS_if_sweep = np.floor(TLS_if_sweep)
			if np.max(abs(TLS_if_sweep)) > 350E6:  # check if parameters are within hardware limit
				print("TLS freq sweep range too large, abort...")
				return machine, None, None

		with program() as TLS_freq_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			df = declare(int)

			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(df,TLS_if_sweep)):
					update_frequency(machine.qubits[qubit_index].name, df)
					if pi_amp_rel==1.0:
						play('pi_tls', machine.qubits[qubit_index].name)
					else:
						play('pi_tls' * amp(pi_amp_rel), machine.qubits[qubit_index].name)
					align()
					square_TLS_swap[0].run()
					align()
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					align()
					save(I, I_st)
					save(Q, Q_st)
					# eliminate charge accumulation, also initialize TLS
					wait(cd_time_qubit * u.ns, machine.resonators[res_index].name)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[flux_index].name, -1)])
					wait(cd_time_TLS * u.ns, machine.resonators[res_index].name)
				save(n, n_st)
			with stream_processing():
				n_st.save('iteration')
				I_st.buffer(len(TLS_if_sweep)).average().save("I")
				Q_st.buffer(len(TLS_if_sweep)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		# Simulate or execute #
		if simulate_flag: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, TLS_freq_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			if calibrate_octave:
				machine = self.set_octave.calibration(machine, qubit_index, res_index, TLS_index = TLS_index, log_flag = True, calibration_flag = True, qubit_only = True)

			qm = qmm.open_qm(config)
			job = qm.execute(TLS_freq_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
		    #%matplotlib qt
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
				# progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())

				if plot_flag:
					plt.cla()
					plt.title("TLS spectroscopy")
					plt.plot((TLS_freq_sweep) / u.MHz, np.sqrt(I**2 +  Q**2), ".")
					plt.xlabel("Frequency [MHz]")
					plt.ylabel("Signal Amplitude [V]")

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
			sig_amp = np.sqrt(I**2 + Q**2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'freq'
			qubit_name = 'Q' + str(qubit_index + 1) + 'TLS' + str(TLS_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"TLS_freq": TLS_freq_sweep, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, TLS_freq_sweep, sig_amp

	def TLS_rabi_length(self, machine, rabi_duration_sweep, qubit_index, res_index, flux_index, TLS_index, pi_amp_rel = 1.0, n_avg = 1E3, cd_time_qubit = 20E3, cd_time_TLS = None, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		1D experiment that runs time rabi of TLS

		uses the iswap defined in machine.flux_lines[flux_index].iswap.length/level[TLS_index]
		the TLS driving pulse is a square wave, with amplitude = machine.qubits[qubit_index].pi_amp_tls[TLS_index] * 0.25 V

		Args:
			machine
			rabi_duration_sweep (): in clock cycles! Must be integers!
			qubit_index ():
			res_index ():
			flux_index ():
			TLS_index ():
			pi_amp_rel ():
			n_avg ():
			cd_time_qubit ():
			cd_time_TLS ():
			simulate_flag ():
			simulation_len ():
			plot_flag ():
			
		Returns:
			machine
			rabi_duration_sweep * 4
			sig_amp
		"""
		if cd_time_TLS is None:
			cd_time_TLS = cd_time_qubit

		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		if min(rabi_duration_sweep) < 4:
			print("some rabi lengths shorter than 4 clock cycles, removed from run")
			rabi_duration_sweep = rabi_duration_sweep[rabi_duration_sweep>3]
		rabi_duration_sweep = rabi_duration_sweep.astype(int)

		# important, need to update if for qua program
		TLS_if = machine.qubits[qubit_index].f_tls[TLS_index] - machine.octaves[0].LO_sources[1].LO_frequency

		# Update the hardware parameters to TLS of interest
		machine.qubits[qubit_index].hardware_parameters.pi_length_tls = machine.qubits[qubit_index].pi_length_tls[TLS_index]
		machine.qubits[qubit_index].hardware_parameters.pi_amp_tls = machine.qubits[qubit_index].pi_amp_tls[TLS_index]

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

		with program() as TLS_rabi_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			t = declare(int)

			update_frequency(machine.qubits[qubit_index].name, TLS_if) # important, otherwise will use the if in configuration, calculated from f_01
			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(t,rabi_duration_sweep)):
					if pi_amp_rel==1.0:
						play('pi_tls', machine.qubits[qubit_index].name, duration = t) # clock cycles
					else:
						play('pi_tls' * amp(pi_amp_rel), machine.qubits[qubit_index].name, duration = t) # clock cycles
					align()
					square_TLS_swap[0].run()
					align()
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					align()
					save(I, I_st)
					save(Q, Q_st)
					# eliminate charge accumulation, also initialize TLS
					wait(cd_time_qubit * u.ns, machine.resonators[res_index].name)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[flux_index].name, -1)])
					wait(cd_time_TLS * u.ns, machine.resonators[res_index].name)
				save(n, n_st)
			with stream_processing():
				n_st.save('iteration')
				I_st.buffer(len(rabi_duration_sweep)).average().save("I")
				Q_st.buffer(len(rabi_duration_sweep)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		# Simulate or execute #
		if simulate_flag: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, TLS_rabi_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(TLS_rabi_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
		    #%matplotlib qt
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
				# progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())

				if plot_flag:
					plt.cla()
					plt.title("TLS time rabi")
					#plt.plot(rabi_duration_sweep * 4, np.sqrt(I**2 +  Q**2), ".")
					plt.plot(rabi_duration_sweep * 4, np.sqrt(I**2 + Q**2), ".")
					plt.xlabel("tau [ns]")
					#plt.ylabel("Signal Amplitude [V]")
					plt.ylabel("Signal Amplitude [V]")

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
			sig_amp = np.sqrt(I**2 + Q**2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'TLS_time_rabi'
			qubit_name = 'Q' + str(qubit_index + 1) + 'TLS' + str(TLS_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"TLS_rabi_duration": rabi_duration_sweep * 4, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, rabi_duration_sweep * 4, sig_amp

	def TLS_rabi_amp(self, machine, rabi_amp_sweep_rel, qubit_index, res_index, flux_index, TLS_index, n_avg = 1E3, cd_time_qubit = 20E3, cd_time_TLS = None, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""

		1D experiment that runs power rabi of TLS

		uses the iswap defined in machine.flux_lines[flux_index].iswap.length/level[TLS_index]
		the TLS driving pulse is a square wave, with amplitude = machine.qubits[qubit_index].pi_amp_tls[TLS_index] * 0.25 V

		note that the input argument is in relative amplitude, the return argument is in absolute amplitude

		:param machine
		:param rabi_amp_sweep: relative amplitude, based on pi_amp_tls
		:param qubit_index:
		:param res_index:
		:param flux_index:
		:param n_avg:
		:param cd_time:
		:param simulate_flag:
		:param simulation_len:
		:param plot_flag:
		Return:
			machine
			rabi_amp_sweep_abs
			sig_amp
		"""

		if cd_time_TLS is None:
			cd_time_TLS = cd_time_qubit
		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		# important, need to update if for qua program
		TLS_if = machine.qubits[qubit_index].f_tls[TLS_index] - machine.octaves[0].LO_sources[1].LO_frequency

		# Update the hardware parameters to TLS of interest
		machine.qubits[qubit_index].hardware_parameters.pi_length_tls = machine.qubits[qubit_index].pi_length_tls[TLS_index]
		machine.qubits[qubit_index].hardware_parameters.pi_amp_tls = machine.qubits[qubit_index].pi_amp_tls[TLS_index]

		if max(abs(rabi_amp_sweep_rel)) > 2:
			print("some relative amps > 2, removed from experiment run")
			rabi_amp_sweep_rel = rabi_amp_sweep_rel[abs(rabi_amp_sweep_rel) < 2]
		rabi_amp_sweep_abs = rabi_amp_sweep_rel * machine.qubits[qubit_index].hardware_parameters.pi_amp_tls # actual rabi amplitude
		if max(abs(rabi_amp_sweep_abs)) > 0.5:
			print("some abs amps > 0.5, removed from experiment run")
			rabi_amp_sweep_rel = rabi_amp_sweep_rel[abs(rabi_amp_sweep_abs) < 0.5]
			rabi_amp_sweep_abs = rabi_amp_sweep_abs[abs(rabi_amp_sweep_abs) < 0.5]

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

		with program() as tls_power_rabi:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			a = declare(fixed)
			
			update_frequency(machine.qubits[qubit_index].name, TLS_if) # important, otherwise will use the if in configuration, calculated from f_01
			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(a, rabi_amp_sweep_rel)):
					play("pi_tls" * amp(a), machine.qubits[qubit_index].name)
					align()
					square_TLS_swap[0].run()
					align()
					readout_avg_macro(machine.resonators[res_index].name, I, Q)
					align()
					save(I, I_st)
					save(Q, Q_st)
					# eliminate charge accumulation, also initialize TLS
					wait(cd_time_qubit * u.ns, machine.resonators[res_index].name)
					align()
					square_TLS_swap[0].run(amp_array=[(machine.flux_lines[flux_index].name, -1)])
					wait(cd_time_TLS * u.ns, machine.resonators[res_index].name)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(rabi_amp_sweep_rel)).average().save("I")
				Q_st.buffer(len(rabi_amp_sweep_rel)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")

		if simulate_flag:
			simulation_config = SimulationConfig(duration=1000)  # in clock cycles
			job = qmm.simulate(config, tls_power_rabi, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(tls_power_rabi)
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
					plt.title("TLS power rabi")
					plt.plot(rabi_amp_sweep_abs, sig_amp, "b.")
					plt.xlabel("rabi amplitude [V]")
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
			exp_name = 'TLS_power_rabi'
			qubit_name = 'Q' + str(qubit_index + 1) + 'TLS' + str(TLS_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"TLS_rabi_amplitude": rabi_amp_sweep_abs, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, rabi_amp_sweep_abs, sig_amp

	def ef_freq(self, machine, ef_freq_sweep, qubit_index, res_index, pi_amp_rel_ef = 1.0, n_avg = 1E3, cd_time = 10E3, readout_state = 'g', simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		ef spectroscopy experiment in 1D

		Args:
		:param machine
		:param ef_freq_sweep: 1D array of qubit ef transition frequency sweep
		:param qubit_index:
		:param res_index:
		:param pi_amp_rel_ef: 1.0 (default). relative amplitude of pi pulse for ef transition
		:param n_avg: repetition of the experiments
		:param cd_time: cooldown time between subsequent experiments
		:param readout_state: state used for readout. If 'g' (default), ground state will be used, so a pi pulse to bring population back to g is employed. If 'e', then no additional pi pulse for readout is sent
		:param ff_amp: fast flux amplitude the overlaps with the Rabi pulse. The ff pulse is 40ns longer than Rabi pulse, and share the same center time.
		:param simulate_flag: True-run simulation; False (default)-run experiment.
		:param simulation_len: Length of the sequence to simulate. In clock cycles (4ns).
		:param plot_flag: True (default) plot the experiment. False, do not plot.
		Return:
			machine
			ef_freq_sweep
			sig_amp
		"""
		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		if readout_state not in ['g','e']:
			print('Readout state not g/e. Abort...')
			return machine, None, None

		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency
		qubit_if = machine.qubits[qubit_index].f_01 - machine.octaves[0].LO_sources[1].LO_frequency
		ef_if_sweep = ef_freq_sweep - qubit_lo
		ef_if_sweep = np.round(ef_if_sweep)

		if abs(qubit_if) > 350E6:
			print("qubit if > 350MHz")
			return machine, None, None
		if abs(qubit_if) < 20E6:
			print("qubit if < 20MHz")
			return machine, None, None
		if np.max(abs(ef_if_sweep)) > 350E6: # check if parameters are within hardware limit
			print("ef if range > 350MHz")
			return machine, None, None
		if np.min(abs(ef_if_sweep)) < 20E6: # check if parameters are within hardware limit
			print("ef if range < 20MHz")
			return machine, None, None

		with program() as ef_freq_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			df = declare(int)
			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(df,ef_if_sweep)):
					update_frequency(machine.qubits[qubit_index].name, qubit_if)
					play('pi'*amp(pi_amp_rel), machine.qubits[qubit_index].name)
					update_frequency(machine.qubits[qubit_index].name, df)
					
					if pi_amp_rel_ef==1.0:
						play('pi_ef', machine.qubits[qubit_index].name)
					else:
						play('pi_ef' * amp(pi_amp_rel_ef), machine.qubits[qubit_index].name)
					
					if readout_state == 'g':
						update_frequency(machine.qubits[qubit_index].name, qubit_if)
						play('pi', machine.qubits[qubit_index].name)
					
					align()
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					wait(cd_time * u.ns, machine.resonators[res_index].name)
					save(I, I_st)
					save(Q, Q_st)
				save(n, n_st)
			with stream_processing():
				n_st.save('iteration')
				I_st.buffer(len(ef_if_sweep)).average().save("I")
				Q_st.buffer(len(ef_if_sweep)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		# Simulate or execute #
		if simulate_flag: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, ef_freq_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(ef_freq_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
		    #%matplotlib qt
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
				# progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())

				if plot_flag == True:
					plt.cla()
					plt.title("ef spectroscopy")
					plt.plot((ef_freq_sweep) / u.MHz, np.sqrt(I**2 +  Q**2), ".")
					plt.xlabel("Frequency [MHz]")
					plt.ylabel("Signal Amplitude [V]")

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
			sig_amp = np.sqrt(I**2 + Q**2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'ef_freq'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"ef_freq": ef_freq_sweep, "sig_amp": sig_amp, "sig_phase": sig_phase, "readout_state": readout_state})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, ef_freq_sweep, sig_amp

	def ef_rabi_length(self, machine, rabi_duration_sweep, qubit_index, res_index, pi_amp_rel_ef = 1.0, n_avg = 1E3, cd_time = 10E3, readout_state = 'g', simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		qubit ef rabi experiment in 1D (sweeps length of rabi pulse)
		
		:param machine
		:param rabi_duration_sweep: in clock cycles!
		:param qubit_index:
		:param res_index:
		:param pi_amp_rel_ef: 1.0 (default). relative amplitude of pi pulse for ef transition
		:param n_avg:
		:param cd_time:
		:param readout_state: state used for readout. 'g' (default), a g-e pi pulse before readout. If 'e', then no additional pi pulse for readout.
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

		if readout_state not in ['g','e']:
			print('Readout state not g/e. Abort...')
			return machine, None, None

		if min(rabi_duration_sweep) < 4:
			print("some rabi lengths shorter than 4 clock cycles, removed from run")
			rabi_duration_sweep = rabi_duration_sweep[rabi_duration_sweep>3]

		rabi_duration_sweep = rabi_duration_sweep.astype(int)
		qubit_if = machine.qubits[qubit_index].f_01 - machine.octaves[0].LO_sources[1].LO_frequency
		ef_if = machine.qubits[qubit_index].f_01 - machine.qubits[qubit_index].anharmonicity - machine.octaves[0].LO_sources[1].LO_frequency

		if abs(qubit_if) > 350E6:
			print("qubit if > 350MHz")
			return machine, None, None
		if abs(qubit_if) < 20E6:
			print("qubit if < 20MHz")
			return machine, None, None
		if abs(ef_if) > 350E6:
			print("ef if > 350MHz")
			return machine, None, None
		if abs(ef_if) < 20E6:
			print("ef if < 20MHz")
			return machine, None, None

		with program() as time_rabi_ef:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, rabi_duration_sweep)):
					update_frequency(machine.qubits[qubit_index].name, qubit_if)
					play('pi', machine.qubits[qubit_index].name)
					update_frequency(machine.qubits[qubit_index].name, ef_if)
					
					if pi_amp_rel_ef==1.0:
						play('pi_ef', machine.qubits[qubit_index].name, duration=t) # clock cycle
					else:
						play('pi_ef' * amp(pi_amp_rel_ef), machine.qubits[qubit_index].name, duration=t) # clock cycle

					if readout_state == 'g':
						update_frequency(machine.qubits[qubit_index].name, qubit_if)
						play('pi'*amp(pi_amp_rel), machine.qubits[qubit_index].name)
					
					align()
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[res_index].name)
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
			job = qmm.simulate(config, time_rabi_ef, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(time_rabi_ef)
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
					plt.title("ef time Rabi")
					plt.plot(rabi_duration_sweep * 4, sig_amp, "b.")
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
			exp_name = 'ef_time_rabi'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"Q_rabi_duration": rabi_duration_sweep * 4, "sig_amp": sig_amp, "sig_phase": sig_phase, "readout_state": readout_state})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, rabi_duration_sweep * 4, sig_amp

	def ef_rabi_amp(self, machine, rabi_amp_sweep_rel, qubit_index, res_index, n_avg = 1E3, cd_time = 10E3, readout_state = 'g', simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		qubit ef rabi experiment in 1D (sweeps amplitude of rabi pulse)
		note that the input argument is in relative amplitude, the return argument is in absolute amplitude

		:param rabi_amp_sweep_rel: relative amplitude, based on pi_amp_ef
		:param qubit_index:
		:param res_index:
		:param flux_index:
		:param pi_amp_rel: for the ge pulse
		:param n_avg:
		:param cd_time:
		:param readout_state: state used for readout. 'g' (default), a g-e pi pulse before readout. If 'e', then no additional pi pulse for readout.
		:param simulate_flag:
		:param simulation_len:
		:param plot_flag:
		Return:
			machine
			rabi_amp_sweep_abs
			sig_amp
		"""
		tPath = self.update_tPath()
		f_str_datetime = self.update_str_datetime()

		if readout_state not in ['g','e']:
			print('Readout state not g/e. Abort...')
			return machine, None, None

		qubit_if = machine.qubits[qubit_index].f_01 - machine.octaves[0].LO_sources[1].LO_frequency
		rabi_amp_sweep_abs = rabi_amp_sweep_rel * machine.qubits[qubit_index].pi_amp_ef # actual rabi amplitude
		ef_if = machine.qubits[qubit_index].f_01 - machine.qubits[qubit_index].anharmonicity - machine.octaves[0].LO_sources[1].LO_frequency

		if abs(qubit_if) > 350E6:
			print("qubit if > 350MHz")
			return None
		if abs(qubit_if) < 20E6:
			print("qubit if < 20MHz")
			return None
		if abs(ef_if) > 350E6:
			print("ef if > 350MHz")
			return None
		if abs(ef_if) < 20E6:
			print("ef if < 20MHz")
			return None

		with program() as power_rabi_ef:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			a = declare(fixed)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(a, rabi_amp_sweep_rel)):
					update_frequency(machine.qubits[qubit_index].name, qubit_if)
					play('pi', machine.qubits[qubit_index].name)
					update_frequency(machine.qubits[qubit_index].name, ef_if)
					play('pi_ef' * amp(a), machine.qubits[qubit_index].name)
					
					if readout_state == 'g':
						update_frequency(machine.qubits[qubit_index].name, qubit_if)
						play('pi', machine.qubits[qubit_index].name)
					
					align()
					readout_avg_macro(machine.resonators[res_index].name, I, Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[res_index].name)
				save(n, n_st)

			with stream_processing():
				I_st.buffer(len(rabi_amp_sweep_rel)).average().save("I")
				Q_st.buffer(len(rabi_amp_sweep_rel)).average().save("Q")
				n_st.save("iteration")

		#  Open Communication with the QOP  #
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")

		if simulate_flag:
			simulation_config = SimulationConfig(duration=1000)  # in clock cycles
			job = qmm.simulate(config, power_rabi_ef, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(power_rabi_ef)
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
					plt.title("ef power Rabi")
					plt.plot(rabi_amp_sweep_abs, sig_amp, "b.")
					plt.xlabel("rabi amplitude [V]")
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
			exp_name = 'ef_power_rabi'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"Q_rabi_amplitude": rabi_amp_sweep_abs, "sig_amp": sig_amp, "sig_phase": sig_phase, "readout_state": readout_state})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, rabi_amp_sweep_abs, sig_amp

	def ef_rabi_length_thermal(self, machine, rabi_duration_sweep, qubit_index, res_index, flux_index, n_avg = 1E3, cd_time = 10E3, readout_state = 'e', simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		qubit ef rabi experiment with no first ge pi pulse
		This is to measure the oscillation of residual |e> state, A_sig in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.114.240501
		
		:param machine
		:param rabi_duration_sweep: in clock cycles!
		:param qubit_index:
		:param res_index:
		:param flux_index:
		:param n_avg:
		:param cd_time:
		:param readout_state: state used for readout. 'e' (default). If 'g', then a g-e pi pulse before readout.
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

		if readout_state not in ['g','e']:
			print('Readout state not g/e. Abort...')
			return machine, None, None

		if min(rabi_duration_sweep) < 4:
			print("some rabi lengths shorter than 4 clock cycles, removed from run")
			rabi_duration_sweep = rabi_duration_sweep[rabi_duration_sweep>3]

		rabi_duration_sweep = rabi_duration_sweep.astype(int)
		qubit_if = machine.qubits[qubit_index].f_01 - machine.octaves[0].LO_sources[1].LO_frequency
		ef_if = machine.qubits[qubit_index].f_01 - machine.qubits[qubit_index].anharmonicity - machine.octaves[0].LO_sources[1].LO_frequency

		if abs(qubit_if) > 350E6:
			print("qubit if > 350MHz")
			return machine, None, None
		if abs(qubit_if) < 20E6:
			print("qubit if < 20MHz")
			return machine, None, None
		if abs(ef_if) > 350E6:
			print("ef if > 350MHz")
			return machine, None, None
		if abs(ef_if) < 20E6:
			print("ef if < 20MHz")
			return machine, None, None

		with program() as time_rabi_ef_thermal:
			[I, Q, n, I_st, Q_st, n_st] = declare_vars()
			t = declare(int)

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(t, rabi_duration_sweep)):
					update_frequency(machine.qubits[qubit_index].name, ef_if)
					play('pi_ef' * amp(pi_amp_rel_ef), machine.qubits[qubit_index].name, duration=t)
					if readout_state == 'g':
						update_frequency(machine.qubits[qubit_index].name, qubit_if)
						play('pi', machine.qubits[qubit_index].name)
					
					align()
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					save(I, I_st)
					save(Q, Q_st)
					wait(cd_time * u.ns, machine.resonators[res_index].name)
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
			job = qmm.simulate(config, time_rabi_ef_thermal, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(time_rabi_ef_thermal)
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
					plt.title("residual e state time rabi")
					plt.plot(rabi_duration_sweep * 4, sig_amp, "b.")
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
			exp_name = 'time_rabi_ef_thermal'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"Q_rabi_duration": rabi_duration_sweep * 4, "sig_amp": sig_amp, "sig_phase": sig_phase, "readout_state": readout_state})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, rabi_duration_sweep * 4, sig_amp
