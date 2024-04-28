class EH_1D:
	"""
	class for some 1D experiments used for 2D scans
	"""
	def res_freq(self, machine, res_freq_sweep, res_index, n_avg, cd_time, simulate_flag = False, simulation_len = 1000, fig = None):
		"""
		resonator spectroscopy experiment
		this experiment find the resonance frequency by localizing the minima in pulsed transmission signal.
		this 1D experiment is not automatically saved
		Args:
		:param machine
		:param res_freq_sweep: 1D array for resonator frequency sweep
		:param res_index:
		:param n_avg: repetition of expeirment
		:param cd_time: cooldown time between subsequent experiments
		:param simulate_flag: True-run simulation; False (default)-run experiment.
		:param simulation_len: Length of the sequence to simulate. In clock cycles (4ns).
		:param fig: None (default). Fig reference, mainly to have the ability to interupt the experiment.
		Return:
			machine
			I
			Q
		"""

		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		if res_lo < 2E9:
			print("LO < 2GHz, abort")
			return machine, None, None
		res_if_sweep = res_freq_sweep - res_lo
		res_if_sweep = np.floor(res_if_sweep)

		if np.max(abs(res_if_sweep)) > 400E6: # check if parameters are within hardware limit
			print("res if range > 400MHz")
			return machine, None, None

		with program() as rr_freq_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			df = declare(int)

			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(df,res_if_sweep)):
					update_frequency(machine.resonators[res_index].name, df)
					readout_avg_macro(machine.resonators[res_index].name,I,Q)
					wait(cd_time * u.ns, machine.resonators[res_index].name)
					save(I, I_st)
					save(Q, Q_st)
				save(n, n_st)
			with stream_processing():
				n_st.save('iteration')
				I_st.buffer(len(res_if_sweep)).average().save("I")
				Q_st.buffer(len(res_if_sweep)).average().save("Q")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		# Simulate or execute #
		if simulate_flag: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, rr_freq_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(rr_freq_prog)
			# Get results from QUA program
			results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
			# Live plotting
		    #%matplotlib qt
			if fig is not None:
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				I, Q, iteration = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
				# progress bar
				#progress_counter(iteration, n_avg, start_time=results.get_start_time())

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)

			return machine, I, Q

	def qubit_freq(self, machine, qubit_freq_sweep, qubit_index, res_index, flux_index, ff_amp = 0.0, n_avg = 1E3, cd_time = 10E3, simulate_flag = False, simulation_len = 1000, fig = None):
		"""
		qubit spectroscopy experiment in 1D (equivalent of ESR for spin qubit)
		this 1D experiment is not automatically saved
		Args:
		:param machine
		:param qubit_freq_sweep: 1D array of qubit frequency sweep
		:param qubit_index:
		:param res_index:
		:param flux_index:
		:param n_avg: repetition of the experiments
		:param cd_time: cooldown time between subsequent experiments
		:param ff_amp: fast flux amplitude the overlaps with the Rabi pulse. The ff pulse is 40ns longer than Rabi pulse, and share the same center time.
		:param machine:
		:param simulate_flag: True-run simulation; False (default)-run experiment.
		:param simulation_len: Length of the sequence to simulate. In clock cycles (4ns).
		:param fig: None (default). Fig reference, mainly to have the ability to interupt the experiment.
		Return:
			machine
			I
			Q
		"""
		qubit_lo = machine.octaves[0].LO_sources[1].LO_frequency

		if qubit_lo < 2E9:
			print("LO < 2GHz, abort")
			return machine, None, None

		qubit_if_sweep = qubit_freq_sweep - qubit_lo
		qubit_if_sweep = np.floor(qubit_if_sweep)
		ff_duration = machine.qubits[qubit_index].pi_length + 40

		if np.max(abs(qubit_if_sweep)) > 4000E6: # check if parameters are within hardware limit
			print("qubit if range > 4000MHz")
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
			if fig is not None:
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
			while results.is_processing():
				time.sleep(0.1)
				# Fetch results
				# I, Q, iteration = results.fetch_all()
				# I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
				# Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)
				# progress bar
				# progress_counter(iteration, n_avg, start_time=results.get_start_time())

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[res_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[res_index].readout_pulse_length)

			return machine, I, Q

	def res_freq_analysis(self,res_freq_sweep, I, Q):
		"""
		analysis for the 1D resonator spectroscopy experiment, and find the resonance frequency by looking for the minima
		Args:
			res_freq_sweep: resonator frequency array
			I: corresponding signal I array
			Q: corresponding signal Q array
		Return:
			 res_freq_sweep[idx]: the resonance frequency
		"""
		sig_amp = np.sqrt(I ** 2 + Q ** 2)
		idx = np.argmin(sig_amp)  # find minimum
		return res_freq_sweep[idx]
