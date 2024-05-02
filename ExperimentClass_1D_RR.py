class EH_RR: # sub
	"""
	class in ExperimentHandle, for Readout Resonator (RR) related 1D experiments
	Methods:
		update_tPath
		update_str_datetime
		time_of_flight(self, qubit_index, n_avg, cd_time, tPath = None, f_str_datetime = None, simulate_flag = False, simulation_len = 1000)
		rr_freq(self, res_freq_sweep, qubit_index, n_avg, cd_time, tPath = None, f_str_datetime = None, simulate_flag = False, simulation_len = 1000)
	"""
	def __init__(self, ref_to_set_octave, ref_to_set_Labber, ref_to_datalogs):
		self.set_octave = ref_to_set_octave
		self.set_Labber = ref_to_set_Labber
		self.datalogs = ref_to_datalogs

	def time_of_flight(self, machine, n_avg = 5E3, cd_time = 20E3, simulate_flag = False, simulation_len = 1000):
		"""
		time of flight 1D experiment
		this experiment calibrates
			1. the time delay between when the RO tone is sent, and when it is received by the adc/digitizer.
				Note it could be different for different qubit, due to resonator Q. Although I do not calibrate for individual qubits.
			2. the dc offset for I, Q readout signal.
			3. if we need to adjust the attenuation of RO on the output side, s.t. the signal fully span the +-0.5V adc range, but does not exceed it (and be cutoff)

		Args:
		:param machine:
		:param qubit_index:
		:param n_avg: repetition of expeirment
		:param cd_time: cooldown time between subsequent experiments
		:param simulate_flag: True-run simulation; False (default)-run experiment.
		:param simulation_len: Length of the sequence to simulate. In clock cycles (4ns).
		:param machine: None (default), will read from quam_state.json
		Return:
			machine
			adc1
			adc2
			adc1_single_run
			adc2_single_run
		"""

		

		with program() as raw_trace_prog:
			n = declare(int)
			adc_st = declare_stream(adc_trace=True)
			n_st = declare_stream()

			with for_(n, 0, n < n_avg, n + 1):
				reset_phase(machine.resonators[qubit_index].name)
				measure("readout", machine.resonators[qubit_index].name, adc_st)
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				save(n,n_st)
			with stream_processing():
				# Will save average:
				adc_st.input1().average().save("adc1")
				adc_st.input2().average().save("adc2")
				# # Will save only last run:
				adc_st.input1().save("adc1_single_run")
				adc_st.input2().save("adc2_single_run")
				n_st.save('iteration')

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")

		if simulate_flag:
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, raw_trace_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(raw_trace_prog)
			res_handles = job.result_handles
			res_handles.wait_for_all_values()
			adc1 = u.raw2volts(res_handles.get("adc1").fetch_all())
			adc2 = u.raw2volts(res_handles.get("adc2").fetch_all())
			adc1_single_run = u.raw2volts(res_handles.get("adc1_single_run").fetch_all())
			adc2_single_run = u.raw2volts(res_handles.get("adc2_single_run").fetch_all())

			# save data
			exp_name = 'time_of_flight'
			qubit_name = 'Q' + str(qubit_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"adc1": adc1, "adc2": adc2, "adc1_single_run": adc1_single_run, "adc2_single_run": adc2_single_run})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, adc1,adc2,adc1_single_run,adc2_single_run

	def rr_freq(self, machine, res_freq_sweep, qubit_index, n_avg = 1E3, cd_time = 20E3, readout_state = 'g', simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		resonator spectroscopy experiment
		this experiment find the resonance frequency by localizing the minima in pulsed transmission signal.

		Args:
		:param machine:
		:param res_freq_sweep: 1D array for resonator frequency sweep
		:param qubit_index:
		:param n_avg: repetition of expeirment
		:param cd_time: cooldown time between subsequent experiments
		:param readout_state: 'g' (default). If 'e'/'f', readout done for |e>. If anything else, return error.
		:param simulate_flag: True-run simulation; False (default)-run experiment.
		:param simulation_len: Length of the sequence to simulate. In clock cycles (4ns).
		:param plot_flag: True (default) plot the experiment. False, do not plot.
		Return:
			machine
			res_freq_sweep
			sig_amp
		"""

		

		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		res_if_sweep = res_freq_sweep - res_lo
		res_if_sweep = np.round(res_if_sweep)

		if np.max(abs(res_if_sweep)) > 400E6: # check if parameters are within hardware limit
			print("res if range > 400MHz")
			return machine, None

		with program() as rr_freq_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()
			df = declare(int)

			with for_(n, 0, n < n_avg, n+1):
				with for_(*from_array(df,res_if_sweep)):
					if readout_state == 'g':
						pass
					elif readout_state == 'e':
						play("pi", machine.qubits[qubit_index].name)
						align()
					elif readout_state == 'f':
						play("pi", machine.qubits[qubit_index].name)
						play("pi_ef", machine.qubits[qubit_index].name)
						align()
					else:
						print("readout state does not exist")
						return
					update_frequency(machine.resonators[qubit_index].name, df)
					readout_avg_macro(machine.resonators[qubit_index].name,I,Q)
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
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
			return machine, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(rr_freq_prog)
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
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				# progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())

				if plot_flag == True:
					plt.cla()
					plt.title("Resonator spectroscopy")
					plt.plot((res_freq_sweep) / u.MHz, np.sqrt(I**2 +  Q**2), ".")
					plt.xlabel("Frequency [MHz]")
					plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [V]")

			# fetch all data after live-updating
			I, Q, iteration = results.fetch_all()
			# Convert I & Q to Volts
			I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
			Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
			sig_amp = np.sqrt(I**2 + Q**2)
			sig_phase = np.angle(I + 1j * Q)

			# save data
			exp_name = 'RR_freq'
			qubit_name = 'Q' + str(qubit_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"RR_freq": res_freq_sweep, "sig_amp": sig_amp, "sig_phase": sig_phase, "readout_state": readout_state})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset

	def rr_switch_delay(self, machine, rr_switch_delay_sweep, qubit_index, n_avg = 10E3, cd_time = 20E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		1D experiment to calibrate switch delay for the resonator.

		Args:
			machine: 
			rr_switch_delay_sweep (): in ns
			qubit_index ():
			n_avg ():
			cd_time ():
			simulate_flag ():
			simulation_len ():
			plot_flag ():
			
		Returns:
			machine
			rr_switch_delay_sweep
			sig_amp
		"""
		

		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		res_if = machine.resonators[qubit_index].f_readout - res_lo

		if abs(res_if) > 400E6: # check if parameters are within hardware limit
			print("res if > 400MHz")
			return machine, None

		with program() as rr_switch_delay_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()

			with for_(n, 0, n < n_avg, n+1):
				update_frequency(machine.resonators[qubit_index].name, res_if)
				readout_avg_macro(machine.resonators[qubit_index].name,I,Q)
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)
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
			job = qmm.simulate(config, rr_switch_delay_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			start_time = time.time()
			I_tot = []
			Q_tot = []
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]

			for delay_index, delay_value in enumerate(rr_switch_delay_sweep):
				#machine.resonators[qubit_index].digital_marker.delay = int(delay_value)
				machine = self.set_digital_delay(machine, "resonators", int(delay_value))
				
				config = build_config(machine)
				qm = qmm.open_qm(config)
				job = qm.execute(rr_switch_delay_prog)
				# Get results from QUA program
				results = fetching_tool(job, data_list=["I", "Q"], mode = "live")

				if plot_flag:
					interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
				while results.is_processing():
					# Fetch results
					time.sleep(0.1)

				I, Q = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				I_tot.append(I)
				Q_tot.append(Q)

				# progress bar
				progress_counter(delay_index, len(rr_switch_delay_sweep), start_time=start_time)

			I_tot = np.array(I_tot)
			Q_tot = np.array(Q_tot)
			sigs_qubit = I_tot + 1j * Q_tot
			sig_amp = np.abs(sigs_qubit)  # Amplitude
			sig_phase = np.angle(sigs_qubit)  # Phase

			if plot_flag == True:
				plt.cla()
				plt.title("res. switch delay")
				plt.plot(rr_switch_delay_sweep, sig_amp, ".")
				plt.xlabel("switch delay [ns]")
				plt.ylabel("Signal [V]")

			# save data
			exp_name = 'res_switch_delay'
			qubit_name = 'Q' + str(qubit_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"res_delay": rr_switch_delay_sweep, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset

	def rr_switch_buffer(self, machine, rr_switch_buffer_sweep, qubit_index, n_avg = 10E3, cd_time = 20E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		1D experiment to calibrate switch delay for the resonator.

		Args:
			machine: 
			rr_switch_buffer_sweep (): in ns, this will be added to both sides of the switch (x2), to account for the rise and fall
			qubit_index ():
			n_avg ():
			cd_time ():
			simulate_flag ():
			simulation_len ():
			plot_flag ():
			
		Returns:
			machine:
			rr_switch_buffer_sweep:
			sig_amp:
		"""
		

		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		res_if = machine.resonators[qubit_index].f_readout - res_lo

		if abs(res_if) > 400E6: # check if parameters are within hardware limit
			print("res if > 400MHz")
			return machine, None

		with program() as rr_switch_buffer_prog:
			[I,Q,n,I_st,Q_st,n_st] = declare_vars()

			with for_(n, 0, n < n_avg, n+1):
				update_frequency(machine.resonators[qubit_index].name, res_if)
				readout_avg_macro(machine.resonators[qubit_index].name,I,Q)
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)
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
			job = qmm.simulate(config, rr_switch_buffer_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None
		else:
			start_time = time.time()
			I_tot = []
			Q_tot = []
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]

			for buffer_index, buffer_value in enumerate(rr_switch_buffer_sweep):
				machine = self.set_digital_buffer(machine, "resonators", int(buffer_value))
				config = build_config(machine)
				qm = qmm.open_qm(config)
				job = qm.execute(rr_switch_buffer_prog)
				# Get results from QUA program
				results = fetching_tool(job, data_list=["I", "Q"], mode = "live")

				if plot_flag:
					interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
				while results.is_processing():
					# Fetch results
					time.sleep(0.1)

				I, Q = results.fetch_all()
				I = u.demod2volts(I, machine.resonators[qubit_index].readout_pulse_length)
				Q = u.demod2volts(Q, machine.resonators[qubit_index].readout_pulse_length)
				I_tot.append(I)
				Q_tot.append(Q)

				# progress bar
				progress_counter(buffer_index, len(rr_switch_buffer_sweep), start_time=start_time)

			I_tot = np.array(I_tot)
			Q_tot = np.array(Q_tot)
			sigs_qubit = I_tot + 1j * Q_tot
			sig_amp = np.abs(sigs_qubit)  # Amplitude
			sig_phase = np.angle(sigs_qubit)  # Phase

			if plot_flag == True:
				plt.cla()
				plt.title("res. switch buffer")
				plt.plot(rr_switch_buffer_sweep, sig_amp, ".")
				plt.xlabel("switch buffer [ns]")
				plt.ylabel("Signal [V]")

			# save data
			exp_name = 'res_switch_buffer'
			qubit_name = 'Q' + str(qubit_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"res_buffer": rr_switch_buffer_sweep, "sig_amp": sig_amp, "sig_phase": sig_phase})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, expt_dataset

	def single_shot_IQ_blob(self, machine, res_freq, qubit_index, n_avg = 1E3, cd_time = 20E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):

		

		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		res_if = np.round(res_freq - res_lo)

		if abs(res_if) > 400E6: # check if parameters are within hardware limit
			print("res if > 400MHz")
			return machine, None, None, None, None

		with program() as rr_IQ_prog:
			n = declare(int)
			I_g = declare(fixed)
			Q_g = declare(fixed)
			I_g_st = declare_stream()
			Q_g_st = declare_stream()
			I_e = declare(fixed)
			Q_e = declare(fixed)
			I_e_st = declare_stream()
			Q_e_st = declare_stream()

			with for_(n, 0, n < n_avg, n+1):
				measure(
					"readout",
					machine.resonators[qubit_index].name,
					None,
					dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
					dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
				)
				save(I_g, I_g_st)
				save(Q_g, Q_g_st)
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)

				align()  # global align

				play("pi", machine.qubits[qubit_index].name)
				align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
				measure(
					"readout",
					machine.resonators[qubit_index].name,
					None,
					dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
					dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
				)

				save(I_e, I_e_st)
				save(Q_e, Q_e_st)
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)

			with stream_processing():
				I_g_st.save_all("I_g")
				Q_g_st.save_all("Q_g")
				I_e_st.save_all("I_e")
				Q_e_st.save_all("Q_e")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port = '9510', octave=octave_config, log_level = "ERROR")
		# Simulate or execute #
		if simulate_flag: # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, rr_IQ_prog, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None, None, None
		else:
			qm = qmm.open_qm(config)
			job = qm.execute(rr_IQ_prog)
			# Get results from QUA program
			res_handles = job.result_handles

			# Waits (blocks the Python console) until all results have been acquired
			res_handles.wait_for_all_values()
			# Fetch the 'I' & 'Q' points for the qubit in the ground and excited states
			Ig = res_handles.get("I_g").fetch_all()["value"]
			Qg = res_handles.get("Q_g").fetch_all()["value"]
			Ie = res_handles.get("I_e").fetch_all()["value"]
			Qe = res_handles.get("Q_e").fetch_all()["value"]

			if plot_flag == True:
				fig = plt.figure()
				ax = plt.gca()
				plt.rcParams['figure.figsize'] = [8, 4]
				plt.plot(Ig, Qg, ".", alpha=0.1, label="Ground", markersize=3)
				plt.plot(Ie, Qe, ".", alpha=0.1, label="Excited", markersize=3)
				ax.set_aspect("equal","box")
				plt.legend(["Ground", "Excited"])
				plt.xlabel("I")
				plt.ylabel("Q")
				plt.title("Original Data")
				ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5,
						   markerscale=5)
				plt.show()

			# save data
			exp_name = 'single_shot_IQ'
			qubit_name = 'Q' + str(qubit_index+1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name), {"I_g": I_g, "Q_g": Q_g, "I_e": I_e, "Q_e": Q_e})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, Ig, Qg, Ie, Qe

	def single_shot_freq_optimization(self, machine, res_freq_sweep, qubit_index, n_avg = 10E3, cd_time = 20E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
		        READOUT OPTIMISATION: FREQUENCY
		This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
		(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
		|e> state). This is done while varying the readout frequency.
		The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to
		determine the Signal-to-Noise Ratio (SNR). The readout frequency that yields the highest SNR is selected as the
		optimal choice.
		
		:param machine:
		:param res_freq_sweep:
		:param qubit_index:
		:param n_avg:
		:param cd_time:
		:param simulate_flag:
		:param simulation_len:
		:param plot_flag:
		:return:
			machine
			SNR
			res_freq_opt
		"""

		

		res_lo = machine.octaves[0].LO_sources[0].LO_frequency
		res_if_sweep = res_freq_sweep - res_lo
		res_if_sweep = np.round(res_if_sweep)

		if np.max(abs(res_if_sweep)) > 400E6:  # check if parameters are within hardware limit
			print("res if range > 400MHz")
			return machine, None, None

		with program() as ro_freq_opt:
			n = declare(int)  # QUA variable for the averaging loop
			df = declare(int)  # QUA variable for the readout IF frequency
			I_g = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |g>
			Q_g = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |g>
			Ig_st = declare_stream()
			Qg_st = declare_stream()
			I_e = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |e>
			Q_e = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |e>
			Ie_st = declare_stream()
			Qe_st = declare_stream()
			n_st = declare_stream()

			with for_(n, 0, n < n_avg, n + 1):
				with for_(*from_array(df, res_if_sweep)):
					# Update the frequency of the digital oscillator linked to the resonator element
					update_frequency(machine.resonators[qubit_index].name, df)
					# Measure the state of the resonator
					measure(
						"readout",
						machine.resonators[qubit_index].name,
						None,
						dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
						dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
					)
					# Wait for the qubit to decay to the ground state
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					# Save the 'I_e' & 'Q_e' quadratures to their respective streams
					save(I_g, Ig_st)
					save(Q_g, Qg_st)

					align()  # global align
					# Play the x180 gate to put the qubit in the excited state
					play("pi", machine.qubits[qubit_index].name)
					# Align the two elements to measure after playing the qubit pulse.
					align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
					# Measure the state of the resonator
					measure(
						"readout",
						machine.resonators[qubit_index].name,
						None,
						dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
						dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
					)
					# Wait for the qubit to decay to the ground state
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					# Save the 'I_e' & 'Q_e' quadratures to their respective streams
					save(I_e, Ie_st)
					save(Q_e, Qe_st)
				# Save the averaging iteration to get the progress bar
				save(n, n_st)

			with stream_processing():
				n_st.save("iteration")
				# mean values
				Ig_st.buffer(len(res_if_sweep)).average().save("Ig_avg")
				Qg_st.buffer(len(res_if_sweep)).average().save("Qg_avg")
				Ie_st.buffer(len(res_if_sweep)).average().save("Ie_avg")
				Qe_st.buffer(len(res_if_sweep)).average().save("Qe_avg")
				# variances to get the SNR
				(
						((Ig_st.buffer(len(res_if_sweep)) * Ig_st.buffer(len(res_if_sweep))).average())
						- (Ig_st.buffer(len(res_if_sweep)).average() * Ig_st.buffer(len(res_if_sweep)).average())
				).save("Ig_var")
				(
						((Qg_st.buffer(len(res_if_sweep)) * Qg_st.buffer(len(res_if_sweep))).average())
						- (Qg_st.buffer(len(res_if_sweep)).average() * Qg_st.buffer(len(res_if_sweep)).average())
				).save("Qg_var")
				(
						((Ie_st.buffer(len(res_if_sweep)) * Ie_st.buffer(len(res_if_sweep))).average())
						- (Ie_st.buffer(len(res_if_sweep)).average() * Ie_st.buffer(len(res_if_sweep)).average())
				).save("Ie_var")
				(
						((Qe_st.buffer(len(res_if_sweep)) * Qe_st.buffer(len(res_if_sweep))).average())
						- (Qe_st.buffer(len(res_if_sweep)).average() * Qe_st.buffer(len(res_if_sweep)).average())
				).save("Qe_var")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port='9510', octave=octave_config, log_level="ERROR")
		# Simulate or execute #
		if simulate_flag:  # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, ro_freq_opt, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			# Open the quantum machine
			qm = qmm.open_qm(config)
			# Send the QUA program to the OPX, which compiles and executes it
			job = qm.execute(ro_freq_opt)  # execute QUA program
			# Get results from QUA program
			results = fetching_tool(
				job,
				data_list=["Ig_avg", "Qg_avg", "Ie_avg", "Qe_avg", "Ig_var", "Qg_var", "Ie_var", "Qe_var", "iteration"],
				mode="live",
			)

			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				Ig_avg, Qg_avg, Ie_avg, Qe_avg, Ig_var, Qg_var, Ie_var, Qe_var, iteration = results.fetch_all()
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				# Derive the SNR
				Z = (Ie_avg - Ig_avg) + 1j * (Qe_avg - Qg_avg)
				var = (Ig_var + Qg_var + Ie_var + Qe_var) / 4
				SNR = ((np.abs(Z)) ** 2) / (2 * var)

				# Plot results
				if plot_flag == True:
					plt.cla()
					plt.plot(res_freq_sweep / u.MHz, SNR, ".-")
					plt.title("Readout frequency optimization")
					plt.xlabel("Readout frequency [MHz]")
					plt.ylabel("SNR")
					plt.grid("on")
					plt.pause(0.1)

			print(f"The optimal readout frequency is {res_freq_sweep[np.argmax(SNR)]} Hz (SNR={max(SNR)})")

			# save data
			exp_name = 'single_shot_freq_optimization'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"RR_freq": res_freq_sweep, "SNR": SNR, "Ig_avg": Ig_avg, "Ie_avg": Ie_avg, "Ig_var": Ig_var, "Ie_var": Ie_var})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, SNR, res_freq_sweep[np.argmax(SNR)]

	def single_shot_amp_optimization(self, machine, res_amp_rel_sweep, qubit_index, n_avg = 10E3, cd_time = 20E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
				READOUT OPTIMISATION: AMPLITUDE
		The sequence consists in measuring the state of the resonator after thermalization (qubit in |g>) and after
		playing a pi pulse to the qubit (qubit in |e>) successively while sweeping the readout amplitude.
		The 'I' & 'Q' quadratures when the qubit is in |g> and |e> are extracted to derive the readout fidelity.
		The optimal readout amplitude is chosen as to maximize the readout fidelity.
		
		:param machine:
		:param res_amp_sweep:
		:param qubit_index:
		:param n_avg:
		:param cd_time:
		:param simulate_flag:
		:param simulation_len:
		:param plot_flag:
		:return:
			machine
			res_amp_sweep_abs: in V
			fidelity
			res_amp_opt
		"""

		

		if max(abs(res_amp_rel_sweep)) > 2.0:
			print("some rel amps > 2.0, removed from experiment run")
			res_amp_rel_sweep = res_amp_rel_sweep[abs(res_amp_rel_sweep) < 2.0]

		readout_amp = machine.resonators[qubit_index].readout_pulse_amp
		res_amp_abs_sweep = readout_amp * res_amp_rel_sweep
		if max(abs(res_amp_abs_sweep)) > 0.5:
			print("some abs amps > 0.5, removed from experiment run")
			res_amp_rel_sweep = res_amp_rel_sweep[abs(res_amp_abs_sweep) < 0.5]

		with program() as ro_amp_opt:
			n = declare(int)  # QUA variable for the number of runs
			counter = declare(int, value=0)  # Counter for the progress bar
			a = declare(fixed)  # QUA variable for the readout amplitude
			I_g = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |g>
			Q_g = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |g>
			Ig_st = declare_stream()
			Qg_st = declare_stream()
			I_e = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |e>
			Q_e = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |e>
			Ie_st = declare_stream()
			Qe_st = declare_stream()
			n_st = declare_stream()

			with for_(*from_array(a, res_amp_rel_sweep)):
				save(counter, n_st)
				with for_(n, 0, n < n_avg, n + 1):
					# Measure the state of the resonator
					measure(
						"readout" * amp(a),
						machine.resonators[qubit_index].name,
						None,
						dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
						dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
					)
					# Wait for the qubit to decay to the ground state
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					# Save the 'I_e' & 'Q_e' quadratures to their respective streams
					save(I_g, Ig_st)
					save(Q_g, Qg_st)

					align()  # global align
					# Play the x180 gate to put the qubit in the excited state
					play("pi", machine.qubits[qubit_index].name)
					# Align the two elements to measure after playing the qubit pulse.
					align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
					# Measure the state of the resonator
					measure(
						"readout" * amp(a),
						machine.resonators[qubit_index].name,
						None,
						dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
						dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
					)
					# Wait for the qubit to decay to the ground state
					wait(cd_time * u.ns, machine.resonators[qubit_index].name)
					# Save the 'I_e' & 'Q_e' quadratures to their respective streams
					save(I_e, Ie_st)
					save(Q_e, Qe_st)
				# Save the counter to get the progress bar
				assign(counter, counter + 1)

			with stream_processing():
				# mean values
				Ig_st.buffer(n_avg).buffer(len(res_amp_rel_sweep)).save("I_g")
				Qg_st.buffer(n_avg).buffer(len(res_amp_rel_sweep)).save("Q_g")
				Ie_st.buffer(n_avg).buffer(len(res_amp_rel_sweep)).save("I_e")
				Qe_st.buffer(n_avg).buffer(len(res_amp_rel_sweep)).save("Q_e")
				n_st.save("iteration")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		config = build_config(machine)
		qmm = QuantumMachinesManager(machine.network.qop_ip, port='9510', octave=octave_config, log_level="ERROR")
		# Simulate or execute #
		if simulate_flag:  # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, ro_amp_opt, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None, None
		else:
			# Open the quantum machine
			qm = qmm.open_qm(config)
			# Send the QUA program to the OPX, which compiles and executes it
			job = qm.execute(ro_amp_opt)  # execute QUA program
			# Get results from QUA program
			results = fetching_tool(job, data_list=["iteration"], mode="live")
			# Get progress counter to monitor runtime of the program
			while results.is_processing():
				# Fetch results
				iteration = results.fetch_all()
				# Progress bar
				progress_counter(iteration[0], len(res_amp_rel_sweep), start_time=results.get_start_time())

			# Fetch the results at the end
			results = fetching_tool(job, data_list=["I_g", "Q_g", "I_e", "Q_e"])
			I_g, Q_g, I_e, Q_e = results.fetch_all()

			# Process the data
			fidelity_vec = []
			for i in range(len(res_amp_rel_sweep)):
				angle, threshold, fidelity, gg, ge, eg, ee = self.two_state_discriminator(I_g[i], Q_g[i], I_e[i], Q_e[i], plot_flag = False, print_flag = False)
				fidelity_vec.append(fidelity)

			# Plot the data
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				plt.plot(res_amp_rel_sweep * readout_amp, fidelity_vec, ".-")
				plt.title("Readout amplitude optimization")
				plt.xlabel("Readout amplitude [V]")
				plt.ylabel("Readout fidelity [%]")

			res_amp_opt = readout_amp * res_amp_rel_sweep[np.argmax(fidelity_vec)]
			print(
				f"The optimal readout amplitude is {res_amp_opt / u.mV:.3f} mV (Fidelity={max(fidelity_vec):.1f}%)"
			)

			# save data
			exp_name = 'single_shot_amp_optimization'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"RR_amp": res_amp_rel_sweep * readout_amp, "fidelity": fidelity_vec})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, res_amp_rel_sweep * readout_amp, fidelity_vec, res_amp_opt

	def single_shot_duration_optimization(self, machine, readout_len, ringdown_len, division_length , qubit_index, n_avg = 10E3, cd_time = 20E3, simulate_flag = False, simulation_len = 1000, plot_flag = True):
		"""
				READOUT OPTIMISATION: DURATION
		This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
		(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
		|e> state). The "demod.accumulated" method is employed to assess the state of the resonator over varying durations.
		Reference: (https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Guides/features/?h=accumulated#accumulated-demodulation)
		The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to determine
		the Signal-to-Noise Ratio (SNR). The readout duration that offers the highest SNR is identified as the optimal choice.
		Note: To observe the resonator's behavior during ringdown, the integration weights length should exceed the readout_pulse length.
		
		:param machine:
		:param readout_len: Readout pulse duration, something much longer than what I typically use, like 2us. In ns
		:param ringdown_len: integration time after readout pulse to observe the ringdown of the resonator, could be 0. In ns
		:param division_length : Size of each demodulation slice in clock cycles
		:param qubit_index:
		:param n_avg:
		:param cd_time:
		:param simulate_flag:
		:param simulation_len:
		:param plot_flag:
		:return:
			machine
			x_plot: in ns, different readout duration
			SNR
			opt_readout_length
		"""

		

		def update_readout_length(new_readout_length, ringdown_length):
			config["pulses"][f"readout_pulse_q{qubit_index}" ]["length"] = new_readout_length
			config["integration_weights"][f"cosine_weights{qubit_index}"] = {
				"cosine": [(1.0, new_readout_length + ringdown_length)],
				"sine": [(0.0, new_readout_length + ringdown_length)],
			}
			config["integration_weights"][f"sine_weights{qubit_index}"] = {
				"cosine": [(0.0, new_readout_length + ringdown_length)],
				"sine": [(1.0, new_readout_length + ringdown_length)],
			}
			config["integration_weights"][f"minus_sine_weights{qubit_index}"] = {
				"cosine": [(0.0, new_readout_length + ringdown_length)],
				"sine": [(-1.0, new_readout_length + ringdown_length)],
			}


		###################
		# The QUA program #
		###################
		# Set maximum readout duration for this scan and update the configuration accordingly
		config = build_config(machine)
		update_readout_length(readout_len * u.ns, ringdown_len * u.ns)
		# Set the accumulated demod parameters
		number_of_divisions = int((readout_len + ringdown_len) / (4 * division_length))
		print("Integration weights chunk-size length in clock cycles:", division_length)
		print("The readout has been sliced in the following number of divisions", number_of_divisions)

		# Time axis for the plots at the end
		x_plot = np.arange(division_length * 4, readout_len + ringdown_len + 1, division_length * 4)

		with program() as ro_duration_opt:
			n = declare(int)
			II = declare(fixed, size=number_of_divisions)
			IQ = declare(fixed, size=number_of_divisions)
			QI = declare(fixed, size=number_of_divisions)
			QQ = declare(fixed, size=number_of_divisions)
			I = declare(fixed, size=number_of_divisions)
			Q = declare(fixed, size=number_of_divisions)
			ind = declare(int)

			n_st = declare_stream()
			Ig_st = declare_stream()
			Qg_st = declare_stream()
			Ie_st = declare_stream()
			Qe_st = declare_stream()

			with for_(n, 0, n < n_avg, n + 1):
				# Measure the ground state of the resonator
				# With demod.accumulated, the results are QUA vectors with 1 point for each accumulated chunk
				measure(
					"readout",
					machine.resonators[qubit_index].name,
					None,
					demod.accumulated("cos", II, division_length, "out1"),
					demod.accumulated("sin", IQ, division_length, "out2"),
					demod.accumulated("minus_sin", QI, division_length, "out1"),
					demod.accumulated("cos", QQ, division_length, "out2"),
				)
				# Save the QUA vectors to their corresponding streams
				with for_(ind, 0, ind < number_of_divisions, ind + 1):
					assign(I[ind], II[ind] + IQ[ind])
					save(I[ind], Ig_st)
					assign(Q[ind], QQ[ind] + QI[ind])
					save(Q[ind], Qg_st)
				# Wait for the qubit to decay to the ground state
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)

				align()

				# Measure the excited state.
				# With demod.accumulated, the results are QUA vectors with 1 point for each accumulated chunk
				play("pi", machine.qubits[qubit_index].name)
				align()
				measure(
					"readout",
					machine.resonators[qubit_index].name,
					None,
					demod.accumulated("cos", II, division_length, "out1"),
					demod.accumulated("sin", IQ, division_length, "out2"),
					demod.accumulated("minus_sin", QI, division_length, "out1"),
					demod.accumulated("cos", QQ, division_length, "out2"),
				)
				# Save the QUA vectors to their corresponding streams
				with for_(ind, 0, ind < number_of_divisions, ind + 1):
					assign(I[ind], II[ind] + IQ[ind])
					save(I[ind], Ie_st)
					assign(Q[ind], QQ[ind] + QI[ind])
					save(Q[ind], Qe_st)

				# Wait for the qubit to decay to the ground state
				wait(cd_time * u.ns, machine.resonators[qubit_index].name)
				# Save the averaging iteration to get the progress bar
				save(n, n_st)

			with stream_processing():
				n_st.save("iteration")
				# mean values
				Ig_st.buffer(number_of_divisions).average().save("Ig_avg")
				Qg_st.buffer(number_of_divisions).average().save("Qg_avg")
				Ie_st.buffer(number_of_divisions).average().save("Ie_avg")
				Qe_st.buffer(number_of_divisions).average().save("Qe_avg")
				# variances
				(
						((Ig_st.buffer(number_of_divisions) * Ig_st.buffer(number_of_divisions)).average())
						- (Ig_st.buffer(number_of_divisions).average() * Ig_st.buffer(number_of_divisions).average())
				).save("Ig_var")
				(
						((Qg_st.buffer(number_of_divisions) * Qg_st.buffer(number_of_divisions)).average())
						- (Qg_st.buffer(number_of_divisions).average() * Qg_st.buffer(number_of_divisions).average())
				).save("Qg_var")
				(
						((Ie_st.buffer(number_of_divisions) * Ie_st.buffer(number_of_divisions)).average())
						- (Ie_st.buffer(number_of_divisions).average() * Ie_st.buffer(number_of_divisions).average())
				).save("Ie_var")
				(
						((Qe_st.buffer(number_of_divisions) * Qe_st.buffer(number_of_divisions)).average())
						- (Qe_st.buffer(number_of_divisions).average() * Qe_st.buffer(number_of_divisions).average())
				).save("Qe_var")

		#####################################
		#  Open Communication with the QOP  #
		#####################################
		qmm = QuantumMachinesManager(machine.network.qop_ip, port='9510', octave=octave_config, log_level="ERROR")
		# Simulate or execute #
		if simulate_flag:  # simulation is useful to see the sequence, especially the timing (clock cycle vs ns)
			simulation_config = SimulationConfig(duration=simulation_len)
			job = qmm.simulate(config, ro_duration_opt, simulation_config)
			job.get_simulated_samples().con1.plot()
			return machine, None, None
		else:
			# Open the quantum machine
			qm = qmm.open_qm(config)
			# Send the QUA program to the OPX, which compiles and executes it
			job = qm.execute(ro_duration_opt)  # execute QUA program
			# Get results from QUA program
			results = fetching_tool(
				job,
				data_list=["Ig_avg", "Qg_avg", "Ie_avg", "Qe_avg", "Ig_var", "Qg_var", "Ie_var", "Qe_var", "iteration"],
				mode="live",
			)
			# Live plotting
			if plot_flag == True:
				fig = plt.figure()
				plt.rcParams['figure.figsize'] = [8, 4]
				interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

			while results.is_processing():
				# Fetch results
				Ig_avg, Qg_avg, Ie_avg, Qe_avg, Ig_var, Qg_var, Ie_var, Qe_var, iteration = results.fetch_all()
				# Progress bar
				progress_counter(iteration, n_avg, start_time=results.get_start_time())
				# Derive the SNR
				ground_trace = Ig_avg + 1j * Qg_avg
				excited_trace = Ie_avg + 1j * Qe_avg
				var = (Ie_var + Qe_var + Ig_var + Qg_var) / 4
				SNR = (np.abs(excited_trace - ground_trace) ** 2) / (2 * var)

				# Plot results
				plt.subplot(221)
				plt.cla()
				plt.plot(x_plot, ground_trace.real, label="ground")
				plt.plot(x_plot, excited_trace.real, label="excited")
				plt.xlabel("Readout duration [ns]")
				plt.ylabel("demodulated traces [a.u.]")
				plt.title("Real part")
				plt.legend()

				plt.subplot(222)
				plt.cla()
				plt.plot(x_plot, ground_trace.imag, label="ground")
				plt.plot(x_plot, excited_trace.imag, label="excited")
				plt.xlabel("Readout duration [ns]")
				plt.title("Imaginary part")
				plt.legend()

				plt.subplot(212)
				plt.cla()
				plt.plot(x_plot, SNR, ".-")
				plt.xlabel("Readout duration [ns]")
				plt.ylabel("SNR")
				plt.title("SNR")
				plt.pause(0.1)
				plt.tight_layout()
			# Get the optimal readout length in ns
			opt_readout_length = int(np.round(np.argmax(SNR) * division_length / 4.0) * 4 * 4)
			print(f"The optimal readout length is {opt_readout_length} ns (SNR={max(SNR)})")

			# save data
			exp_name = 'single_shot_duration_optimization'
			qubit_name = 'Q' + str(qubit_index + 1)
			f_str = qubit_name + '-' + exp_name + '-' + f_str_datetime
			file_name = f_str + '.mat'
			json_name = f_str + '_state.json'
			savemat(os.path.join(tPath, file_name),
					{"RR_duration": x_plot, "SNR": SNR, "Ig": Ig_avg, "Ie": Ie_avg, "Qg": Qg_avg, "Qe": Qe_avg})
			machine._save(os.path.join(tPath, json_name), flat_data=False)

			return machine, x_plot, SNR, opt_readout_length


	# these are functions required for the single shot optimization
	def two_state_discriminator(self, Ig, Qg, Ie, Qe, plot_flag = True, print_flag = True):
		"""
		Given two blobs in the IQ plane representing two states, finds the optimal threshold to discriminate between them
		and calculates the fidelity. Also returns the angle in which the data needs to be rotated in order to have all the
		information in the `I` (`X`) axis.

		.. note::
			This function assumes that there are only two blobs in the IQ plane representing two states (ground and excited)
			Unexpected output will be returned in other cases.


		:param float Ig: A vector containing the `I` quadrature of data points in the ground state
		:param float Qg: A vector containing the `Q` quadrature of data points in the ground state
		:param float Ie: A vector containing the `I` quadrature of data points in the excited state
		:param float Qe: A vector containing the `Q` quadrature of data points in the excited state
		:param bool plot_flag: When true (default), plot the results
		:param bool print_flag: When true (default), print the results
		:returns: A tuple of (angle, threshold, fidelity, gg, ge, eg, ee).
			angle - The angle (in radians) in which the IQ plane has to be rotated in order to have all the information in
				the `I` axis.
			threshold - The threshold in the rotated `I` axis. The excited state will be when the `I` is larger (>) than
				the threshold.
			fidelity - The fidelity for discriminating the states.
			gg - The matrix element indicating a state prepared in the ground state and measured in the ground state.
			ge - The matrix element indicating a state prepared in the ground state and measured in the excited state.
			eg - The matrix element indicating a state prepared in the excited state and measured in the ground state.
			ee - The matrix element indicating a state prepared in the excited state and measured in the excited state.
		"""

		# Condition to have the Q equal for both states:
		angle = np.arctan2(np.mean(Qe) - np.mean(Qg), np.mean(Ig) - np.mean(Ie))
		C = np.cos(angle)
		S = np.sin(angle)
		# Condition for having e > Ig
		if np.mean((Ig - Ie) * C - (Qg - Qe) * S) > 0:
			angle += np.pi
			C = np.cos(angle)
			S = np.sin(angle)

		Ig_rotated = Ig * C - Qg * S
		Qg_rotated = Ig * S + Qg * C

		Ie_rotated = Ie * C - Qe * S
		Qe_rotated = Ie * S + Qe * C

		fit = minimize(
			self._false_detections,0.5 * (np.mean(Ig_rotated) + np.mean(Ie_rotated)),(Ig_rotated, Ie_rotated),method="Nelder-Mead",)
		threshold = fit.x[0]

		gg = np.sum(Ig_rotated < threshold) / len(Ig_rotated)
		ge = np.sum(Ig_rotated > threshold) / len(Ig_rotated)
		eg = np.sum(Ie_rotated < threshold) / len(Ie_rotated)
		ee = np.sum(Ie_rotated > threshold) / len(Ie_rotated)

		fidelity = 100 * (gg + ee) / 2

		if print_flag == True:
			# print out the confusion matrix
			print(
				f"""
			Fidelity Matrix:
			-----------------
			| {gg:.3f} | {ge:.3f} |
			----------------
			| {eg:.3f} | {ee:.3f} |
			-----------------
			IQ plane rotated by: {180 / np.pi * angle:.1f}{chr(176)}
			Threshold: {threshold:.3e}
			Fidelity: {fidelity:.1f}%
			"""
			)

		if plot_flag:
			fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
			plt.rcParams['figure.figsize'] = [9, 8]
			ax1.plot(Ig, Qg, ".", alpha=0.1, label="Ground", markersize=3)
			ax1.plot(Ie, Qe, ".", alpha=0.1, label="Excited", markersize=3)
			ax1.axis("equal")
			ax1.legend(["Ground", "Excited"])
			ax1.set_xlabel("I")
			ax1.set_ylabel("Q")
			ax1.set_title("Original Data")
			ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5,
					   markerscale=5)

			ax2.plot(Ig_rotated, Qg_rotated, ".", alpha=0.1, label="Ground", markersize=3)
			ax2.plot(Ie_rotated, Qe_rotated, ".", alpha=0.1, label="Excited", markersize=3)
			ax2.axis("equal")
			ax2.set_xlabel("I")
			ax2.set_ylabel("Q")
			ax2.set_title("Rotated Data")
			ax2.legend(["Ground", "Excited"])
			ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True, ncol=5,
					   markerscale=5)

			ax3.hist(Ig_rotated, bins=50, alpha=0.75, label="Ground")
			ax3.hist(Ie_rotated, bins=50, alpha=0.75, label="Excited")
			ax3.axvline(x=threshold, color="k", ls="--", alpha=0.5)
			text_props = dict(
				horizontalalignment="center",
				verticalalignment="center",
				transform=ax3.transAxes,
			)
			ax3.text(0.7, 0.9, f"Threshold:\n {threshold:.3e}", text_props)
			ax3.set_xlabel("I")
			ax3.set_ylabel("Counts")
			ax3.set_title("1D Histogram")

			ax4.imshow(np.array([[gg, ge], [eg, ee]]))
			ax4.set_xticks([0, 1])
			ax4.set_yticks([0, 1])
			ax4.set_xticklabels(labels=["|g>", "|e>"])
			ax4.set_yticklabels(labels=["|g>", "|e>"])
			ax4.set_ylabel("Prepared")
			ax4.set_xlabel("Measured")
			ax4.text(0, 0, f"{100 * gg:.1f}%", ha="center", va="center", color="k")
			ax4.text(1, 0, f"{100 * ge:.1f}%", ha="center", va="center", color="w")
			ax4.text(0, 1, f"{100 * eg:.1f}%", ha="center", va="center", color="w")
			ax4.text(1, 1, f"{100 * ee:.1f}%", ha="center", va="center", color="k")
			ax4.set_title("Fidelities")
			fig.tight_layout()
			fig.subplots_adjust(hspace=.5)
			plt.show()
		return angle, threshold, fidelity, gg, ge, eg, ee

	def _false_detections(self, threshold, Ig, Ie):
		if np.mean(Ig) < np.mean(Ie):
			false_detections_var = np.sum(Ig > threshold) + np.sum(Ie < threshold)
		else:
			false_detections_var = np.sum(Ig < threshold) + np.sum(Ie > threshold)
		return false_detections_var
