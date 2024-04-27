from quam import QuAM

machine = QuAM("quam_bootstrap_state.json")

qubits_name = ["q" + str(i) for i in range(1, 7)]
qubits_connectivity = [(3, 4, 3, "con1")]
qubits_frequencies = [6.289e9,6.107e9,5.982e9,5.768e9,5.567e9,5.442e9]
resonators_name = ["r" + str(i) for i in range(1, 7)]
resonators_connectivity = [(1, 2, 1, "con1")]
resonators_frequencies = [7.130e9, 6.999e9, 6.875e9, 6.217e9, 6.112e9, 6.018e9]
flux_lines_name = ["flux" + str(i) for i in range(1, 7)]
flux_lines_connectivity = [(8, "con1"),(9, "con1"),(10, "con1"),(7, "con1"),(8, "con1"),(9, "con1")]
ROI = [38.0,38.0,36.0,40.0,42.0,40.0]
TWPA_freq = [6324E6,6324E6,6324E6,6730.6E6,6730.6E6,6730.6E6]
TWPA_pwr = [-10.0,-10.0,-10.0,-11.3,-11.3,-11.3]

for i in range(6):
    machine.qubits.append(
        {
            "name": qubits_name[i],
            "f_01": qubits_frequencies[i],
            "f_tls": [],
            "anharmonicity": 180e6,
            "drag_coefficient": 0.0,
            "ac_stark_detuning": 0.0,
            "x180_length": 40,  # for gaussian wave
            "x180_amp": 0.1,
            "pi_length": 40,  # for square wave
            "pi_amp": 0.025,
            "pi_length_ef": 80,
            "pi_amp_ef": 0.015,
            "pi_length_tls": [200],
            "pi_amp_tls": [0.3],
            "T1": 2500,
            "T2": 3000,
            "DC_tuning_curve": [0.0, 0.0, 0.0],
            "AC_tuning_curve": [0.0, 0.0, 0.0],
            "wiring": {
                "controller": qubits_connectivity[0][3],
                "I": qubits_connectivity[0][0],
                "Q": qubits_connectivity[0][1],
                "digital_marker": qubits_connectivity[0][2],
            },
            "hardware_parameters": {  # current TLS pulse. Always update using TLS index in experiments
                "pi_length_tls": 200,
                "pi_amp_tls": 0.3,
                "RF_output_gain": 0,
            },
        }
    )

    machine.flux_lines.append(
        {
            "name": flux_lines_name[i],
            "max_frequency_point": 0.0,
            "flux_pulse_amp": 0.25,
            "flux_pulse_length": 100,
            "iswap": {
                # will use baking for these most of the time. No need to define a separate pulse for iswap. Just store parameters here
                "length": [16],
                "level": [0.2],
            },
            "wiring": {
                "controller": flux_lines_connectivity[i][1],
                "port": flux_lines_connectivity[i][0],
                "filter": {"iir_taps": [], "fir_taps": []},
            },
            "hardware_parameters": {  # a general flux pulse. Keep amp = 0.25 to convenient scaling.
                "Z_delay": 19,
                "dc_voltage": 0.0,
            },
        },
    )

for i in range(6):
        machine.resonators.append(
        {
            "name": resonators_name[i],
            "f_readout": resonators_frequencies[i],
            "depletion_time": 10_000, # keep it, cooldown time for resonator
            "readout_pulse_amp": 0.5,
            "readout_pulse_length": 500,
            "optimal_pulse_length": 2_000, # keep it if I want non-uniform weight
            "rotation_angle": 0.0,
            "ge_threshold": 0.0,
            "tuning_curve": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "wiring": {
                "controller": resonators_connectivity[0][3],
                "I": resonators_connectivity[0][0],
                "Q": resonators_connectivity[0][1],
                "digital_marker": resonators_connectivity[0][2],
            },
            "hardware_parameters": { # readout pulse
                "time_of_flight": 304,
                "downconversion_offset_I": 0.0,
                "downconversion_offset_Q": 0.0,
                "downconversion_gain": 0,
                "RO_delay": 0,
                "RO_attenuation": [ROI[i], 10],
                "TWPA": [TWPA_freq[i], TWPA_pwr[i]], 
                "RF_output_gain": 0,
            },
        }
    )
machine._save("quam_state.json", flat_data=False)