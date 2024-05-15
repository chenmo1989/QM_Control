import quam_sdk.constructor

# The system state is a high level abstraction of the experiment written in the language of physicists
# The structure is almost completely free
state = {
    "network": {"qop_ip": "192.168.88.178", "octave1_ip": "192.168.88.179", "qop_port": 80, "octave_port": 80, "cluster_name": "DF5", "save_dir": ""},
    "qubits": [
        {
            "name": "q0",
            "f_01": 6379991612.0,
            "f_tls": [6379991612.0],
            "anharmonicity": 180e6,
            "drag_coefficient": 0.0,
            "ac_stark_detuning": 0.0,
            "x180_length": 40, # for gaussian wave
            "x180_amp": 0.1,
            "pi_length": 100, # for square wave
            "pi_amp": 0.025,
            "pi_length_ef": 80,
            "pi_amp_ef": 0.015,
            "pi_length_tls": [200],
            "pi_amp_tls": [0.3],
            "T1": 2500,
            "T2": 3000,
            "DC_tuning_curve": [0.0,0.0,0.0],
            "AC_tuning_curve": [0.0,0.0,0.0],
            "wiring": {
                "controller": "con1",
                "I": 3,
                "Q": 4,
                "digital_marker": 3,
            },
            "hardware_parameters": { # current TLS pulse. Always update using TLS index in experiments
                "pi_length_tls": 200,
                "pi_amp_tls": 0.3,
            },
        },
    ],
    "flux_lines": [
        {
            "name": "flux0",
            "flux_pulse_amp": 0.25,
            "flux_pulse_length": 100,
            "iswap": { # will use baking for these most of the time. No need to define a separate pulse for iswap. Just store parameters here
                "length": [16],
                "level": [0.2],
            },
            "wiring": {
                "controller": "con1",
                "port": 7,
                "filter": {"iir_taps": [], "fir_taps": []},
            },
            "Z_delay": 19,
        },
    ],
    "resonators": [
        {
            "name": "r0",
            "f_readout": 7.256e9,
            "depletion_time": 10_000, # keep it, cooldown time for resonator
            "readout_pulse_amp": 0.5,
            "readout_pulse_length": 500,
            "optimal_pulse_length": 2_000, # keep it if I want non-uniform weight
            "rotation_angle": 0.0,
            "ge_threshold": 0.0,
            "tuning_curve": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "wiring": {
                "controller": "con1",
                "I": 1,
                "Q": 2,
                "digital_marker": 1,
            },
            "time_of_flight": 304,
            "downconversion_offset_I": 0.0,
            "downconversion_offset_Q": 0.0,
            "downconversion_gain": 0,
            "RO_delay": 0,
            "RO_attenuation": [32.0, 10.0],
            "TWPA": [6324E6,-10.0],
        },
    ],
    "dc_flux": [
        {   
            "name": "dc0",
            "max_frequency_point": 0.0,
            "dc_voltage": 0.0,
        },
    ],
    "octaves": [ # here we can have multiple octaves
        {
            "name": "octave1",
            "LO_sources": [ # 3 LO sources for each octave
                {
                    "name": "LO1",
                    "LO_frequency": 5E9,
                    "LO_source": "internal",
                    "output_mode": "always_on", # can be: "always_on" / "always_off" (default)/ "triggered" / "triggered_reversed".
                    "gain": 0,
                    "digital_marker": {
                        "delay": 57,
                        "buffer": 18,
                    },
                    "input_attenuators": "OFF",
                },
                {
                    "name": "LO2",
                    "LO_frequency": 5E9,
                    "LO_source": "internal",
                    "output_mode": "always_on",
                    "gain": 0,
                    "digital_marker": {
                        "delay": 57,
                        "buffer": 18,
                    },
                    "input_attenuators": "OFF",
                },
                {
                    "name": "LO3",
                    "LO_frequency": 5E9,
                    "LO_source": "internal",
                    "output_mode": "always_on",
                    "gain": 0,
                    "digital_marker": {
                        "delay": 57,
                        "buffer": 18,
                    },
                    "input_attenuators": "OFF",
                },
            ],
        },
    ],
    "global_parameters": {
        "name": "quam_state.json",
        "saturation_amp": 0.1,
        "saturation_len": 14000,
        "downconversion_offset_I": [0.0],
        "downconversion_offset_Q": [0.0],
        "downconversion_gain": [0],
        "RO_delay": [0],
    },
}

# Now we use QuAM SDK to construct the Python class out of the state
quam_sdk.constructor.quamConstructor(state, flat_data=False)