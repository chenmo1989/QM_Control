import quam_sdk.constructor

# The system state is a high level abstraction of the experiment written in the language of physicists
# The structure is almost completely free
state = {
    "network": {"qop_ip": "192.168.88.254", "octave1_ip": "192.168.88.179", "qop_port": 80, "octave_port": 80, "cluster_name": 'DF5', "save_dir": ""},
    "qubits": {
        {
            "name": "q0",
            "f_01": 6567349000.0,
            "f_tls": [6567349000.0],
            "anharmonicity": 180e6,
            "drag_coefficient": 0.0,
            "ac_stark_detuning": 0.0,
            "x180_length": 40,
            "x180_amp": 0.1,
            "pi_length": 40,
            "pi_amp": 0.025,
            "pi_length_ef": 80,
            "pi_amp_ef": 0.015
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
    },
    "flux_lines": {
        {
            "name": "flux0",
            "max_frequency_point": 0.0,
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
            "hardware_parameters": { # a general flux pulse. Keep amp = 0.25 to convenient scaling.
                "Z_delay": 19,
                "dc_voltage": 0.0, # add qdac initialization function, which reset this. Modify qdac function s.t. it updates this value every time.
            },
        },
    },
    "resonators": {
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
            "hardware_parameters": { # readout pulse
                "time_of_flight": 304,
                "con1_downconversion_offset_I": 0.0,
                "con1_downconversion_offset_Q": 0.0,
                "con1_downconversion_gain": 0,
                "RO_delay": 0,      
                "RO_attenuation": [32.0, 10.0],
                "TWPA": [6324E6,-10.0], 
            },
        },
    },
    "octave": {
        "LO1": {
            "LO_frequency": 5E9,
            "LO_source": "internal",
            "output_mode": "trig_normal",
            "gain": 0,    
            "time_of_flight": 304,
            "con1_downconversion_offset_I": 0.0,
            "con1_downconversion_offset_Q": 0.0,
            "con1_downconversion_gain": 0,
            "RO_delay": 0,       
        }
        "LO2": {
            "LO_frequency": 5E9,
            "LO_source": "internal",
            "output_mode": "trig_normal",
            "gain": 0,           
        }
        "LO3": {
            "LO_frequency": 5E9,
            "LO_source": "internal",
            "output_mode": "trig_normal",
            "gain": 0,           
        }
        "digital_marker1": {
                "delay": 57,
                "buffer": 18,
        },
        "digital_marker2": {
                "delay": 57,
                "buffer": 18,
        },
        "digital_marker3": {
                "delay": 57,
                "buffer": 18,
        },
    }
    "global_parameters": {
        "RO_delay": 0,
        "con1_downconversion_offset_I": 0.0,
        "con1_downconversion_offset_Q": 0.0,
        "con1_downconversion_gain": 0,
        "saturation_amp": 0.1,
        "saturation_len": 14000,
        
    },
}

# Now we use QuAM SDK to construct the Python class out of the state
quam_sdk.constructor.quamConstructor(state, flat_data=False)