"""
Octave configuration working for QOP222 and qm-qua==1.1.5 and newer.
"""
from quam import QuAM
from pathlib import Path
import numpy as np
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit
from set_octave import OctaveUnit, octave_declaration
#from qualang_tools.loops import from_array
#from qualang_tools.results import fetching_tool, progress_counter
#from qualang_tools.plot import interrupt_on_close

############################
# Set octave configuration #
############################
# Custom port mapping example
# port_mapping_1 = [
#     {
#         ("con1", 1): ("octave1", "I1"),
#         ("con1", 2): ("octave1", "Q1"),
#         ("con1", 3): ("octave1", "I2"),
#         ("con1", 4): ("octave1", "Q2"),
#         ("con1", 5): ("octave1", "I3"),
#         ("con1", 6): ("octave1", "Q3"),
#         ("con1", 7): ("octave1", "I4"),
#         ("con1", 8): ("octave1", "Q4"),
#         ("con1", 9): ("octave1", "I5"),
#         ("con1", 10): ("octave1", "Q5"),
#     }
# ]

#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)

##############
# non-QUA function #
##############
def datetime_format_string():
    return "%Y-%m-%d %H:%M:%S"

######################
# Network parameters #
######################
qop_ip = "192.168.88.179"  # octave IP address
cluster_name = "DF5"  # Write your cluster_name if version >= QOP220
octave_port = 80  # octave port if version < QOP220

# Path to save data
#save_dir = Path().absolute() / "QM" / "INSTALLATION" / "data"
save_dir = ""

############################
# Set octave configuration #
############################

# The Octave port is 11xxx, where xxx are the last three digits of the Octave internal IP that can be accessed from
# the OPX admin panel if you QOP version is >= QOP220. Otherwise, it is 50 for Octave1, then 51, 52 and so on.
octave_1 = OctaveUnit("octave1", qop_ip, port=octave_port, con="con1")
# octave_2 = OctaveUnit("octave2", qop_ip, port=11051, con="con1")

# Add the octaves
octaves = [octave_1]
# Configure the Octaves
octave_config = octave_declaration(octaves)

#####################
# OPX configuration #
#####################
def build_config(quam: QuAM):
    x180_I_wf = []
    x180_Q_wf = []
    x90_I_wf = []
    x90_Q_wf = []
    minus_x90_I_wf = []
    minus_x90_Q_wf = []
    y180_I_wf = []
    y180_Q_wf = []
    y90_I_wf = []
    y90_Q_wf = []
    minus_y90_I_wf = []
    minus_y90_Q_wf = []
    # No DRAG when alpha=0, it's just a gaussian.
    for i in range(len(quam.qubits)):
        # x180
        x180_wf, x180_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                quam.qubits[i].x180_amp,
                quam.qubits[i].x180_length,
                quam.qubits[i].x180_length / 5,
                quam.qubits[i].drag_coefficient,
                quam.qubits[i].anharmonicity,
                quam.qubits[i].ac_stark_detuning,
            )
        )
        x180_I_wf.append(x180_wf)
        x180_Q_wf.append(x180_der_wf)
        # x90
        x90_wf, x90_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                quam.qubits[i].x180_amp / 2,
                quam.qubits[i].x180_length,
                quam.qubits[i].x180_length / 5,
                quam.qubits[i].drag_coefficient,
                quam.qubits[i].anharmonicity,
                quam.qubits[i].ac_stark_detuning,
            )
        )
        x90_I_wf.append(x90_wf)
        x90_Q_wf.append(x90_der_wf)
        # -x90
        minus_x90_wf, minus_x90_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                -quam.qubits[i].x180_amp / 2,
                quam.qubits[i].x180_length,
                quam.qubits[i].x180_length / 5,
                quam.qubits[i].drag_coefficient,
                quam.qubits[i].anharmonicity,
                quam.qubits[i].ac_stark_detuning,
            )
        )
        minus_x90_I_wf.append(minus_x90_wf)
        minus_x90_Q_wf.append(minus_x90_der_wf)
        # y180
        y180_wf, y180_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                quam.qubits[i].x180_amp,
                quam.qubits[i].x180_length,
                quam.qubits[i].x180_length / 5,
                quam.qubits[i].drag_coefficient,
                quam.qubits[i].anharmonicity,
                quam.qubits[i].ac_stark_detuning,
            )
        )
        y180_I_wf.append((-1) * y180_der_wf)
        y180_Q_wf.append(y180_wf)
        # y90
        y90_wf, y90_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                quam.qubits[i].x180_amp / 2,
                quam.qubits[i].x180_length,
                quam.qubits[i].x180_length / 5,
                quam.qubits[i].drag_coefficient,
                quam.qubits[i].anharmonicity,
                quam.qubits[i].ac_stark_detuning,
            )
        )
        y90_I_wf.append((-1) * y90_der_wf)
        y90_Q_wf.append(y90_wf)
        # -y90
        minus_y90_wf, minus_y90_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                -quam.qubits[i].x180_amp / 2,
                quam.qubits[i].x180_length,
                quam.qubits[i].x180_length / 5,
                quam.qubits[i].drag_coefficient,
                quam.qubits[i].anharmonicity,
                quam.qubits[i].ac_stark_detuning,
            )
        )
        minus_y90_I_wf.append((-1) * minus_y90_der_wf)
        minus_y90_Q_wf.append(minus_y90_wf)

    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "analog_outputs": {
                    1: {"offset": 0.0, "delay": quam.global_parameters.RO_delay[0]}, # I readout
                    2: {"offset": 0.0, "delay": quam.global_parameters.RO_delay[0]}, # Q readout
                    3: {"offset": 0.0}, # I qubit XY
                    4: {"offset": 0.0}, # Q qubit XY
                    5: {"offset": 0.0}, # I auxiliary XY
                    6: {"offset": 0.0}, # Q auxiliary XY
                    7: {"offset": 0.0, "delay": quam.flux_lines[0].Z_delay}, 
                    8: {"offset": 0.0, "delay": quam.flux_lines[0].Z_delay}, 
                    9: {"offset": 0.0, "delay": quam.flux_lines[0].Z_delay}, 
                    10: {"offset": 0.0, "delay": quam.flux_lines[0].Z_delay},
                },
                "digital_outputs": {
                    1: {},
                    2: {},
                    3: {},
                    4: {},
                    5: {},
                    6: {},
                    7: {},
                    8: {},
                    9: {},
                    10: {},
                },
                "analog_inputs": {
                    1: {
                        "offset": quam.global_parameters.downconversion_offset_I[0],
                        "gain_db": quam.global_parameters.downconversion_gain[0],
                    }, # I from down-conversion 
                    2: {
                        "offset": quam.global_parameters.downconversion_offset_Q[0],
                        "gain_db": quam.global_parameters.downconversion_gain[0],
                    }, # Q from down-conversion
                },
            },
        },
        "elements": {
            **{
                quam.resonators[i].name: {
                    "RF_inputs": {"port": ("octave1", 1)},
                    "RF_outputs": {"port": ("octave1", 1)},
                    "intermediate_frequency":  (quam.resonators[i].f_readout - quam.octaves[0].LO_sources[0].LO_frequency),
                    "operations": {
                        "cw": "const_pulse",
                        "readout": f"readout_pulse_q{i}",
                        },
                    "time_of_flight": quam.resonators[i].time_of_flight,
                    "smearing": 0,
                    "digitalInputs": {
                        "output_switch": {
                            "port": ("con1", 1),
                            "delay": quam.octaves[0].LO_sources[0].digital_marker.delay,
                            "buffer": quam.octaves[0].LO_sources[0].digital_marker.buffer,
                        },
                    },
                }
                for i in range(len(quam.resonators))
            },
            **{
                quam.qubits[i].name: {
                    "RF_inputs": {"port": ("octave1", 2)},
                    "intermediate_frequency": (quam.qubits[i].f_01 - quam.octaves[0].LO_sources[1].LO_frequency),
                    "operations": {
                        "cw": "const_pulse",
                        "pi": f"pi_pulse{i}",
                        "pi2": f"pi_over_two_pulse{i}",
                        "pi2y": f"pi_over_two_y_pulse{i}",
                        "pi_ef": f"pi_pulse_ef{i}",
                        "pi2_ef": f"pi_over_two_pulse_ef{i}",
                        "pi_tls": f"pi_pulse_tls{i}",
                        "-pi_tls": f"pi_pulse_-x_tls{i}",
                        "pi2_tls": f"pi_over_two_pulse_tls{i}",
                        "pi2y_tls": f"pi_over_two_y_pulse_tls{i}",
                        "x180": f"x180_pulse{i}",
                        "x90": f"x90_pulse{i}",
                        "-x90": f"-x90_pulse{i}",
                        "y90": f"y90_pulse{i}",
                        "y180": f"y180_pulse{i}",
                        "-y90": f"-y90_pulse{i}",
                    },
                    "digitalInputs": {
                        "output_switch": {
                            "port": ("con1", 3),
                            "delay": quam.octaves[0].LO_sources[1].digital_marker.delay,
                            "buffer": quam.octaves[0].LO_sources[1].digital_marker.buffer,
                        },
                    },
                    
                    
                }
                for i in range(len(quam.qubits))
            },
            **{
                quam.flux_lines[i].name: {
                    "singleInput": {
                        "port": (quam.flux_lines[i].wiring.controller, quam.flux_lines[i].wiring.port),
                    },
                    "operations": {
                        "const": f"const_flux_pulse{i}",
                    },
                }
                for i in range(len(quam.flux_lines))
            },
        },
        "octaves": {
            "octave1": {
                "RF_outputs": {
                    1: {
                        "LO_frequency": quam.octaves[0].LO_sources[0].LO_frequency,
                        "LO_source": "internal",
                        "output_mode": quam.octaves[0].LO_sources[0].output_mode,  # can be: "always_on" / "always_off" (default)/ "triggered" / "triggered_reversed".
                        "gain": quam.octaves[0].LO_sources[0].gain,  # can be in the range [-20 : 0.5 : 20]dB
                        "input_attenuators": quam.octaves[0].LO_sources[0].input_attenuators, # Can be "ON" / "OFF" (default). "ON" means that the I and Q signals have a 10 dB attenuation before entering the octave's internal mixer.
                    },
                    2: {
                        "LO_frequency": quam.octaves[0].LO_sources[1].LO_frequency,
                        "LO_source": "internal",
                        "output_mode": quam.octaves[0].LO_sources[1].output_mode,
                        "gain": quam.octaves[0].LO_sources[1].gain,
                        "input_attenuators": quam.octaves[0].LO_sources[1].input_attenuators, # Can be "ON" / "OFF" (default). "ON" means that the I and Q signals have a 10 dB attenuation before entering the octave's internal mixer.
                    },
                },
                "RF_inputs": {
                    1: {
                        "LO_frequency": quam.octaves[0].LO_sources[0].LO_frequency,
                        "LO_source": "internal", # internal is the default
                        "IF_mode_I": "direct",  # can be: "direct" / "mixer" / "envelope" / "off". direct is default
                        "IF_mode_Q": "direct",
                    },
                },
                "connectivity": "con1",
            },
        },
        "pulses": {
            "const_pulse": {
                "operation": "control",
                "length": 1000,
                "waveforms": {
                    "I": "const_wf",
                    "Q": "zero_wf",
                },
                "digital_marker": "ON",
            },
            **{
                f"const_flux_pulse{i}": {
                    "operation": "control",
                    "length": quam.flux_lines[i].flux_pulse_length,
                    "waveforms": {
                        "single": f"const_flux{i}_wf",
                    },
                }
                for i in range(len(quam.flux_lines))
            },
            **{
                f"readout_pulse_q{i}": {
                    "operation": "measurement",
                    "length": quam.resonators[i].readout_pulse_length,
                    "waveforms": {
                        "I": f"readout{i}_wf",
                        "Q": "zero_wf",
                    },
                    "digital_marker": "ON",
                    "integration_weights": {
                        "cos": f"cosine_weights{i}",
                        "sin": f"sine_weights{i}",
                        "minus_sin": f"minus_sine_weights{i}",
                        "rotated_cos": f"rotated_cosine_weights{i}",
                        "rotated_sin": f"rotated_sine_weights{i}",
                        "rotated_minus_sin": f"rotated_minus_sine_weights{i}",
                        "opt_cos": f"opt_cosine_weights{i}",
                        "opt_sin": f"opt_sine_weights{i}",
                        "opt_minus_sin": f"opt_minus_sine_weights{i}",
                    },
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"pi_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].pi_length,
                    "waveforms": {
                        "I": f"pi_wf{i}",
                        "Q": "zero_wf",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_over_two_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].pi_length,
                    "waveforms": {
                        "I": f"pi_over_two_wf{i}",
                        "Q": "zero_wf",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_over_two_y_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].pi_length,
                    "waveforms": {
                        "I": "zero_wf",
                        "Q": f"pi_over_two_wf{i}",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_pulse_ef{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].pi_length_ef,
                    "waveforms": {
                        "I": f"pi_ef_wf{i}",
                        "Q": "zero_wf",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_over_two_pulse_ef{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].pi_length_ef,
                    "waveforms": {
                        "I": f"pi_over_two_ef_wf{i}",
                        "Q": "zero_wf",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_pulse_tls{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].hardware_parameters.pi_length_tls,
                    "waveforms": {
                        "I": f"pi_tls_wf{i}",
                        "Q": "zero_wf",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_pulse_-x_tls{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].hardware_parameters.pi_length_tls,
                    "waveforms": {
                        "I": f"pi_minus_x_tls_wf{i}",
                        "Q": "zero_wf",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_over_two_pulse_tls{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].hardware_parameters.pi_length_tls,
                    "waveforms": {
                        "I": f"pi_over_two_tls_wf{i}",
                        "Q": "zero_wf",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_over_two_y_pulse_tls{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].hardware_parameters.pi_length_tls,
                    "waveforms": {
                        "I": "zero_wf",
                        "Q": f"pi_over_two_tls_wf{i}",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"x90_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].x180_length,
                    "waveforms": {
                        "I": f"x90_I_wf{i}",
                        "Q": f"x90_Q_wf{i}",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"x180_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].x180_length,
                    "waveforms": {
                        "I": f"x180_I_wf{i}",
                        "Q": f"x180_Q_wf{i}",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"-x90_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].x180_length,
                    "waveforms": {
                        "I": f"minus_x90_I_wf{i}",
                        "Q": f"minus_x90_Q_wf{i}",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"y90_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].x180_length,
                    "waveforms": {
                        "I": f"y90_I_wf{i}",
                        "Q": f"y90_Q_wf{i}",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"y180_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].x180_length,
                    "waveforms": {
                        "I": f"y180_I_wf{i}",
                        "Q": f"y180_Q_wf{i}",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"-y90_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].x180_length,
                    "waveforms": {
                        "I": f"minus_y90_I_wf{i}",
                        "Q": f"minus_y90_Q_wf{i}",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.qubits))
            },
        },
        "waveforms": {
            "zero_wf": {"type": "constant", "sample": 0.0},
            "const_wf": {"type": "constant", "sample": 0.25},
            **{
                f"const_flux{i}_wf": {"type": "constant", "sample": quam.flux_lines[i].flux_pulse_amp}
                for i in range(len(quam.flux_lines))
            },
            **{
                f"readout{i}_wf": {"type": "constant", "sample": quam.resonators[i].readout_pulse_amp}
                for i in range(len(quam.resonators))
            },
            **{
                f"pi_wf{i}": {"type": "constant", "sample": quam.qubits[i].pi_amp}
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_over_two_wf{i}": {"type": "constant", "sample": quam.qubits[i].pi_amp/2}
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_ef_wf{i}": {"type": "constant", "sample": quam.qubits[i].pi_amp_ef}
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_over_two_ef_wf{i}": {"type": "constant", "sample": quam.qubits[i].pi_amp_ef/2}
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_tls_wf{i}": {"type": "constant", "sample": quam.qubits[i].hardware_parameters.pi_amp_tls}
                for i in range(len(quam.qubits))
            },
            **{
                f"pi_minus_x_tls_wf{i}": {"type": "constant", "sample": -quam.qubits[i].hardware_parameters.pi_amp_tls}
                for i in range(len(quam.qubits))
            },
            ** {
                f"pi_over_two_tls_wf{i}": {"type": "constant", "sample": quam.qubits[i].hardware_parameters.pi_amp_tls/2}
                for i in range(len(quam.qubits))
            },
            **{f"x90_I_wf{i}": {"type": "arbitrary", "samples": x90_I_wf[i].tolist()} for i in range(len(quam.qubits))},
            **{f"x90_Q_wf{i}": {"type": "arbitrary", "samples": x90_Q_wf[i].tolist()} for i in range(len(quam.qubits))},
            **{
                f"x180_I_wf{i}": {"type": "arbitrary", "samples": x180_I_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{
                f"x180_Q_wf{i}": {"type": "arbitrary", "samples": x180_Q_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{
                f"minus_x90_I_wf{i}": {"type": "arbitrary", "samples": minus_x90_I_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{
                f"minus_x90_Q_wf{i}": {"type": "arbitrary", "samples": minus_x90_Q_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{f"y90_I_wf{i}": {"type": "arbitrary", "samples": y90_I_wf[i].tolist()} for i in range(len(quam.qubits))},
            **{f"y90_Q_wf{i}": {"type": "arbitrary", "samples": y90_Q_wf[i].tolist()} for i in range(len(quam.qubits))},
            **{
                f"y180_I_wf{i}": {"type": "arbitrary", "samples": y180_I_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{
                f"y180_Q_wf{i}": {"type": "arbitrary", "samples": y180_Q_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{
                f"minus_y90_I_wf{i}": {"type": "arbitrary", "samples": minus_y90_I_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{
                f"minus_y90_Q_wf{i}": {"type": "arbitrary", "samples": minus_y90_Q_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
        },
        "digital_waveforms": {
            "ON": {"samples": [(1, 0)]},
            "OFF": {"samples": [(0, 0)]},
        },
        "integration_weights": {
            **{
                f"cosine_weights{i}": {
                    "cosine": [(1.0, quam.resonators[i].readout_pulse_length)],
                    "sine": [(0.0, quam.resonators[i].readout_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"sine_weights{i}": {
                    "cosine": [(0.0, quam.resonators[i].readout_pulse_length)],
                    "sine": [(1.0, quam.resonators[i].readout_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"minus_sine_weights{i}": {
                    "cosine": [(0.0, quam.resonators[i].readout_pulse_length)],
                    "sine": [(-1.0, quam.resonators[i].readout_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"rotated_cosine_weights{i}": {
                    "cosine": [(np.cos(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                    "sine": [(-np.sin(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"rotated_sine_weights{i}": {
                    "cosine": [(np.sin(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                    "sine": [(np.cos(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"rotated_minus_sine_weights{i}": {
                    "cosine": [(-np.sin(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                    "sine": [(-np.cos(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"opt_cosine_weights{i}": {
                    "cosine": [(1.0, quam.resonators[i].optimal_pulse_length)],
                    "sine": [(0.0, quam.resonators[i].optimal_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"opt_sine_weights{i}": {
                    "cosine": [(0.0, quam.resonators[i].optimal_pulse_length)],
                    "sine": [(1.0, quam.resonators[i].optimal_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"opt_minus_sine_weights{i}": {
                    "cosine": [(0.0, quam.resonators[i].optimal_pulse_length)],
                    "sine": [(-1.0, quam.resonators[i].optimal_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
        },
    }
    return config