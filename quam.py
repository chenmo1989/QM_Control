# QuAM class automatically generated using QuAM SDK (ver 0.11.0)
# open source code and documentation is available at
# https://github.com/entropy-lab/quam-sdk

from typing import List, Union
import sys
import os
from quam_sdk.classes import QuamComponent, quam_data, quam_tags



__all__ = ["QuAM"]


class _add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass



@quam_data
class Network(QuamComponent):
    qop_ip: str
    octave1_ip: str
    qop_port: int
    octave_port: int
    cluster_name: str
    save_dir: str
    


@quam_data
class Wiring(QuamComponent):
    controller: str
    I: int
    Q: int
    digital_marker: int
    


@quam_data
class Hardware_parameters(QuamComponent):
    pi_length_tls: int
    pi_amp_tls: float
    


@quam_data
class Qubit(QuamComponent):
    name: str
    f_01: float
    f_tls: List[Union[str, int, float, bool, list]]
    anharmonicity: float
    drag_coefficient: float
    ac_stark_detuning: float
    x180_length: int
    x180_amp: float
    pi_length: int
    pi_amp: float
    pi_length_ef: int
    pi_amp_ef: float
    pi_length_tls: List[Union[str, int, float, bool, list]]
    pi_amp_tls: List[Union[str, int, float, bool, list]]
    T1: int
    T2: int
    DC_tuning_curve: List[Union[str, int, float, bool, list]]
    AC_tuning_curve: List[Union[str, int, float, bool, list]]
    wiring: Wiring
    hardware_parameters: Hardware_parameters
    


@quam_data
class Iswap(QuamComponent):
    length: List[Union[str, int, float, bool, list]]
    level: List[Union[str, int, float, bool, list]]
    


@quam_data
class Filter(QuamComponent):
    iir_taps: List[Union[str, int, float, bool, list]]
    fir_taps: List[Union[str, int, float, bool, list]]
    


@quam_data
class Wiring2(QuamComponent):
    controller: str
    port: int
    filter: Filter
    


@quam_data
class Hardware_parameters2(QuamComponent):
    Z_delay: int
    dc_voltage: float
    


@quam_data
class Flux_line(QuamComponent):
    name: str
    max_frequency_point: float
    flux_pulse_amp: float
    flux_pulse_length: int
    iswap: Iswap
    wiring: Wiring2
    hardware_parameters: Hardware_parameters2
    


@quam_data
class Wiring3(QuamComponent):
    controller: str
    I: int
    Q: int
    digital_marker: int
    


@quam_data
class Hardware_parameters3(QuamComponent):
    time_of_flight: int
    downconversion_offset_I: float
    downconversion_offset_Q: float
    downconversion_gain: int
    RO_delay: int
    RO_attenuation: List[Union[str, int, float, bool, list]]
    TWPA: List[Union[str, int, float, bool, list]]
    


@quam_data
class Resonator(QuamComponent):
    name: str
    f_readout: float
    depletion_time: int
    readout_pulse_amp: float
    readout_pulse_length: int
    optimal_pulse_length: int
    rotation_angle: float
    ge_threshold: float
    tuning_curve: List[Union[str, int, float, bool, list]]
    wiring: Wiring3
    hardware_parameters: Hardware_parameters3
    


@quam_data
class Digital_marker(QuamComponent):
    delay: int
    buffer: int
    


@quam_data
class LO_source(QuamComponent):
    name: str
    LO_frequency: float
    LO_source: str
    output_mode: str
    gain: int
    digital_marker: Digital_marker
    


@quam_data
class Octave(QuamComponent):
    name: str
    time_of_flight: int
    downconversion_offset_I: float
    downconversion_offset_Q: float
    downconversion_gain: int
    RO_delay: int
    LO_sources: List[LO_source]
    


@quam_data
class Global_parameters(QuamComponent):
    saturation_amp: float
    saturation_len: int
    


@quam_data
class QuAM(QuamComponent):
    qubits: List[Qubit]
    flux_lines: List[Flux_line]
    resonators: List[Resonator]
    octaves: List[Octave]
    network: Network
    global_parameters: Global_parameters
    

