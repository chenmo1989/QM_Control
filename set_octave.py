"""
set_octave.py: script for setting all the octave parameters
"""
import os
from qm.octave import QmOctaveConfig
#from qm.octave.octave_manager import ClockMode

class OctaveUnit:
    """Class for keeping track of OctavesSettings in inventory."""

    def __init__(
        self,
        name: str,
        ip: str,
        port: int = 50,
        clock: str = "Internal",
        con: str = "con1",
        port_mapping: Union[str, list] = "default",
    ):
        """Class for keeping track of OctavesSettings in inventory.

        :param name: Name of the Octave.
        :param ip: IP address of the router to which the Octave is connected.
        :param port: Port of the Octave.
        :param clock: Clock setting of the Octave. Can be "Internal", "External_10MHz", "External_100MHz" or "External_1000MHz"
        :param con: Controller to which the Octave is connected. Only used when port mapping set to default.
        :param port_mapping: Port mapping of the Octave. Default mapping is set with mapping="default", otherwise the custom mapping must be specified as a list of dictionary where each key as the following format: ('con1',  1) : ('octave1', 'I1').
        """
        self.name = name
        self.ip = ip
        self.port = port
        self.con = con
        self.clock = clock
        self.port_mapping = port_mapping

def octave_declaration(octaves: list = ()):
    """
    Initiate octave_config class, set the calibration file, add octaves info and set the port mapping between the OPX and the octaves.

    :param octaves: objects that holds the information about octave's name, the controller that is connected to this octave, octave's ip octave's port and octave's clock settings
    """
    octave_config = QmOctaveConfig()
    octave_config.set_calibration_db(os.getcwd())
    for i in range(len(octaves)):
        if octaves[i].name is None:
            raise TypeError(f"Please insert the octave name for the {i}'s octave")
        if octaves[i].con is None:
            raise TypeError(f"Please insert the controller that is connected to the {i}'s octave")
        if octaves[i].ip is None:
            raise TypeError(f"Please insert the octave ip for the {i}'s octave")
        if octaves[i].port is None:
            raise TypeError(f"Please insert the octave port for the {i}'s octave")
        octave_config.add_device_info(octaves[i].name, octaves[i].ip, octaves[i].port)
        if octaves[i].port_mapping == "default":
            octave_config.set_opx_octave_mapping([(octaves[i].con, octaves[i].name)])
        else:
            octave_config.add_opx_octave_port_mapping(octaves[i].port_mapping)

    return octave_config