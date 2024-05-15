from quam import QuAM

machine = QuAM("quam_bootstrap_state.json")

dc_flux_name = ["dc" + str(i) for i in range(1, 7)]

for i in range(6):
    machine.dc_flux.append(
        {   
            "name": dc_flux_name[i],
            "max_frequency_point": 0.0,
            "dc_voltage": 0.0
        },
    )

machine._save(machine.global_parameters.name, flat_data=True)