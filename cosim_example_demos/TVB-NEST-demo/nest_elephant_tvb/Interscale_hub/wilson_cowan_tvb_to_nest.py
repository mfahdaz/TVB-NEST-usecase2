import os
import numpy as np

from ..configure import configure


# FRONTEND:
# Frontend functions are used to configure the TVB --> NEST interfaces
# utilizing the capabilities of tvb-multiscale interface builders,
# and to write those configurations to files.

def prepare_TVBtoSpikeNet_transformer_interface():
    config, n_regions, NEST_MODEL_BUILDERS, nest_nodes_inds, n_neurons = configure()

    from tvb_multiscale.core.interfaces.base.builders import EBRAINSTVBtoSpikeNetRemoteTransformerBuilder
    tvb_to_spikeNet_trans_interface_builder = \
            EBRAINSTVBtoSpikeNetRemoteTransformerBuilder(config=config)  # non opinionated builder

    from tvb_multiscale.core.utils.file_utils import load_pickled_dict
    sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
    if not os.path.isfile(sim_serial_filepath):
        from ..tvb.wilson_cowan import build_tvb_simulator
        # In order to be independent create a TVB simulator, serialize it and write it to file:
        build_tvb_simulator();
    tvb_to_spikeNet_trans_interface_builder.tvb_simulator_serialized = load_pickled_dict(sim_serial_filepath)

    # This can be used to set default tranformer and proxy models:
    tvb_to_spikeNet_trans_interface_builder.model = "RATE"  # "RATE" (or "SPIKES", "CURRENT") TVB->NEST interface
    tvb_to_spikeNet_trans_interface_builder.input_label = "TVBtoTrans"
    tvb_to_spikeNet_trans_interface_builder.output_label = "TransToSpikeNet"
    # If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
    # and then applied with no time delay via a single "TVB proxy node" / NEST device for each spiking region,
    # "1-to-1" TVB->NEST coupling.
    # If any other value, we need 1 "TVB proxy node" / NEST device for each TVB sender region node, and
    # large-scale coupling for spiking regions is computed in NEST,
    # taking into consideration the TVB connectome weights and delays,
    # in this "1-to-many" TVB->NEST coupling.
    tvb_to_spikeNet_trans_interface_builder.proxy_inds = nest_nodes_inds
    tvb_to_spikeNet_trans_interface_builder.N_E = n_neurons
    tvb_to_spikeNet_trans_interface_builder.N_I = n_neurons

    tvb_to_spikeNet_trans_interface_builder.output_interfaces = []
    tvb_to_spikeNet_trans_interface_builder.input_interfaces = []

    return tvb_to_spikeNet_trans_interface_builder


def configure_TVBtoSpikeNet_transformer_interfaces():
    tvb_to_spikeNet_trans_interface_builder = prepare_TVBtoSpikeNet_transformer_interface()

    # or setting a nonopinionated builder:
    from tvb_multiscale.core.interfaces.tvb.interfaces import TVBtoSpikeNetModels

    # This is a user defined TVB -> Spiking Network interface configuration:
    tvb_to_spikeNet_trans_interface_builder.output_interfaces = \
        [{  # Set the enum entry or the corresponding label name for the "transformer_model",
            # or import and set the appropriate tranformer class, e.g., ScaleRate, directly
            # options: "RATE", "SPIKES", "SPIKES_SINGE_INTERACTION", "SPIKES_MULTIPLE_INTERACTION", "CURRENT"
            # see tvb_multiscale.core.interfaces.base.transformers.models.DefaultTVBtoSpikeNetTransformers for options and related Transformer classes,
            # and tvb_multiscale.core.interfaces.base.transformers.models.DefaultTVBtoSpikeNetModels for default choices
            'transformer_model': "RATE"
        }
        ]

    for interface in tvb_to_spikeNet_trans_interface_builder.output_interfaces:
        # The "scale_factor" scales the TVB state variable to convert it to an
        # instantaneous rate:
        if tvb_to_spikeNet_trans_interface_builder.model == TVBtoSpikeNetModels.SPIKES.name:
            # The "number_of_neurons" will determine how many spike trains will be generated:
            interface["transformer_params"] = \
                {"scale_factor": np.array([100]),
                 "number_of_neurons": np.array([tvb_to_spikeNet_trans_interface_builder.N_E])}
        else:  # RATE
            # Here the rate is a total rate, assuming a number of sending neurons:
            interface["transformer_params"] = {"scale_factor":
                                                   1e6 * np.array([tvb_to_spikeNet_trans_interface_builder.N_E])}

    # This is how the user defined TVB -> Spiking Network interface looks after configuration
    print("\noutput (->Transformer-> coupling) interfaces' configurations:\n")
    display(tvb_to_spikeNet_trans_interface_builder.output_interfaces)

    tvb_to_spikeNet_trans_interface_builder.dump_all_interfaces()

    return tvb_to_spikeNet_trans_interface_builder


def frontEnd_TVBtoSpikeNet():
    return configure_TVBtoSpikeNet_transformer_interfaces()  # not necessary to return anything
