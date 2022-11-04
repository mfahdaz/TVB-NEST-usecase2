import os
import numpy as np

from ..configure import configure


# FRONTEND:
# Frontend functions are used to configure the TVB <-- NEST interfaces
# utilizing the capabilities of tvb-multiscale interface builders,
# and to write those configurations to files.

def prepare_spikeNetToTVB_transformer_interface():
    config, SIM_MODE, n_regions, NEST_MODEL_BUILDERS, nest_nodes_inds, n_neurons = configure()

    from tvb_multiscale.core.interfaces.base.builders import EBRAINSSpikeNetToTVBRemoteTransformerBuilder
    spikeNet_to_tvb_trans_interface_builder = \
            EBRAINSSpikeNetToTVBRemoteTransformerBuilder(config=config)  # non opinionated builder

    from tvb_multiscale.core.utils.file_utils import load_pickled_dict
    sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
    if not os.path.isfile(sim_serial_filepath):
        from ..tvb.wilson_cowan import build_tvb_simulator
        # In order to be independent create a TVB simulator, serialize it and write it to file:
        build_tvb_simulator();
    spikeNet_to_tvb_trans_interface_builder.tvb_simulator_serialized = load_pickled_dict(sim_serial_filepath)

    # This can be used to set default tranformer and proxy models:
    spikeNet_to_tvb_trans_interface_builder.model = "RATE"  # "RATE" (or "SPIKES", "CURRENT") TVB->NEST interface
    spikeNet_to_tvb_trans_interface_builder.input_label = "spikeNetToTrans"
    spikeNet_to_tvb_trans_interface_builder.output_label = "TransToTVB"
    # If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
    # and then applied with no time delay via a single "TVB proxy node" / NEST device for each spiking region,
    # "1-to-1" TVB->NEST coupling.
    # If any other value, we need 1 "TVB proxy node" / NEST device for each TVB sender region node, and
    # large-scale coupling for spiking regions is computed in NEST,
    # taking into consideration the TVB connectome weights and delays,
    # in this "1-to-many" TVB->NEST coupling.
    spikeNet_to_tvb_trans_interface_builder.proxy_inds = nest_nodes_inds
    spikeNet_to_tvb_trans_interface_builder.N_E = n_neurons
    spikeNet_to_tvb_trans_interface_builder.N_I = n_neurons

    spikeNet_to_tvb_trans_interface_builder.output_interfaces = []
    spikeNet_to_tvb_trans_interface_builder.input_interfaces = []

    return spikeNet_to_tvb_trans_interface_builder


def configure_spikeNetToTVB_transformer_interfaces():
    spikeNet_to_TVB_transformer_interface_builder = prepare_spikeNetToTVB_transformer_interface()

    for ii, N in enumerate([spikeNet_to_TVB_transformer_interface_builder.N_E,
                            spikeNet_to_TVB_transformer_interface_builder.N_I]):
        spikeNet_to_TVB_transformer_interface_builder.input_interfaces.append(
            {  # Set the enum entry or the corresponding label name for the "transformer_model",
                # or import and set the appropriate tranformer class, e.g., ElephantSpikesHistogramRate, directly
                # options: "SPIKES", "SPIKES_TO_RATE", "SPIKES_TO_HIST", "SPIKES_TO_HIST_RATE"
                # see tvb_multiscale.core.interfaces.base.transformers.models.DefaultSpikeNetToTVBTransformers for options and related Transformer classes,
                # and tvb_multiscale.core.interfaces.base.transformers.models.DefaultSpikeNetToTVBModels for default choices
                "transformer_model": "SPIKES_TO_HIST_RATE",
                # The "scale_factor" scales the instantaneous rate coming from NEST, before setting it to TVB,
                # in our case converting the rate to a mean reate
                # and scaling it to be in the TVB model's state variable range [0.0, 1.0]
                "transformer_params": {"scale_factor": np.array([1e-4]) / N}
            })

    # This is how the user defined Spiking Network -> TVB interfaces look after configuration
    print("\ninput (TVB<-...-Transformer<-...-spikeNet update) interfaces' configurations:\n")
    display(spikeNet_to_TVB_transformer_interface_builder.input_interfaces)

    spikeNet_to_TVB_transformer_interface_builder.dump_all_interfaces()

    return spikeNet_to_TVB_transformer_interface_builder


def frontEnd_spikeNetToTVB():
    return configure_spikeNetToTVB_transformer_interfaces() # not necessary to return anything
