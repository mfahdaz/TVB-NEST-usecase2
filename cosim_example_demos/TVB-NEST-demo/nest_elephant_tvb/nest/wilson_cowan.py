import os
import numpy as np

from ..configure import configure


# FRONTEND:
# Frontend functions are used to configure the NEST output interfaces
# utilizing the capabilities of tvb-multiscale interface builders,
# and to write those configurations to files.

def prepare_spikeNet_interface_builder(nest_network=None):
    config, n_regions, NEST_MODEL_BUILDERS, nest_nodes_inds, n_neurons = configure()

    # ---------------------------- Non opinionated TVB<->NEST interface builder----------------------------
    from tvb_multiscale.tvb_nest.interfaces.builders import NESTMPIInterfaceBuilder
    spikeNet_interface_builder = NESTMPIInterfaceBuilder(config=config)  # non opinionated builder

    if nest_network:
        spikeNet_interface_builder.spiking_network = nest_network

    from tvb_multiscale.core.utils.file_utils import load_pickled_dict
    sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
    if not os.path.isfile(sim_serial_filepath):
        from ..tvb.wilson_cowan import build_tvb_simulator
        # In order to be independent create a TVB simulator, serialize it and write it to file:
        build_tvb_simulator();
    spikeNet_interface_builder.tvb_simulator_serialized = load_pickled_dict(sim_serial_filepath)

    # This can be used to set default tranformer and proxy models:
    spikeNet_interface_builder.model = "RATE"  # "RATE" (or "SPIKES", "CURRENT") TVB->NEST interface
    spikeNet_interface_builder.input_label = "TransToSpikeNet"
    spikeNet_interface_builder.output_label = "spikeNetToTrans"
    # If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
    # and then applied with no time delay via a single "TVB proxy node" / NEST device for each spiking region,
    # "1-to-1" TVB->NEST coupling.
    # If any other value, we need 1 "TVB proxy node" / NEST device for each TVB sender region node, and
    # large-scale coupling for spiking regions is computed in NEST,
    # taking into consideration the TVB connectome weights and delays,
    # in this "1-to-many" TVB->NEST coupling.
    spikeNet_interface_builder.default_coupling_mode = "TVB"
    if nest_network is None:
        # Number of neurons per population to be used to compute population mean instantaneous firing rates:
        spikeNet_interface_builder.N_E = n_neurons
        spikeNet_interface_builder.N_I = n_neurons
    else:
        # Number of neurons per population to be used to compute population mean instantaneous firing rates:
        spikeNet_interface_builder.N_E = nest_network.brain_regions[nest_nodes_inds[0]]["E"].number_of_neurons
        spikeNet_interface_builder.N_I = nest_network.brain_regions[nest_nodes_inds[0]]["I"].number_of_neurons
    spikeNet_interface_builder.proxy_inds = nest_nodes_inds
    # Set exclusive_nodes = True (Default) if the spiking regions substitute for the TVB ones:
    spikeNet_interface_builder.exclusive_nodes = True

    spikeNet_interface_builder.output_interfaces = []
    spikeNet_interface_builder.input_interfaces = []

    return spikeNet_interface_builder, nest_nodes_inds


def configure_spikeNet_interfaces(nest_network=None):
    spikeNet_interface_builder, nest_nodes_inds = prepare_spikeNet_interface_builder(nest_network)

    #     # Using all default parameters for this example of an opinionated builder
    #     tvb_spikeNet_model_builder.default_config()

    # or setting a nonopinionated builder:

    # This is a user defined TVB -> Spiking Network interface configuration:
    spikeNet_interface_builder.input_interfaces = \
        [{'populations': np.array(["E"]),  # NEST populations to couple to
          # --------------- Arguments that can default if not given by the user:------------------------------
          # 'model': 'RATE',  # This can be used to set default proxy models
          'coupling_mode': 'TVB',  # or "spikeNet", "NEST", etc
          'proxy_inds': nest_nodes_inds,  # TVB proxy region nodes' indices
          # Set the enum entry or the corresponding label name for the "proxy_model",
          # or import and set the appropriate NEST proxy device class, e.g., NESTInhomogeneousPoissonGeneratorSet, directly
          # options: "RATE", "RATE_TO_SPIKES", SPIKES", "PARROT_SPIKES" or CURRENT"
          # see tvb_multiscale.tvb_nest.interfaces.io.NESTInputProxyModels for options and related NESTDevice classes,
          # and tvb_multiscale.tvb_nest.interfaces.io.DefaultTVBtoNESTModels for the default choices
          'proxy_model': "RATE",
          'spiking_proxy_inds': nest_nodes_inds  # Same as "proxy_inds" for this kind of interface
          }
         ]

    # These are user defined Spiking Network -> TVB interfaces configurations:
    for pop in ["E", "I"]:
        spikeNet_interface_builder.output_interfaces.append(
            {'populations': np.array([pop]),
             'proxy_inds': nest_nodes_inds,
             # --------------- Arguments that can default if not given by the user:------------------------------
             # Set the enum entry or the corresponding label name for the "proxy_model",
             # or import and set the appropriate NEST proxy device class, e.g., NESTSpikeRecorderMeanSet, directly
             # options "SPIKES" (i.e., spikes per neuron), "SPIKES_MEAN", "SPIKES_TOTAL"
             # (the last two are identical for the moment returning all populations spikes together)
             # see tvb_multiscale.tvb_nest.interfaces.io.NESTOutputProxyModels for options and related NESTDevice classes,
             # and tvb_multiscale.tvb_nest.interfaces.io.DefaultNESTtoTVBModels for the default choices
             'proxy_model': "SPIKES_MEAN",
             }
        )

    # This is how the user defined TVB -> Spiking Network interface looks after configuration
    print("\noutput (NEST -> coupling) interfaces' configurations:\n")
    display(spikeNet_interface_builder.output_interfaces)

    # This is how the user defined Spiking Network -> TVB interfaces look after configuration
    print("\ninput (NEST <- update) interfaces' configurations:\n")
    display(spikeNet_interface_builder.input_interfaces)

    spikeNet_interface_builder.dump_all_interfaces()

    return spikeNet_interface_builder


def frontEnd_spikeNet():
    # This function configure the NEST <-> interfaces
    # and write the configurations to files
    return configure_spikeNet_interfaces() # not necessary to return anything


# BACKEND:
# Backend functions will be run upon runtime to build:
# - the NEST network model and simulator, based on a user provided script (build_nest_network.py),
# - and the NEST interfaces based on the configurations written to files by the FRONTEND.

# This would run on NEST only before creating any multiscale cosimulation interface connections.
# Here it is assumed that the TVB simulator is already created and we can get some of its attributes,
# either by directly accessing it, or via serialization.
def build_nest_network(nest=None):
    config, n_regions, NEST_MODEL_BUILDERS, nest_nodes_inds, n_neurons = configure()

    # Build a NEST network model with the corresponding builder
    from tvb_multiscale.tvb_nest.nest_models.builders.nest_factory import load_nest, configure_nest_kernel

    # Load serialized TVB simulator from a file...:
    from tvb_multiscale.core.utils.file_utils import load_pickled_dict
    sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
    if not os.path.isfile(sim_serial_filepath):
        from ..tvb.wilson_cowan import build_tvb_simulator
        # In order to be independent create a TVB simulator, serialize it and write it to file:
        build_tvb_simulator();
    sim_serial = load_pickled_dict(sim_serial_filepath)

    if nest is None:
        # Load NEST and use defaults to configure its kernel:
        nest = configure_nest_kernel(load_nest(config=config), config)

    # Local (i.e. within brain region) neuronal populations' connections' rescaling
    # to account for the reduction (increase) to the number of neurons,
    # with respect to the originally set 100 neurons per population.
    w_n_neurons_factor = 100.0 / n_neurons

    if NEST_MODEL_BUILDERS:

        if NEST_MODEL_BUILDERS == "opinionated":

            # ------------------- Instantiating an opinionated nest network builder for this model, -------------------
            # using all default parameters for this example

            from tvb_multiscale.tvb_nest.nest_models.models.wilson_cowan import WilsonCowanBuilder

            nest_model_builder = WilsonCowanBuilder(sim_serial,  # simulator,
                                                    spiking_nodes_inds=nest_nodes_inds, nest_instance=nest,
                                                    config=config)

            # ... or modifying some of the builder's attributes:
            nest_model_builder.w_ee = w_n_neurons_factor * sim_serial['model.c_ee'][0]  # simulator.model.c_ie[0]
            nest_model_builder.w_ei = w_n_neurons_factor * sim_serial['model.c_ei'][0]  # simulator.model.c_ie[0]
            nest_model_builder.w_ie = -w_n_neurons_factor * sim_serial['model.c_ie'][0]  # simulator.model.c_ie[0]
            nest_model_builder.w_ii = -w_n_neurons_factor * sim_serial['model.c_ii'][0]  # simulator.model.c_ii[0]
            nest_model_builder.output_devices_record_to = "memory"  # "ascii"
            nest_model_builder.population_order = n_neurons
            nest_model_builder.tvb_to_spiking_dt_ratio = 2  # 2 NEST integration steps for 1 TVB integration step
            nest_model_builder.monitor_period = 1.0

        else:

            # ------ Alternatively, instantiating a non-opinionated nest network builder for this model, -----------------
            # ... and setting desired network description:

            from tvb_multiscale.tvb_nest.nest_models.builders.base import NESTNetworkBuilder

            nest_model_builder = NESTNetworkBuilder(sim_serial,  # simulator,
                                                    spiking_nodes_inds=nest_nodes_inds, nest_instance=nest,
                                                    config=config)
            nest_model_builder.output_devices_record_to = "memory"  # "ascii"
            nest_model_builder.population_order = n_neurons
            nest_model_builder.tvb_to_spiking_dt_ratio = 2  # 2 NEST integration steps for 1 TVB integration step
            nest_model_builder.monitor_period = 1.0

            # Set populations:
            nest_model_builder.populations = []
            for pop in ["E", "I"]:
                nest_model_builder.populations.append(
                    {"label": pop,
                     "model": config.DEFAULT_SPIKING_MODEL,  # "iaf_cond_alpha" by default
                     # ---------------- Possibly functions of spiking_nodes_inds --------------------------
                     "params": {},  # parameters for NEST neuronal model
                     "scale": 1.0,  # nest_model_builder.multiply population_order for the exact populations' size
                     # ---------------- Possibly functions of spiking_nodes_inds --------------------------
                     "nodes": None})  # None means "all" -> building this population to all spiking_nodes_inds

            # "static_synapse" by default:
            synapse_model = config.DEFAULT_CONNECTION["synapse_model"]
            # Default conn_spec: {'rule': "all_to_all", "allow_autapses": True, 'allow_multapses': True}
            conn_spec = config.DEFAULT_CONNECTION["conn_spec"]

            # Set populations' connections within brain region nodes
            nest_model_builder.populations_connections = [
                {"source": "E", "target": "E",  # E -> E This is a self-connection for population "E"
                 # ---------------- Possibly functions of spiking_nodes_inds --------------------------
                 "synapse_model": synapse_model,
                 "conn_spec": conn_spec,
                 "weight": w_n_neurons_factor * sim_serial['model.c_ee'][0],
                 # simulator.model.c_ee[0], # default = 1.0
                 "delay": 0.1,  # by default = 1 TVB time step
                 "receptor_type": 0,  # default = 0
                 # ---------------- Possibly functions of spiking_nodes_inds --------------------------
                 "nodes": None},  # None means "all" -> performing this connection to all spiking_nodes_inds
                {"source": "E", "target": "I",  # E -> I
                 "synapse_model": synapse_model,
                 "conn_spec": conn_spec,
                 "weight": w_n_neurons_factor * sim_serial['model.c_ei'][0],  # simulator.model.c_ei[0],
                 "delay": 0.1,
                 "receptor_type": 0,
                 "nodes": None},
                {"source": "I", "target": "E",  # I -> E
                 "synapse_model": synapse_model,
                 "conn_spec": conn_spec,
                 "weight": -w_n_neurons_factor * sim_serial['model.c_ie'][0],  # -simulator.model.c_ie[0],
                 "delay": 0.1,
                 "receptor_type": 0,
                 "nodes": None},
                {"source": "I", "target": "I",  # I -> I, This is a self-connection for population "I"
                 "synapse_model": synapse_model,
                 "conn_spec": conn_spec,
                 "weight": -w_n_neurons_factor * sim_serial['model.c_ii'][0],  # -simulator.model.c_ii[0],
                 "delay": 0.1,
                 "receptor_type": 0,
                 "nodes": None}
            ]

            # Set populations' connections among brain region node:
            nest_model_builder.nodes_connections = [
                {"source": "E", "target": ["E", "I"],
                 # --------- Possibly functions of (source_node_ind, target_node_ind, *args, **kwargs) -------------
                 "synapse_model": synapse_model,
                 "conn_spec": conn_spec,
                 # ...using TVB connectome weights:
                 "weight":
                     lambda source_node_ind, target_node_ind:
                     sim_serial['coupling.a'][0] * sim_serial['connectivity.weights'][
                         target_node_ind, source_node_ind],
                 # ...using TVB connectome delays
                 "delay":
                     lambda source_node_ind, target_node_ind:
                     np.maximum(sim_serial['integrator.dt'],
                                sim_serial['connectivity.delays'][target_node_ind, source_node_ind]),
                 "receptor_type": 0,
                 # --------- Possibly functions of (source_node_ind, target_node_ind, *args, **kwargs) -------------
                 "source_nodes": None,  # None means "all" -> performing this connection from all spiking_nodes_inds
                 "target_nodes": None}  # None means "all" -> performing this connection to all spiking_nodes_inds
            ]

            # Set output recorder devices:
            params_spike_recorder = config.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder"].copy()
            params_spike_recorder["record_to"] = nest_model_builder.output_devices_record_to
            params_multimeter = config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"].copy()
            params_multimeter["record_to"] = nest_model_builder.output_devices_record_to
            params_multimeter["interval"] = nest_model_builder.monitor_period
            nest_model_builder.output_devices = [
                {"model": "spike_recorder",
                 "connections": {"E": "E",  # Record spikes with label "E" from populations "E"
                                 "I": "I"},  # Record spikes with label "I" from populations "I"
                 # ---------------- Possibly functions of spiking_nodes_inds --------------------------
                 "params": params_spike_recorder,
                 # ---------------- Possibly functions of spiking_nodes_inds --------------------------
                 "nodes": None},  # None means all here -> recording from all spiking_nodes_inds
                {"model": "multimeter",
                 "connections": {"Excitatory": "E",  # Record time series with label "E_ts" from populations "E"
                                 "Inhibitory": "I"},  # Record time series with label "I_ts" from populations "I"
                 # ---------------- Possibly functions of spiking_nodes_inds --------------------------
                 "params": params_multimeter,
                 # ---------------- Possibly functions of spiking_nodes_inds --------------------------
                 "nodes": None},  # None means all here -> recording from all spiking_nodes_inds

            ]

            # Set input stimulation devices:
            nest_model_builder.input_devices = [
                {"model": "poisson_generator",
                 "connections": {"Stimulus": "E"},  # connect stimulus "Stimulus" to populations "E"
                 # ---------------- Possibly functions of spiking_nodes_inds --------------------------
                 "params": {"rate": 7000.0, "origin": 0.0, "start": nest_model_builder.spiking_dt},
                 "weights": 1.0,
                 "delays": nest_model_builder.spiking_dt,
                 "receptor_type": 0,
                 # ---------------- Possibly functions of spiking_nodes_inds --------------------------
                 "nodes": None  # None means all here -> stimulating all spiking_nodes_inds
                 }

            ]

        nest_model_builder.configure()

        nest_network = nest_model_builder.build()

    else:

        # ------------------- Construct the NEST network model manually -------------------

        from tvb_multiscale.tvb_nest.nest_models.network import NESTNetwork
        from tvb_multiscale.tvb_nest.nest_models.brain import NESTBrain
        from tvb_multiscale.tvb_nest.nest_models.region_node import NESTRegionNode
        from tvb_multiscale.tvb_nest.nest_models.population import NESTPopulation
        from tvb_multiscale.core.spiking_models.devices import DeviceSet, DeviceSets
        from tvb_multiscale.tvb_nest.nest_models.devices import NESTSpikeRecorder, NESTMultimeter
        from tvb_multiscale.tvb_nest.nest_models.devices import NESTPoissonGenerator

        # First configure NEST kernel:
        nest.SetKernelStatus({"resolution": 0.05})

        print("Building NESTNetwork...")

        # Create NEST network...
        nest_network = NESTNetwork(nest)

        # ...starting from neuronal populations located at specific brain regions...
        nest_network.brain_regions = NESTBrain()
        for node_ind in nest_nodes_inds:
            region_name = sim_serial['connectivity.region_labels'][node_ind]
            # region_name = simulator.connectivity.region_labels[node_ind]
            if region_name not in nest_network.brain_regions.keys():
                nest_network.brain_regions[region_name] = NESTRegionNode(label=region_name)
            for pop in ["E", "I"]:
                nest_network.brain_regions[region_name][pop] = \
                    NESTPopulation(nest.Create(config.DEFAULT_SPIKING_MODEL, n_neurons),
                                   # possible NEST model params as well here
                                   nest, label=pop, brain_region=region_name)
                print("\n...created: %s..." % nest_network.brain_regions[region_name][pop].summary_info())

        # "static_synapse" by default:
        synapse_model = config.DEFAULT_CONNECTION["synapse_model"]
        # Default
        conn_spec = {'rule': "all_to_all", "allow_autapses": True, 'allow_multapses': True}

        # Connecting populations...
        for src_node_ind in nest_nodes_inds:
            src_node_lbl = sim_serial['connectivity.region_labels'][src_node_ind]
            # src_node_lbl = simulator.connectivity.region_labels[src_node_ind]
            for trg_node_ind in nest_nodes_inds:
                trg_node_lbl = sim_serial['connectivity.region_labels'][trg_node_ind]
                # trg_node_lbl = simulator.connectivity.region_labels[trg_node_ind]
                if src_node_ind == trg_node_ind:
                    # ...within brain regions...:
                    for src_pop, trg_pop, w in zip(["E", "E", "I", "I"],
                                                   ["E", "I", "E", "I"],
                                                   [w_n_neurons_factor * sim_serial['model.c_ee'][0].item(),
                                                    # simulator.model.c_ee[0].item(),
                                                    w_n_neurons_factor * sim_serial['model.c_ei'][0].item(),
                                                    # simulator.model.c_ei[0].item(),
                                                    -w_n_neurons_factor * sim_serial['model.c_ie'][0].item(),
                                                    # -simulator.model.c_ie[0].item(),
                                                    -w_n_neurons_factor * sim_serial['model.c_ii'][0].item()
                                                    # -simulator.model.c_ii[0].item()
                                                    ]):
                        nest.Connect(nest_network.brain_regions[src_node_lbl][src_pop].nodes,
                                     nest_network.brain_regions[src_node_lbl][trg_pop].nodes,
                                     syn_spec={"synapse_model": synapse_model,
                                               "weight": w, "delay": 0.1, "receptor_type": 0},
                                     conn_spec=conn_spec)
                        print("\n...connected populations %s -> %s in brain region %s..."
                              % (src_pop, trg_pop, src_node_lbl))
                else:

                    # ...between brain regions...:
                    nest.Connect(nest_network.brain_regions[src_node_lbl]["E"].nodes,
                                 nest.NodeCollection(nest_network.brain_regions[trg_node_lbl]["E"].gids
                                                     + nest_network.brain_regions[trg_node_lbl]["I"].gids),
                                 syn_spec={"synapse_model": synapse_model,
                                           "weight":
                                               sim_serial['coupling.a'][0].item() *
                                               sim_serial['connectivity.weights'][
                                                   trg_node_ind, src_node_ind].item(),
                                           "delay":
                                               np.maximum(0.1,
                                                          sim_serial['connectivity.delays'][
                                                              trg_node_ind, src_node_ind].item()),
                                           "receptor_type": 0},
                                 conn_spec=conn_spec)
                    print("\n...connected populations E - %s -> [E, I] - %s..." % (src_node_lbl, trg_node_lbl))

        # Create output recorder devices:
        params_spike_recorder = config.NEST_OUTPUT_DEVICES_PARAMS_DEF["spike_recorder"].copy()
        params_spike_recorder["record_to"] = "memory"
        params_multimeter = config.NEST_OUTPUT_DEVICES_PARAMS_DEF["multimeter"].copy()
        params_multimeter["record_to"] = "memory"
        params_multimeter["interval"] = 1.0
        for pop in ["E", "I"]:
            nest_network.output_devices[pop] = DeviceSet(label=pop, model="spike_recorder")
            pop_lbl = np.where(pop == "E", "Excitatory", "Inhibitory").item()
            nest_network.output_devices[pop_lbl] = DeviceSet(label=pop_lbl, model="multimeter")
            for node_ind in nest_nodes_inds:
                region_name = sim_serial['connectivity.region_labels'][node_ind]
                # region_name = simulator.connectivity.region_labels[node_ind]

                # Create and connect population spike recorder for this region:
                nest_network.output_devices[pop][region_name] = \
                    NESTSpikeRecorder(nest.Create("spike_recorder", 1, params=params_spike_recorder),
                                      nest, model="spike_recorder", label=pop, brain_region=region_name)
                nest.Connect(nest_network.brain_regions[region_name][pop].nodes,
                             nest_network.output_devices[pop][region_name].device)
                nest_network.output_devices[pop].update()  # update DeviceSet after the new NESTDevice entry
                print("\n...created spike_recorder device for population %s in brain region %s..." % (
                    pop, region_name))

                # Create and connect population multimeter for this region:
                nest_network.output_devices[pop_lbl][region_name] = \
                    NESTMultimeter(nest.Create("multimeter", 1, params=params_multimeter),
                                   nest, model="multimeter", label=pop_lbl, brain_region=region_name)
                nest.Connect(nest_network.output_devices[pop_lbl][region_name].device,
                             nest_network.brain_regions[region_name][pop].nodes)
                nest_network.output_devices[pop_lbl].update()  # update DeviceSet after the new NESTDevice entry
                print("\n...created multimeter device for population %s in brain region %s..." % (pop, region_name))

        # Create input stimulation devices:
        nest_network.input_devices["Stimulus"] = DeviceSet(label="Stimulus", model="poisson_generator")
        nest_dt = nest.GetKernelStatus('resolution')
        for node_ind in nest_nodes_inds:
            region_name = sim_serial['connectivity.region_labels'][node_ind]
            # region_name = simulator.connectivity.region_labels[node_ind]
            # Create and connect population spike recorder for this region:
            nest_network.input_devices["Stimulus"][region_name] = \
                NESTPoissonGenerator(nest.Create("poisson_generator", 1,
                                                 params={"rate": 7000.0, "origin": 0.0, "start": nest_dt}),
                                     nest, model="poisson_generator", label="Stimulus", brain_region=region_name)
            nest.Connect(nest_network.input_devices["Stimulus"][region_name].device,
                         nest_network.brain_regions[region_name]["E"].nodes,
                         syn_spec={"weight": 1.0, "delay": nest_dt})
            nest_network.input_devices["Stimulus"].update()  # update DeviceSet after the new NESTDevice entry
            print("\n...created poisson_generator device for population E in brain region %s..." % region_name)

    # Configure NESTNetwork class:
    nest_network.configure()
    nest_network.print_summary_info_details(recursive=3, connectivity=True)

    return nest_network, nest_nodes_inds
