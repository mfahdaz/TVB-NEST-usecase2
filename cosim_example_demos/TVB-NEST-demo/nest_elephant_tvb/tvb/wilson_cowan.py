import os
import numpy as np

from ..configure import configure


# FRONTEND:
# Frontend functions are used to configure the TVB output interfaces
# utilizing the capabilities of tvb-multiscale interface builders,
# and to write those configurations to files.

def print_interfaces_model_info():
    # Options for a nonopinionated builder:
    from tvb_multiscale.core.interfaces.base.transformers.models.models import Transformers
    from tvb_multiscale.core.interfaces.base.transformers.builders import \
        DefaultTVBtoSpikeNetTransformers, DefaultSpikeNetToTVBTransformers, \
        DefaultTVBtoSpikeNetModels, DefaultSpikeNetToTVBModels
    from tvb_multiscale.tvb_nest.interfaces.builders import \
        TVBtoNESTModels, NESTInputProxyModels, DefaultTVBtoNESTModels, \
        NESTtoTVBModels, NESTOutputProxyModels, DefaultNESTtoTVBModels

    def print_enum(enum):
        print("\n", enum)
        for name, member in enum.__members__.items():
            print(name, "= ", member.value)

    print("Available input (NEST->TVB update) / output (TVB->NEST coupling) interface models:")
    print_enum(TVBtoNESTModels)
    print_enum(NESTtoTVBModels)

    print("\n\nAvailable input (spikeNet->TVB update) / output (TVB->spikeNet coupling) transformer models:")

    print_enum(DefaultTVBtoSpikeNetModels)
    print_enum(DefaultTVBtoSpikeNetTransformers)

    print_enum(DefaultSpikeNetToTVBModels)
    print_enum(DefaultSpikeNetToTVBTransformers)

    print("\n\nAvailable input (NEST->TVB update) / output (TVB->NEST coupling) proxy models:")

    print_enum(DefaultTVBtoNESTModels)
    print_enum(NESTInputProxyModels)

    print_enum(NESTOutputProxyModels)
    print_enum(DefaultNESTtoTVBModels)

    print("\n\nAll basic transformer models:")
    print_enum(Transformers)


def prepare_TVB_interface_builder(simulator=None):
    config, n_regions, NEST_MODEL_BUILDERS, nest_nodes_inds, n_neurons = configure()

    from tvb_multiscale.core.interfaces.base.io import RemoteSenders, RemoteReceivers
    from tvb_multiscale.core.interfaces.tvb.builders import TVBRemoteInterfaceBuilder
    tvb_interface_builder = TVBRemoteInterfaceBuilder(config=config)  # non opinionated builder
    # Set communicators to MPI:
    tvb_interface_builder._default_remote_sender_type = RemoteSenders.WRITER_TO_MPI
    tvb_interface_builder._default_remote_receiver_type = RemoteReceivers.READER_FROM_MPI

    if tvb_interface_builder is not None:
        if simulator is not None:
            tvb_interface_builder.tvb_cosimulator = simulator
        # This can be used to set default tranformer and proxy models:
        tvb_interface_builder.model = "RATE"  # "RATE" (or "SPIKES", "CURRENT") TVB->NEST interface
        tvb_interface_builder.input_label = "TransToTVB"
        tvb_interface_builder.output_label = "TVBtoTrans"
        # If default_coupling_mode = "TVB", large scale coupling towards spiking regions is computed in TVB
        # and then applied with no time delay via a single "TVB proxy node" / NEST device for each spiking region,
        # "1-to-1" TVB->NEST coupling.
        # If any other value, we need 1 "TVB proxy node" / NEST device for each TVB sender region node, and
        # large-scale coupling for spiking regions is computed in NEST,
        # taking into consideration the TVB connectome weights and delays,
        # in this "1-to-many" TVB->NEST coupling.
        tvb_interface_builder.default_coupling_mode = "TVB"
        tvb_interface_builder.proxy_inds = nest_nodes_inds
        # Set exclusive_nodes = True (Default) if the spiking regions substitute for the TVB ones:
        tvb_interface_builder.exclusive_nodes = True

        tvb_interface_builder.output_interfaces = []
        tvb_interface_builder.input_interfaces = []

    return tvb_interface_builder, nest_nodes_inds


def configure_TVB_interfaces(simulator=None):
    tvb_interface_builder, nest_nodes_inds = prepare_TVB_interface_builder(simulator=simulator)

    # or setting a nonopinionated builder:

    # This is a user defined TVB -> Spiking Network interface configuration:
    tvb_interface_builder.output_interfaces = \
        [{'voi': np.array(["E"]),  # TVB state variable to get data from
          # --------------- Arguments that can default if not given by the user:------------------------------
          'model': 'RATE',  # This can be used to set default tranformer and proxy models
          'coupling_mode': 'TVB',  # or "spikeNet", "NEST", etc
          'proxy_inds': nest_nodes_inds  # TVB proxy region nodes' indices
          }
         ]

    # These are user defined Spiking Network -> TVB interfaces configurations:
    for pop, sv in zip(["E", "I"], ["E", "I"]):
        tvb_interface_builder.input_interfaces.append(
            {'voi': np.array([sv]),
             'proxy_inds': nest_nodes_inds
             }
        )

    # This is how the user defined TVB -> Spiking Network interface looks after configuration
    print("\noutput (TVB-> coupling) interfaces' configurations:\n")
    display(tvb_interface_builder.output_interfaces)

    # This is how the user defined Spiking Network -> TVB interfaces look after configuration
    print("\ninput (TVB<- update) interfaces' configurations:\n")
    display(tvb_interface_builder.input_interfaces)

    tvb_interface_builder.dump_all_interfaces()

    return tvb_interface_builder


def frontEnd_TVB():
    # This function will configure the TVB <-> Transformer interfaces' builder
    # and write the configurations to files
    return configure_TVB_interfaces()  # not necessary to return anything


# BACKEND:
# Backend functions will be run upon runtime to build:
# - the TVB cosimulator, based on a user provided script (build_tvb_simulator.py),
# - and the TVB interfaces based on the configurations written to files by the FRONTEND.

# This would run on TVB only before creating any multiscale cosimulation interface connections.
def build_tvb_simulator():
    config, n_regions = configure()[:2]

    from tvb_multiscale.core.tvb.cosimulator.models.wilson_cowan_constraint import WilsonCowan

    # Create a TVB simulator and set all desired inputs
    # (connectivity, model, surface, stimuli etc)
    # We choose all defaults in this example
    # -----------------------------------Wilson Cowan oscillatory regime--------------------------------
    model_params = {
        "r_e": np.array([0.0]),
        "r_i": np.array([0.0]),
        "k_e": np.array([1.0]),
        "k_i": np.array([1.0]),
        "tau_e": np.array([10.0]),
        "tau_i": np.array([10.0]),
        "c_ee": np.array([10.0]),
        "c_ei": np.array([6.0]),
        "c_ie": np.array([10.0]),
        "c_ii": np.array([1.0]),
        "alpha_e": np.array([1.2]),
        "alpha_i": np.array([2.0]),
        "a_e": np.array([1.0]),
        "a_i": np.array([1.0]),
        "b_e": np.array([0.0]),
        "b_i": np.array([0.0]),
        "c_e": np.array([1.0]),
        "c_i": np.array([1.0]),
        "theta_e": np.array([2.0]),
        "theta_i": np.array([3.5]),
        "P": np.array([0.5]),
        "Q": np.array([0.0])
    }

    # -----------------------------------Build cosimulator manually--------------------------------
    from tvb_multiscale.core.tvb.cosimulator.cosimulator_parallel import CoSimulatorMPI

    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator.integrators import HeunStochastic
    from tvb.simulator.monitors import Raw  # , Bold, EEG

    simulator = CoSimulatorMPI()

    simulator.model = WilsonCowan(**model_params)

    simulator.integrator = HeunStochastic()
    simulator.integrator.dt = 0.1
    simulator.integrator.noise.nsig = np.array([config.DEFAULT_NSIG, config.DEFAULT_NSIG])  # 0.001

    # Load connectivity
    # config.DEFAULT_CONNECTIVITY_ZIP = "/home/docker/packages/tvb_data/tvb_data/mouse/allen_2mm/ConnectivityAllen2mm.zip"
    connectivity = Connectivity.from_file(config.DEFAULT_CONNECTIVITY_ZIP)

    # -------------- Pick a minimal brain of only the first n_regions regions: ----------------
    n_regions = 4
    connectivity.number_of_regions = n_regions
    connectivity.region_labels = connectivity.region_labels[:n_regions]
    connectivity.centres = connectivity.centres[:n_regions]
    connectivity.areas = connectivity.areas[:n_regions]
    connectivity.orientations = connectivity.orientations[:n_regions]
    connectivity.hemispheres = connectivity.hemispheres[:n_regions]
    connectivity.cortical = connectivity.cortical[:n_regions]
    connectivity.weights = connectivity.weights[:n_regions][:, :n_regions]
    connectivity.tract_lengths = connectivity.tract_lengths[:n_regions][:, :n_regions]
    # Remove diagonal self-connections:
    np.fill_diagonal(connectivity.weights, 0.0)
    # -----------------------------------------------------------------------------------------

    # Normalize connectivity weights
    connectivity.weights = connectivity.scaled_weights(mode="region")
    connectivity.weights /= np.percentile(connectivity.weights, 99)
    # connectivity.weights[connectivity.weights > 1.0] = 1.0

    # connectivity.tract_lengths = np.maximum(connectivity.speed * simulator.integrator.dt,
    #                                         connectivity.tract_lengths)

    connectivity.configure()

    simulator.connectivity = connectivity

    simulator.initial_conditions = np.zeros((1, 2, connectivity.number_of_regions, 1))

    mon_raw = Raw(period=1.0)  # ms
    simulator.monitors = (mon_raw,)

    simulator.configure()

    # # -----------------------------------Or use the CoSimulator builder--------------------------------
    # from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorSerialBuilder

    # simulator_builder = CoSimulatorSerialBuilder()
    # simulator_builder.config = config
    # simulator_builder.model = WilsonCowan()
    # simulator_builder.model_params = model_params
    # simulator_builder.initial_conditions = np.zeros((1, 1, 1, 1))

    # # simulator_builder.configure()
    # simulator_builder.print_summary_info_details(recursive=1)

    # simulator = simulator_builder.build()

    # simulator.print_summary_info_details(recursive=1)

    # Serializing TVB cosimulator is necessary for parallel cosimulation:
    from tvb_multiscale.core.utils.file_utils import dump_pickled_dict
    from tvb_multiscale.core.tvb.cosimulator.cosimulator_serialization import serialize_tvb_cosimulator
    sim_serial_filepath = os.path.join(config.out.FOLDER_RES, "tvb_serial_cosimulator.pkl")
    simulator._preconfigure_synchronization_time()
    sim_serial = serialize_tvb_cosimulator(simulator)
    display(sim_serial)

    # Dumping the serialized TVB cosimulator to a file will be necessary for parallel cosimulation.
    dump_pickled_dict(sim_serial, sim_serial_filepath)

    simulator.configure()

    return simulator
