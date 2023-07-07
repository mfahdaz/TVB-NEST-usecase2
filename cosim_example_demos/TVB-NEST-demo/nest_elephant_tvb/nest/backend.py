import numpy as np

from tvb_multiscale.tvb_nest.nest_models.builders.base import NESTNetworkBuilder
from tvb_multiscale.tvb_nest.orchestrators import NESTParallelApp


def init(config, nest_network_builder, **kwargs):

    # Create a NEST App
    nest_app = NESTParallelApp(config=config,
                               synchronization_time=getattr(config, "SYNCHRONIZATION_TIME", 0.0),
                               **kwargs)

    # Set...
    if isinstance(nest_network_builder, NESTNetworkBuilder):
        # ...a NEST Network builder class instance:
        nest_app.spikeNet_builder = nest_network_builder
    else:
        # ...or, a callable function
        nest_app.spikeNet_builder_function = nest_network_builder

    nest_app.start()
    # Configure App (and CoSimulator and interface builders)
    nest_app.configure()

    # Build (CoSimulator if not built already, and interfaces)
    nest_app.build()

    # Configure App for CoSimulation
    nest_app.configure_simulation()

    return nest_app


def run_for_synchronization_time(nest_app, trans_to_nest_cosim_updates):
    """Function for cosimulating for one loop of synchronization time.
       It could be the highest level possible ENTRYPOINT for a parallel cosimulation.
       In that case, the cosimulation manager would be completely agnostic
       - of what the Apps of the different processed do,
       - including the transformation function they employ.
       The ENTRYPOINT here is just the cosimulation updates' data,
       which are "thrown over the wall" for the necessary data exchanges.
    """
    # Loop using this ENTRYPOINT:
    # NEST t -> t + Tsync
    # Simulate NEST with or without inputs
    # Input "over the wall": trans_to_nest_cosim_updates
    # Output "over the wall": nest_to_trans_cosim_updates
    nest_to_trans_cosim_updates = nest_app.run_for_synchronization_time(trans_to_nest_cosim_updates)
    return nest_app, nest_to_trans_cosim_updates


def run_cosimulation(nest_app):
    """Function for running the whole cosimulation, assuming all Apps are built and configured.
       This function shows the necessary initialization of the cosimulation.
    """

    # Store this hear safely:
    synchronization_time = nest_app.synchronization_time

    # Initial conditions of co-simulation:
    # NEST initial condition update towards TVB:
    nest_to_trans_cosim_updates = None

    # TODO: Some get the time_to_simulate from the Simulation Manager!
    # Loop for steps_to_simulate in steps of synchronization_time:
    while time_to_simulate > 0:
        nest_app.synchronization_time = time_to_simulate
        nest_to_trans_cosim_updates = run_for_synchronization_time(nest_app, nest_to_trans_cosim_updates)

    # Restore synchronization_time:
    nest_app.synchronization_time = synchronization_time

    return nest_app, nest_to_trans_cosim_updates


def final(nest_app, plot=True):
    # Plot if necessary (for the moment, necessary for the test to run):
    if plot:
        # Create a Plotter instance (or it will be created by default within each App):
        from tvb_multiscale.core.plot.plotter import Plotter
        config = nest_app.config
        config.figures.SHOW_FLAG = True
        config.figures.SAVE_FLAG = True
        config.figures.FIG_FORMAT = 'png'
        plotter = Plotter(config.figures)
        # Kwargs for NEST to plot (they will default if not provided to the Apps):
        plot_kwargs = {  # TODO: Check if time is necessary!!!
            "time": np.arange(0.0,
                              nest_app.nest_instance.GetKernelStatus("biological_time"),
                              nest_app.tvb_dt),
            "transient": config.TRANSIENT,
            "plotter": plotter,
            # Set to False for faster plotting of only mean field variables and dates,
            # apart from spikes" rasters:
            "plot_per_neuron": False}
        nest_app.plot(**plot_kwargs)
    nest_app.clean_up()
    nest_app.stop()
    return nest_app


def backend(config, plot=True):
    """Function that
       - builds all components based on user provided configurations,
       - configures them for cosimulation,
       - performs cosimulation,
       - and finalizes (plotting, cleaning up).
       This function shows how the backend should be split for the NEST process"""

    # NEST app, including NEST network and NEST input and output interfaces:
    nest_app = init(config, config.NEST_CONFIG)

    # Integration:
    # NEST initial condition update towards TVB:
    # nest_to_trans_cosim_updates = None
    # Loop with run_for_synchronization_time
    nest_app = run_cosimulation(nest_app)[0]

    # Finalize (including optional plotting), cleaning up, etc...
    nest_app = final(nest_app, plot=plot)

    # Delete app, optionally:
    del nest_app
