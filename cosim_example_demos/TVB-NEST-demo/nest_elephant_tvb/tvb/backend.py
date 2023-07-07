from tvb_multiscale.core.tvb.cosimulator.cosimulator_builder import CoSimulatorParallelBuilder
from tvb_multiscale.core.orchestrators.tvb_app import TVBParallelApp


def init(config, tvb_cosimulator_builder, **kwargs):

    # Create a TVBApp
    tvb_app = TVBParallelApp(config=config, **kwargs)
    # Set...
    if isinstance(tvb_cosimulator_builder, CoSimulatorParallelBuilder):
        # ...a TVB CoSimulator builder class instance:
        tvb_app.cosimulator_builder = tvb_cosimulator_builder
    else:
        # ...or, a callable function
        tvb_app.cosimulator_builder_function = tvb_cosimulator_builder

    tvb_app.start()
    # Configure App (and CoSimulator and interface builders)
    tvb_app.configure()

    # Build (CoSimulator if not built already, and interfaces)
    tvb_app.build()

    # Configure App for CoSimulation
    tvb_app.configure_simulation()

    return tvb_app


def run_for_synchronization_time(tvb_app, trans_to_tvb_cosim_updates):
    """Function for cosimulating for one loop of synchronization time.
       It could be the highest level possible ENTRYPOINT for a parallel cosimulation.
       In that case, the cosimulation manager would be completely agnostic
       - of what the Apps of the different processed do,
       - including the transformation function they employ.
       The ENTRYPOINT here is just the cosimulation updates' data,
       which are "thrown over the wall" for the necessary data exchanges.
    """
    # Loop using this ENTRYPOINT:
    # TVB t -> t + Tsync
    # Simulate TVB with or without inputs
    # Input "over the wall": trans_to_tvb_cosim_updates
    # Output "over the wall": tvb_to_trans_cosim_updates
    tvb_to_trans_cosim_updates = tvb_app.run_for_synchronization_time(trans_to_tvb_cosim_updates)
    return tvb_app, tvb_to_trans_cosim_updates


def run_cosimulation(tvb_app, advance_simulation_for_delayed_monitors_output=True):
    """Function for running the whole cosimulation, assuming all Apps are built and configured.
       This function shows the necessary initialization of the cosimulation.
    """

    import time
    import numpy as np

    # Keep the following cosimulation attributes safe and easy to access:
    simulation_length = tvb_app.cosimulator.simulation_length
    synchronization_time = tvb_app.cosimulator.synchronization_time
    synchronization_n_step = tvb_app.cosimulator.synchronization_n_step  # store the configured value
    if advance_simulation_for_delayed_monitors_output:
        simulation_length += synchronization_time
    dt = tvb_app.cosimulator.integrator.dt

    # Initial conditions of co-simulation:
    # Steps left to simulate:
    remaining_steps = int(np.round(simulation_length / dt))
    # Steps already simulated:
    simulated_steps = 0
    # TVB initial condition cosimulation coupling towards NEST:
    tvb_to_trans_cosim_updates = tvb_app.tvb_init_cosim_coupling

    # Loop for steps_to_simulate in steps of synchronization_time:
    tvb_app.cosimulator._tic = time.time()
    while remaining_steps > 0:
        # Set the remaining steps as simulation time,
        # if it is less than the original synchronization time:
        tvb_app.cosimulator.synchronization_n_step = np.minimum(remaining_steps, synchronization_n_step)
        time_to_simulate = dt * tvb_app.cosimulator.synchronization_n_step
        tvb_app.cosimulator.synchronization_time = time_to_simulate
        tvb_to_trans_cosim_updates = \
            run_for_synchronization_time(tvb_app,  tvb_to_trans_cosim_updates)
        simulated_steps += tvb_app.cosimulator.n_tvb_steps_ran_since_last_synch
        tvb_app.cosimulator._log_print_progress_message(simulated_steps, simulation_length)
        remaining_steps -= tvb_app.cosimulator.n_tvb_steps_ran_since_last_synch

    # Update the simulation length of the TVB cosimulator:
    tvb_app.cosimulator.simulation_length = simulated_steps * dt  # update the configured value
    # Restore the original synchronization_time
    tvb_app.cosimulator.synchronization_n_step = synchronization_n_step
    tvb_app.cosimulator.synchronization_time = synchronization_time

    return tvb_app, tvb_to_trans_cosim_updates


def final(tvb_app, plot=True, **kwargs):
    if plot:
        tvb_app.plot(**kwargs)
    tvb_app.clean_up()
    tvb_app.stop()
    return tvb_app


def backend(config, plot=True, advance_simulation_for_delayed_monitors_output=True):
    """Function that
       - builds all components based on user provided configurations,
       - configures them for cosimulation,
       - performs cosimulation,
       - and finalizes (plotting, cleaning up).
       This function shows how the backend should be split for the TVB process"""

    # Build and configure all Apps, and their components, up to the point to start simulation:
    # TVB app, including TVB Simulator and TVB input and output interfaces.
    tvb_app = init(config, config.TVB_CONFIG)

    # Integration:
    # TVB initial condition cosimulation coupling towards NEST:
    tvb_app = run_cosimulation(tvb_app, advance_simulation_for_delayed_monitors_output)[0]

    # Get TVB results:
    results = list(tvb_app.return_tvb_results())

    # Plot if necessary (for the moment, necessary for the test to run):
    tvb_plot_kwargs = {}
    if plot:
        # Create a Plotter instance (or it will be created by default within each App):
        from tvb_multiscale.core.plot.plotter import Plotter
        config.figures.SHOW_FLAG = True
        config.figures.SAVE_FLAG = True
        config.figures.FIG_FORMAT = 'png'
        plotter = Plotter(config.figures)
        # Kwargs for TVB to plot (they will default if not provided to the Apps):
        plot_kwargs = {  # Set the transient time to be optionally removed from results:
            "transient": config.TRANSIENT,
            "plotter": plotter}

    # Finalize (including optional plotting), cleaning up, etc...
    tvb_app = final(tvb_app, plot=plot, **plot_kwargs)

    # Delete app, optionally:
    del tvb_app

    return results, config
