#!/usr/bin/env python

# # WORKFLOW:

# Imports that would be identical for any TVB<->NEST cosimulation script:
from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

# Backend cosimulation scripts agnostic to any specific example/mode/use case:
from nest_elephant_tvb.tvb.backend import init as tvb_init
from nest_elephant_tvb.nest.backend import init as nest_init
from nest_elephant_tvb.Interscale_hub.backend import tvb_to_nest_init, nest_to_tvb_init
from nest_elephant_tvb.tvb.backend import final as tvb_final
from nest_elephant_tvb.nest.backend import final as nest_final
from nest_elephant_tvb.Interscale_hub.backend import final as trans_final


def run_for_synchronization_time(tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app,
                                 tvb_to_trans_cosim_updates=None,
                                 nest_to_trans_cosim_updates=None  # None for t = t0
                                 ):
    """Function for cosimulating for one loop of synchronization time.
       It could be the highest level possible ENTRYPOINT for a parallel cosimulation.
       In that case, the cosimulation manager would be completely agnostic
       - of what the Apps of the different processed do,
       - including the transformation function they employ.
       The ENTRYPOINT here is just the cosimulation updates' data,
       which are "thrown over the wall" for the necessary data exchanges.
    """
    # Transform TVB -> NEST couplings at time t...
    if tvb_to_trans_cosim_updates is not None:
        # ...if any:
        trans_to_nest_cosim_updates = tvb_to_nest_app.run_for_synchronization_time(tvb_to_trans_cosim_updates)
    else:
        trans_to_nest_cosim_updates = None
    # Transform NEST -> TVB updates at time t...
    if nest_to_trans_cosim_updates is not None:
        # ...if any:
        trans_to_tvb_cosim_updates = nest_to_tvb_app.run_for_synchronization_time(nest_to_trans_cosim_updates)
    else:
        trans_to_tvb_cosim_updates = None
    # TVB t -> t + Tsync
    # Simulate TVB with or without inputs
    tvb_to_trans_cosim_updates = tvb_app.run_for_synchronization_time(trans_to_tvb_cosim_updates)
    # NEST t -> t + Tsync
    # Simulate TVB with or without inputs
    nest_to_trans_cosim_updates = nest_app.run_for_synchronization_time(trans_to_nest_cosim_updates)
    return tvb_to_trans_cosim_updates, nest_to_trans_cosim_updates


def run_cosimulation(tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app,
                     advance_simulation_for_delayed_monitors_output=True):
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
    # NEST initial condition update towards TVB:
    nest_to_trans_cosim_updates = None

    # Loop for steps_to_simulate in steps of synchronization_time:
    tvb_app.cosimulator._tic = time.time()
    while remaining_steps > 0:
        # Set the remaining steps as simulation time,
        # if it is less than the original synchronization time:
        tvb_app.cosimulator.synchronization_n_step = np.minimum(remaining_steps, synchronization_n_step)
        time_to_simulate = dt * tvb_app.cosimulator.synchronization_n_step
        tvb_app.cosimulator.synchronization_time = time_to_simulate
        nest_app.synchronization_time = time_to_simulate
        tvb_to_trans_cosim_updates, nest_to_trans_cosim_updates = \
            run_for_synchronization_time(tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app,
                                         tvb_to_trans_cosim_updates, nest_to_trans_cosim_updates)
        simulated_steps += tvb_app.cosimulator.n_tvb_steps_ran_since_last_synch
        tvb_app.cosimulator._log_print_progress_message(simulated_steps, simulation_length)
        remaining_steps -= tvb_app.cosimulator.n_tvb_steps_ran_since_last_synch

    # Update the simulation length of the TVB cosimulator:
    tvb_app.cosimulator.simulation_length = simulated_steps * dt  # update the configured value
    # Restore the original synchronization_time
    tvb_app.cosimulator.synchronization_n_step = synchronization_n_step
    tvb_app.cosimulator.synchronization_time = synchronization_time
    nest_app.synchronization_time = synchronization_time

    return tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app, \
           tvb_to_trans_cosim_updates, nest_to_trans_cosim_updates


def backend(config, plot=True):
    """Function that
       - builds all components based on user provided configurations,
       - configures them for cosimulation,
       - performs cosimulation,
       - and finalizes (plotting, cleaning up)."""

    # Build and configure all Apps, and their components, up to the point to start simulation:
    #TVB app, including TVB Simulator and TVB input and output interfaces.
    tvb_app = tvb_init(config, config.TVB_CONFIG)
    # NEST app, including NEST network and NEST input and output interfaces:
    nest_app = nest_init(config, config.NEST_CONFIG)
    # TVB to NEST app, including TVB to NEST interfaces and their transformers:
    tvb_to_nest_app = tvb_to_nest_init(config)
    # NEST to TVB app, including NEST to TVB interfaces and their transformers:
    nest_to_tvb_app = nest_to_tvb_init(config)

    # Run serially for this test:
    tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app, tvb_to_trans_cosim_updates, nest_to_trans_cosim_update = \
        run_cosimulation(tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app,
                         advance_simulation_for_delayed_monitors_output=True)

    # Get TVB results:
    results = list(tvb_app.return_tvb_results())

    # Plot if necessary (for the moment, necessary for the test to run):
    tvb_plot_kwargs = {}
    nest_plot_kwargs = {}
    if plot:
        # Create a Plotter instance (or it will be created by default within each App):
        from tvb_multiscale.core.plot.plotter import Plotter
        config.figures.SHOW_FLAG = True
        config.figures.SAVE_FLAG = True
        config.figures.FIG_FORMAT = 'png'
        plotter = Plotter(config.figures)
        # Kwargs for TVB and NEST to plot (they will default if not provided to the Apps):
        tvb_plot_kwargs = {# Set the transient time to be optionally removed from results:
                           "transient": 0.1 * tvb_app.cosimulator.simulation_length,
                           "plotter": plotter}
        nest_plot_kwargs = dict(tvb_plot_kwargs)
        nest_plot_kwargs.update({"time": results[0][0], # TODO: Check if time is necessary!!!
                                 # Set to False for faster plotting of only mean field variables and dates,
                                 # apart from spikes" rasters:
                                 "plot_per_neuron": False})

        # ### TVB plots and Spiking Network plots upon finalizing

    # Finalize (including optional plotting), cleaning up, etc...
    nest_to_tvb_app = trans_final(nest_to_tvb_app)
    tvb_to_nest_app = trans_final(tvb_to_nest_app)
    nest_app = nest_final(nest_app, plot=plot, **nest_plot_kwargs)
    tvb_app = tvb_final(tvb_app, plot=plot, **tvb_plot_kwargs)

    # Delete apps, optionally:
    del tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app

    return results, config
