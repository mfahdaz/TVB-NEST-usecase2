#!/usr/bin/env python
# coding: utf-8

# ## tvb-multiscale toolbox:
# 
# ### https://github.com/the-virtual-brain/tvb-multiscale
# 
# For questions use the git issue tracker, or write an e-mail to me: dionysios.perdikis@charite.de

# # TVB - NEST co-simulation 

# ## Wilson - Cowan TVB mean field model
# 
# For every region node $n\prime$ modelled as a mean-field node in TVB:
# 
# Population activity dynamics (1 excitatory and 1 inhibitory population):
# 
#  $\dot{E}_k = \dfrac{1}{\tau_e} (-E_k  + (k_e - r_e E_k) \mathcal{S}_e (\alpha_e \left( c_{ee} E_k - c_{ei} I_k  + P_k - \theta_e + \mathbf{\Gamma}(E_k, E_j, u_{kj}) + W_{\zeta}\cdot E_j + W_{\zeta}\cdot I_j\right) )) $
#  
# $
#             \dot{I}_k = \dfrac{1}{\tau_i} (-I_k  + (k_i - r_i I_k) \mathcal{S}_i (\alpha_i \left( c_{ie} E_k - c_{ee} I_k  + Q_k - \theta_i + \mathbf{\Gamma}(E_k, E_j, u_{kj}) + W_{\zeta}\cdot E_j + W_{\zeta}\cdot I_j\right) ))$
# 

# ## Spiking network model in NEST
# 
# using "iaf_cond_alpha" spiking neuronal model.

# ## TVB to NEST coupling
# TVB couples to NEST via instantaneous spike rate $ w_{TVB->NEST} * E(t) $, 
# 
# Inhomogeneous spike generator NEST devices are used as TVB "proxy" nodes and generate independent Poisson-random spike trains 
# 
# $ \left[ \sum_k \delta(t-\tau_{n\prime n}-{t_j}^k) \right]_{j \in n\prime} $
# 
# Alternatively, the spike trains are generated outside NEST using the Elephant software and inserted to NEST via spike generator devices.
# 
# 

# ## NEST to TVB update
# 
# A NEST spike detector device is used to count spike for each time step, and convert it to an instantaneous population mean rate that overrides
# 
# $ {E_{_{n}}}(t) =  \frac{\sum_j\left[ \sum_k \delta(t-\tau_n-{t_j}^k) \right]_{j \in E_n}}{N_E * dt} $ 
# 
# $ {I_{_{n}}}(t) =  \frac{\sum_j\left[ \sum_k \delta(t-\tau_n-{t_j}^k) \right]_{j \in I_n}}{N_I * dt} $
# 
# in  spikes/sec.
# 
# This update process concerns only the TVB region nodes that are simulated exclusively in NEST, as spiking networks. All the rest of TVB nodes will follow the equations of the mean field model described above.
# 

# ## Simulator loop
# 
# ### Simulating several (i.e., minimally 2) NEST time steps for every 1 TVB time step for stable integration
# 
# ### Synchronizaion every minimum delay time between the two simulators.


# # References
#
# 1 Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide, <br>
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013) <br>
#   The Virtual Brain: a simulator of primate brain network dynamics. <br>
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010) <br>
#   https://www.thevirtualbrain.org/tvb/zwei <br>
#   https://github.com/the-virtual-brain <br>
#
# 2 Ritter P, Schirner M, McIntosh AR, Jirsa VK. 2013.  <br>
#   The Virtual Brain integrates computational modeling  <br>
#   and multimodal neuroimaging. Brain Connectivity 3:121–145. <br>
#
# 3 Jordan, Jakob; Mørk, Håkon; Vennemo, Stine Brekke;   Terhorst, Dennis; Peyser, <br>
#   Alexander; Ippen, Tammo; Deepu, Rajalekshmi;   Eppler, Jochen Martin; <br>
#   van Meegen, Alexander;   Kunkel, Susanne; Sinha, Ankur; Fardet, Tanguy; Diaz, <br>
#   Sandra; Morrison, Abigail; Schenck, Wolfram; Dahmen, David;   Pronold, Jari; <br>
#   Stapmanns, Jonas;   Trensch, Guido; Spreizer, Sebastian;   Mitchell, Jessica; <br>
#   Graber, Steffen; Senk, Johanna; Linssen, Charl; Hahne, Jan; Serenko, Alexey; <br>
#   Naoumenko, Daniel; Thomson, Eric;   Kitayama, Itaru; Berns, Sebastian;   <br>
#   Plesser, Hans Ekkehard <br>
#   NEST is a simulator for spiking neural network models that focuses <br>
#   on the dynamics, size and structure of neural systems rather than on <br>
#   the exact morphology of individual neurons. <br>
#   For further information, visit http://www.nest-simulator.org. <br>
#   The release notes for this release are available at  <br>
#   https://github.com/nest/nest-simulator/releases/tag/v2.18.0 <br>


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

# Scripts specific to an example/model/use case:
# Both front and back end:
from examples.parallel.wilson_cowan.config import configure
# Frontend scripts:
from examples.parallel.tvb_nest.wilson_cowan.tvb_interface_config import configure_TVB_interfaces
from examples.parallel.tvb_nest.wilson_cowan.nest_interface_config import configure_NEST_interfaces
from examples.parallel.tvb_nest.wilson_cowan.transformers_config import \
    configure_TVBtoNEST_transformer_interfaces, configure_NESTtoTVB_transformer_interfaces
# Backend scripts:
from examples.parallel.tvb_nest.wilson_cowan.tvb_config import build_tvb_simulator
from examples.parallel.tvb_nest.wilson_cowan.nest_config import build_nest_network


def frontend(config=None):
    """Frontend code to run if the necessary configuration files haven't been made already available by the user.
       These files are:
       - tvb_serial_cosimulator.pkl: serialized TVB simulator
                                   Parameters usually needed:
                                    -- simulator.integrator.dt: the integration step of TVB
                                    -- simulator.connectivity.weights: the TVB connectome weights
                                    -- simulator.connectivity.delays: the TVB connectome delays
                                    -- simulator.synchronization_time: the synchronization time of co-simulation,
                                                                       preconfigured by default based
                                                                       on the minimum TVB connectome delay
                                    -- other possible simulator.model parameters,
                                       depending on the specific example/model/use case.
                                    -- transformers that perform integration,
                                        might need other attributes of simulator.integrator
                                    --  other parameters depending on the specific example/model/use case...
                                    So, for the moment, we use this way for making these parameters available
                                    to other processes than the TVB one.
                                    TODO: Have another way to share these configurations...
                                    ...and/or send them to the other processes at run time.
       - TVBInterfaceBuilder_interface: general configuration for TVB input/output interfaces
       - TVBInterfaceBuilder_output_interface_%id: one configuration for each TVB output interface, id=0,1,...
       - TVBInterfaceBuilder_input_interface_%id: one configuration for each TVB input interface, id=0,1,...
       - NESTInterfaceBuilder_interface: general configuration for NEST input/output interfaces
       - NESTInterfaceBuilder_output_interface_%id: one configuration for each NEST output interface, id=0,1,...
       - NESTInterfaceBuilder_input_interface_%id: one configuration for each NEST input interface, id=0,1,...
       - TVBtoSpikeNetTransformerInterfaceBuilder_interface: general configuration
                                                             for TVB to NEST transformer interfaces
       - TVBtoSpikeNetTransformerInterfaceBuilder_output_interface_%id: one configuration
                                                                        for each NEST to TVB transformer interface
       - SpikeNetToTVBTransformerInterfaceBuilder_interface: general configuration
                                                             for TVB to NEST transformer interfaces
       - SpikeNetToTVBTransformerInterfaceBuilder_input_interface_%id: one configuration
                                                                        for each NEST to TVB transformer interface
    """
    if config is None:
        from tvb_multiscale.tvb_nest.config import Config
        config = configure(config_class=Config)

    # This is a BACKEND script for creating the TVB Simulator instance, which we run here, at the FRONTEND,
    # in order to produce the serialized TVB Simulator dictionary file,
    # to be used for parametrizing the interface configurations
    simulator = build_tvb_simulator(config=config, config_class=Config)

    # FRONTEND: Configure the TVB<->NEST interfaces
    tvb_interface_builder = configure_TVB_interfaces(simulator=simulator, config=config, config_class=Config)
    nest_interface_builder = configure_NEST_interfaces(config=config, config_class=Config)
    tvb_to_nest_interface_builder = configure_TVBtoNEST_transformer_interfaces(config=config, config_class=Config)
    nest_to_tvb_interface_builder = configure_NESTtoTVB_transformer_interfaces(config=config, config_class=Config)

    # Outputs only for debugging:
    return tvb_interface_builder, nest_interface_builder, \
           tvb_to_nest_interface_builder, nest_to_tvb_interface_builder, simulator


def run_for_synchronization_time(tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app,
                                 tvb_to_trans_cosim_updates=None,
                                 nest_to_trans_cosim_updates=None  # None for t = t0
                                 ):
    """Function for cosimulating for one loop of synchronization.
       It could be the highest level possible ENTRYPOINT for a parallel cosimulation.
       In that case, the cosimulation manager would be completely agnostic
       - of what the Apps of the different processed do,
       - including the transformation function they employ.
       The ENTRYPOINT here is just the cosimulation updates' data,
       which are "thrown over the wall" for the necessary data exchanges.
    """
    # Transform inputs from NEST at time t...
    if nest_to_trans_cosim_updates is not None:
        # ...if any:
        trans_to_tvb_cosim_updates = nest_to_tvb_app.run_for_synchronization_time(nest_to_trans_cosim_updates)
    else:
        trans_to_tvb_cosim_updates = None
    # Transform inputs from TVB, at time t...
    if tvb_to_trans_cosim_updates is not None:
        # ...if any:
        trans_to_nest_cosim_updates = tvb_to_nest_app.run_for_synchronization_time(tvb_to_trans_cosim_updates)
    else:
        trans_to_nest_cosim_updates = None
    # TVB t -> t + Tsync
    # Simulate TVB with or without inputs
    tvb_to_trans_cosim_updates = tvb_app.run_for_synchronization_time(trans_to_tvb_cosim_updates)
    # NEST t -> t + Tsync
    # Simulate TVB with or without inputs
    nest_to_trans_cosim_updates = nest_app.run_for_synchronization_time(trans_to_nest_cosim_updates)
    return tvb_app.cosimulator.n_tvb_steps_ran_since_last_synch, tvb_to_trans_cosim_updates, nest_to_trans_cosim_updates


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
    dt = tvb_app.cosimulator.integrator.dt

    # Initial conditions of co-simulation:
    # Steps left to simulate:
    steps_to_simulate = int(np.round(simulation_length / dt))
    # Steps already simulated:
    simulated_steps = 0
    # TVB initial condition cosimulation coupling towards NEST:
    tvb_to_trans_cosim_updates = tvb_app.send_cosim_coupling()
    # NEST initial condition update towards TVB:
    nest_to_trans_cosim_updates = None

    # Loop for steps_to_simulate in steps of synchronization_time:
    tvb_app.cosimulator._tic = time.time()
    while steps_to_simulate - simulated_steps > 0:
        current_simulated_steps, tvb_to_trans_cosim_updates, nest_to_trans_cosim_update = \
            run_for_synchronization_time(tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app,
                                         tvb_to_trans_cosim_updates, nest_to_trans_cosim_updates)
        simulated_steps += current_simulated_steps
        tvb_app.cosimulator._log_print_progress_message(simulated_steps, simulation_length)

    # We might need to simulate a bit more, to get the last delayed results of TVB:
    if advance_simulation_for_delayed_monitors_output:
        # Run once more for synchronization steps in order to get the full delayed monitors' outputs:
        # Find how many steps we need to simulate:
        remaining_steps = \
            int(np.round((simulation_length + synchronization_time - simulated_steps * dt) / dt))
        if remaining_steps:
            remaining_time = remaining_steps * dt
            if tvb_app.verbosity:
                tvb_app._logprint("Simulating for excess time %0.3f..." % remaining_time)
            # Set the remaining steps as simulation time, assuming it is less than the original synchronization time:
            tvb_app.cosimulator.synchronization_time = remaining_time
            tvb_app.cosimulator.synchronization_n_step = remaining_steps
            nest_app.synchronization_time = remaining_time
            current_simulated_steps, tvb_to_trans_cosim_updates, nest_to_trans_cosim_update = \
                run_for_synchronization_time(tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app,
                                             tvb_to_trans_cosim_updates, nest_to_trans_cosim_update)
            simulated_steps += current_simulated_steps
            # Restore the original synchronization_time
            tvb_app.cosimulator.synchronization_time = synchronization_time
            tvb_app.cosimulator.synchronization_n_step = synchronization_n_step
            nest_app.synchronization_time = synchronization_time

    # Update the simulation length of the TVB cosimulator:
    tvb_app.cosimulator.simulation_length = simulated_steps * dt  # update the configured value

    return tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app, tvb_to_trans_cosim_updates, nest_to_trans_cosim_update


def backend(config=None, plot=True):
    """Function that
       - builds all components based on user provided configurations,
       - configures them for cosimulation,
       - performs cosimulation,
       - and finalizes (plotting, cleaning up)."""

    if config is None:
        from tvb_multiscale.tvb_nest.config import Config
        config = configure(config_class=Config)

    # Build and configure all Apps, and their components, up to the point to start simulation:
    #TVB app, including TVB Simulator and TVB input and output interfaces.
    tvb_app = tvb_init(config, build_tvb_simulator)
    # NEST app, including NEST network and NEST input and output interfaces:
    nest_app = nest_init(config, build_nest_network)
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
                           # Set the transient time to be optionally removed from results:
        tvb_plot_kwargs = {"transient": 0.1 * tvb_app.cosimulator.simulation_length,
                           "plotter": plotter}
        nest_plot_kwargs = dict(tvb_plot_kwargs)
        nest_plot_kwargs.update({"time": results[0][0],
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


def run_example(plot=True):

    # Genereate the common configuration:
    from tvb_multiscale.tvb_nest.config import Config
    config = configure(config_class=Config)

    # Run the frontend if the necessary configuration files have not been already made available by the user:
    frontend(config=None)

    # BACKEND:
    return backend(config=None, plot=True)


def test():
    import os
    import numpy as np
    from xarray import DataArray
    from tvb_multiscale.core.utils.file_utils import load_pickled_dict

    SPIKES_NUMBERS_PER_REG = [2900, 2800]

    config = run_example()[-1]

    # TVB
    tvb_ts = DataArray.from_dict(
        load_pickled_dict(
            os.path.join(config.out.FOLDER_RES, "source_ts.pkl")))

    # Time:
    time = tvb_ts.coords["Time"].values
    assert time.size == 9004
    dts = np.diff(time)
    assert np.allclose([np.mean(dts), np.min(dts), np.max(dts)], 0.1, atol=1e-06)

    # data
    assert tvb_ts.shape == (9004, 2, 68, 1)
    try:
        assert np.allclose(tvb_ts.values.squeeze().mean(axis=0).mean(axis=1),
                           np.array([0.51479334, 0.58060805]), atol=1e-06)
    except Exception as e:
        print(tvb_ts.values.squeeze().mean(axis=0).mean(axis=1))
        raise e

    # NEST data
    nest_mean_rate = DataArray.from_dict(
        load_pickled_dict(
            os.path.join(config.out.FOLDER_RES, "Mean Populations' Spikes' Rates.pkl")))
    assert nest_mean_rate.shape == (2, 2)
    try:
        assert np.allclose(nest_mean_rate.values,
                           np.array([[28.76551672, 27.65915069],
                                     [28.76551672, 27.65915069]]),
                           atol=1e-06
                           )
    except Exception as e:
        print(nest_mean_rate.values)
        raise e

    nest_spikes = load_pickled_dict(os.path.join(config.out.FOLDER_RES, "Spikes.pkl"))
    for pop, pop_spks in nest_spikes.items():
        for iR, (reg, reg_spks) in enumerate(pop_spks.items()):
            try:
                assert reg_spks.loc["senders"].size == reg_spks.loc["times"].size == SPIKES_NUMBERS_PER_REG[iR]
            except Exception as e:
                print(iR)
                print(reg_spks.loc["senders"].size)
                raise e


if __name__ == "__main__":
    import sys

    if sys.argv[-1] == "test":
        test()
    else:
        run_example()
