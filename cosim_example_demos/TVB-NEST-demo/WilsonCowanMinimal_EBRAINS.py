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

from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

import numpy as np

from tvb_multiscale.tvb_nest.config import Config

from examples.parallel.wilson_cowan.config import configure
from examples.parallel.tvb_nest.wilson_cowan.tvb_config import build_tvb_simulator
from examples.parallel.tvb_nest.wilson_cowan.nest_config import build_nest_network
from examples.parallel.tvb_nest.wilson_cowan.tvb_interface_config import configure_TVB_interfaces
from examples.parallel.tvb_nest.wilson_cowan.nest_interface_config import configure_NEST_interfaces
from examples.parallel.tvb_nest.wilson_cowan.transformers_config import \
    configure_TVBtoNEST_transformer_interfaces, configure_NESTtoTVB_transformer_interfaces

from nest_elephant_tvb.tvb.backend import init as tvb_init
from nest_elephant_tvb.nest.backend import init as nest_init
from nest_elephant_tvb.Interscale_hub.backend import tvb_to_nest_init, nest_to_tvb_init
from nest_elephant_tvb.tvb.backend import final as tvb_final
from nest_elephant_tvb.nest.backend import final as nest_final
from nest_elephant_tvb.Interscale_hub.backend import final as trans_final


def run_for_synchronization_time(tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app,
                                 trans_to_tvb_cosim_updates, trans_to_nest_cosim_updates):
    # t0 -> t1
    tvb_to_trans_cosim_updates = tvb_app.run_for_synchronization_time(trans_to_tvb_cosim_updates)
    trans_to_nest_cosim_updates = tvb_to_nest_app.run_for_synchronization_time(tvb_to_trans_cosim_updates)
    # t0 -> t1
    nest_to_trans_cosim_updates = nest_app.run_for_synchronization_time(trans_to_nest_cosim_updates)
    trans_to_tvb_cosim_updates = nest_to_tvb_app.run_for_synchronization_time(nest_to_trans_cosim_updates)
    return tvb_app.cosimulator.n_tvb_steps_ran_since_last_synch, trans_to_tvb_cosim_updates, trans_to_nest_cosim_updates


def run_cosimulation(tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app,
                     advance_simulation_for_delayed_monitors_output=True):
    import time
    import numpy as np

    simulation_length = tvb_app.cosimulator.simulation_length
    synchronization_time = tvb_app.cosimulator.synchronization_time
    synchronization_n_step = tvb_app.cosimulator.synchronization_n_step  # store the configured value
    dt = tvb_app.cosimulator.integrator.dt

    steps_to_simulate = int(np.round(simulation_length / dt))
    simulated_steps = 0
    trans_to_tvb_cosim_updates = None

    tvb_app.cosimulator._tic = time.time()
    while steps_to_simulate - simulated_steps > 0:
        current_simulated_steps, trans_to_tvb_cosim_updates = \
            run_for_synchronization_time(tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app,
                                         trans_to_tvb_cosim_updates)
        simulated_steps += current_simulated_steps
        tvb_app.cosimulator._log_print_progress_message(simulated_steps, simulation_length)

    if advance_simulation_for_delayed_monitors_output:
        # Run once more for synchronization steps in order to get the full delayed monitors' outputs:
        remaining_steps = \
            int(np.round((simulation_length + synchronization_time - simulated_steps * dt) / dt))
        if remaining_steps:
            remaining_time = remaining_steps * dt
            if tvb_app.verbosity:
                tvb_app._logprint("Simulating for excess time %0.3f..." % remaining_time)
            tvb_app.cosimulator.synchronization_time = remaining_time
            tvb_app.cosimulator.synchronization_n_step = remaining_steps
            nest_app.synchronization_time = remaining_time
            current_simulated_steps, trans_to_tvb_cosim_updates = \
                run_for_synchronization_time(tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app,
                                             trans_to_tvb_cosim_updates)
            simulated_steps += current_simulated_steps
            tvb_app.cosimulator.synchronization_time = synchronization_time
            tvb_app.cosimulator.synchronization_n_step = synchronization_n_step
            nest_app.synchronization_time = synchronization_time

    tvb_app.cosimulator.simulation_length = simulated_steps * dt  # update the configured value

    return tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app, trans_to_tvb_cosim_updates


def run_example(plot=True):

    config = configure(config_class=Config)

    # ## BACKEND: 1. Load structural data <br> (minimally a TVB connectivity)  <br> & prepare TVB simulator  <br> (region mean field model, integrator, monitors etc)

    simulator = build_tvb_simulator(config=config, config_class=Config)


    # ## BACKEND: 2. Build and connect the NEST network model <br> (networks of spiking neural populations for fine-scale <br>regions, stimulation devices, spike detectors etc)

    # This would run on NEST only before creating any multiscale cosimulation interface connections.
    # Here it is assumed that the TVB simulator is already created and we can get some of its attributes,
    # either by directly accessing it, or via serialization.

    # nest_network = build_nest_network(config=config, config_class=Config)


    # ## FRONTEND: 3. Build the TVB-NEST interface


    tvb_interface_builder = configure_TVB_interfaces(simulator=simulator, config=config, config_class=Config)

    nest_interface_builder = configure_NEST_interfaces(config=config, config_class=Config)

    tvb_to_nest_interface_builder = configure_TVBtoNEST_transformer_interfaces(config=config, config_class=Config)

    nest_to_tvb_interface_builder = configure_NESTtoTVB_transformer_interfaces(config=config, config_class=Config)


    # ## BACKEND:
    # ### - Build TVB and Spiking Network models and simulators
    # ### - Build interfaces
    # ### - Configure co-simulation

    tvb_app = tvb_init(config, tvb_cosimulator_builder=build_tvb_simulator)

    nest_app = nest_init(config, build_nest_network)

    tvb_to_nest_app = tvb_to_nest_init(config)

    nest_to_tvb_app = nest_to_tvb_init(config)

    # Run serially for this test:
    tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app, trans_to_tvb_cosim_updates = \
        run_cosimulation(tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app,
                         advance_simulation_for_delayed_monitors_output=True)

    results = list(tvb_app.return_tvb_results())

    tvb_plot_kwargs = {}
    nest_plot_kwargs = {}
    if plot:
        from tvb_multiscale.core.plot.plotter import Plotter
        # set to False for faster plotting of only mean field variables and dates, apart from spikes" rasters:
        plot_per_neuron = False
        MAX_VARS_IN_COLS = 3
        MAX_REGIONS_IN_ROWS = 10
        MIN_REGIONS_FOR_RASTER_PLOT = 9
        # Set the transient time to be optionally removed from results:
        simulation_length = tvb_app.cosimulator.simulation_length
        transient = 0.1 * simulation_length
        config.figures.SHOW_FLAG = True
        config.figures.SAVE_FLAG = True
        config.figures.FIG_FORMAT = 'png'
        plotter = Plotter(config.figures)

        tvb_plot_kwargs = {"transient": transient, "plotter": plotter}
        nest_plot_kwargs = dict(tvb_plot_kwargs)
        nest_plot_kwargs.update({"time": results[0][0], "plot_per_neuron": plot_per_neuron})

        # ### TVB plots and Spiking Network plots upon finalizing

    nest_to_tvb_app = trans_final(nest_to_tvb_app)
    tvb_to_nest_app = trans_final(tvb_to_nest_app)
    nest_app = nest_final(nest_app, plot=plot, **nest_plot_kwargs)
    tvb_app = tvb_final(tvb_app, plot=plot, **tvb_plot_kwargs)

    del tvb_app, nest_app, tvb_to_nest_app, nest_to_tvb_app

    return results, config


def test():
    import os
    from xarray import DataArray
    from tvb_multiscale.core.utils.file_utils import load_pickled_dict

    SPIKES_NUMBERS_PER_REG = [2900, 2800]

    run_example()

    config = configure(config_class=Config)

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
