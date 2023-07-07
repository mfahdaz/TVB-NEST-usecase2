from tvb_multiscale.core.orchestrators.transformer_app import TVBtoSpikeNetTransformerApp
from tvb_multiscale.core.orchestrators.transformer_app import SpikeNetToTVBTransformerApp


def init(config, transformer_app_class, **kwargs):

    # Create a TransformerApp
    transformer_app = \
        transformer_app_class(config=config,
                              **kwargs)

    transformer_app.start()
    # Configure App (and Transformer interface builders)
    transformer_app.configure()

    # Build (Transformer interfaces)
    transformer_app.build()

    # Configure App for CoSimulation
    transformer_app.configure_simulation()

    return transformer_app


def tvb_to_nest_init(config, **kwargs):
    return init(config, TVBtoSpikeNetTransformerApp, **kwargs)


def nest_to_tvb_init(config, **kwargs):
    return init(config, SpikeNetToTVBTransformerApp, **kwargs)


def final(trans_app):
    trans_app.clean_up()
    trans_app.stop()
    return trans_app


def run_for_synchronization_time(trans_app, cosim_to_trans_cosim_updates):
    """Function for cosimulating for one loop of synchronization time.
       It could be the highest level possible ENTRYPOINT for a parallel cosimulation.
       In that case, the cosimulation manager would be completely agnostic
       - of what the Apps of the different processed do,
       - including the transformation function they employ.
       The ENTRYPOINT here is just the cosimulation updates' data,
       which are "thrown over the wall" for the necessary data exchanges.
    """
    # Loop using this ENTRYPOINT:
    if cosim_to_trans_cosim_updates is not None:
        # ...if any:
        trans_to_cosim_updates = trans_app.run_for_synchronization_time(cosim_to_trans_cosim_updates)
    else:
        trans_to_cosim_updates = None
    return trans_app, trans_to_cosim_updates


def run_cosimulation(trans_app):
    """Function for running the whole cosimulation, assuming all Apps are built and configured.
       This function shows the necessary initialization of the cosimulation.
    """

    # Initial conditions of co-simulation:
    cosim_to_trans_cosim_updates = None

    # TODO: Some get the time_to_simulate from the Simulation Manager!
    # Loop for steps_to_simulate in steps of synchronization_time:
    while time_to_simulate > 0:
        trans_to_cosim_updates = run_for_synchronization_time(trans_app, cosim_to_trans_cosim_updates)

    return trans_app, trans_to_cosim_updates


def backend_tvb_to_nest(config):
    """Function that
       - builds all components based on user provided configurations,
       - configures them for cosimulation,
       - performs cosimulation,
       - and finalizes (plotting, cleaning up).
       This function shows how the backend should be split for the interscale hub process
       """

    # TVB to NEST app, including TVB to NEST interfaces and their transformers:
    tvb_to_nest_app = tvb_to_nest_init(config)

    # Integration:
    tvb_to_nest_app = run_cosimulation(tvb_to_nest_app)[0]

    # Finalize (including optional plotting), cleaning up, etc...
    tvb_to_nest_app = final(tvb_to_nest_app)

    # Delete apps, optionally:
    del tvb_to_nest_app


def backend_nest_to_tvb(config):
    """Function that
       - builds all components based on user provided configurations,
       - configures them for cosimulation,
       - performs cosimulation,
       - and finalizes (plotting, cleaning up).
       This function shows how the backend should be split for the interscale hub process
       """

    # NEST to TVB app, including NEST to TVB interfaces and their transformers:
    nest_to_tvb_app = nest_to_tvb_init(config)

    # Integration:
    # nest_to_tvb_app = run_cosimulation(nest_to_tvb_app)[0]

    # Finalize (including optional plotting), cleaning up, etc...
    nest_to_tvb_app = final(nest_to_tvb_app)

    # Delete apps, optionally:
    del nest_to_tvb_app
