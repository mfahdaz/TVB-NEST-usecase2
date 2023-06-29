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
