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


def final(nest_app, plot=True, **kwargs):
    if plot:
        nest_app.plot(**kwargs)
    nest_app.clean_up()
    nest_app.stop()
    return nest_app
