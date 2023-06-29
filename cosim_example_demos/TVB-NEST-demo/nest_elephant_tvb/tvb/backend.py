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


def final(tvb_app, plot=True, **kwargs):
    if plot:
        tvb_app.plot(**kwargs)
    tvb_app.clean_up()
    tvb_app.stop()
    return tvb_app

