#!/usr/bin/env python

# # WORKFLOW:

# Imports that would be identical for any TVB<->NEST cosimulation script:
from tvb.basic.profile import TvbProfile
TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)


def frontend(config):
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

    # This is a BACKEND script for creating the TVB Simulator instance, which we run here, at the FRONTEND,
    # in order to produce the serialized TVB Simulator dictionary file,
    # to be used for parametrizing the interface configurations
    simulator = config.TVB_CONFIG(config=config)

    # FRONTEND: Configure the TVB<->NEST interfaces
    tvb_interface_builder = config.TVB_INTERFACE_CONFIG(simulator=simulator, config=config)
    nest_interface_builder = config.NEST_INTERFACE_CONFIG(config=config)
    tvb_to_nest_interface_builder = config.TVB_to_NEST_INTERFACE_CONFIG(config=config)
    nest_to_tvb_interface_builder = config.NEST_to_TVB_INTERFACE_CONFIG(config=config)

    # Outputs only for debugging:
    return tvb_interface_builder, nest_interface_builder, \
           tvb_to_nest_interface_builder, nest_to_tvb_interface_builder, simulator
