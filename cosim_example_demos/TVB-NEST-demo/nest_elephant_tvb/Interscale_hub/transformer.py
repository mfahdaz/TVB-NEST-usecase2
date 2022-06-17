# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor
#  license agreements; and to You under the Apache License, Version 2.0. "
#
# Forschungszentrum Jülich
#  Institute: Institute for Advanced Simulation (IAS)
#    Section: Jülich Supercomputing Centre (JSC)
#   Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
#       Team: Multi-scale Simulation and Design
#
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements; and to You under the Apache License,
# Version 2.0. "
#
# Forschungszentrum Jülich
# Institute: Institute for Advanced Simulation (IAS)
# Section: Jülich Supercomputing Centre (JSC)
# Division: High Performance Computing in Neuroscience
# Laboratory: Simulation Laboratory Neuroscience
# Team: Multi-scale Simulation and Design
# ------------------------------------------------------------------------------


from actions_adapters.parameters import Parameters
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories


# BACKEND:
# Backend functions will be run upon runtime to build the TVB --> NEST interfaces
# This is the entrypoint for the EBRAINS Cosimulation platform
def build_TVBtoSpikeNet_transformer_interfaces(prepare_TVBtoSpikeNet_transformer_interface):
    tvb_to_spikeNet_trans_interface_builder = prepare_TVBtoSpikeNet_transformer_interface()

    # Load TVB to spikeNet interfaces configurations
    tvb_to_spikeNet_trans_interface_builder.load_all_interfaces()

    # Configure TVB to spikeNet interfaces' builder:
    tvb_to_spikeNet_trans_interface_builder.configure()
    # tvb_to_spikeNet_trans_interface_builder.print_summary_info_details(recursive=1)

    # Build TVB to spikeNet interfaces
    tvb_to_spikeNet_trans_interfaces = tvb_to_spikeNet_trans_interface_builder.build()

    print("\n\noutput (TVB -> ... -> Transformer -> ... -> spikeNet coupling) interfaces:\n")
    tvb_to_spikeNet_trans_interfaces.print_summary_info_details(recursive=2)

    # All that EBRAINS wants is the transformers:
    transformers = []
    for interface in tvb_to_spikeNet_trans_interfaces.interfaces:
        transformers.append(interface.transformer)

    return transformers


# BACKEND:
# Backend functions will be run upon runtime to build the TVB <-- NEST interfaces
# This is the entrypoint for the EBRAINS Cosimulation platform
def build_spikeNetToTVB_transformer_interfaces(prepare_spikeNetToTVB_transformer_interface):
    spikeNet_to_tvb_trans_interface_builder = prepare_spikeNetToTVB_transformer_interface()

    # Load spikeNet to TVB interfaces configurations
    spikeNet_to_tvb_trans_interface_builder.load_all_interfaces()

    # Configure spikeNet to TVB interfaces' builder:
    spikeNet_to_tvb_trans_interface_builder.configure()
    # spikeNet_to_tvb_trans_interface_builder.print_summary_info_details(recursive=1)

    # Build spikeNet to TVB interfaces
    spikeNet_to_tvb_trans_interfaces = spikeNet_to_tvb_trans_interface_builder.build()

    print("\n\ninput (TVB<- ... <- Transformer <- ... <- spikeNet update) interfaces:\n")
    spikeNet_to_tvb_trans_interfaces.print_summary_info_details(recursive=2)

    # All that EBRAINS wants is the transformers:
    transformers = []
    for interface in spikeNet_to_tvb_trans_interfaces.interfaces:
        transformers.append(interface.transformer)

    return transformers


class Transformer:
    """
    Transforms the data to change the scales. It wraps the functionality of the libraries
    such as ELEPHANT for transformation.
    NOTE this wrapper class exposes only the functionality that is supported by
    InterscaleHub.
    """

    def __init__(self, param, configurations_manager, log_settings, sci_params=None):
        """
        """
        self._log_settings = log_settings
        self._configurations_manager = configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
            name="InterscaleTransformer",
            log_configurations=self._log_settings,
            target_directory=DefaultDirectories.SIMULATION_RESULTS)

        self.__parameters = Parameters(self.__path_to_parameters_file)

        # This is the function that plays the role of the Entrypoint:
        self._tvb_to_nest_transformers = \
            self.__parameters.build_TVBtoSpikeNet_transformer_interfaces(
                self.__parameters.prepare_TVBtoSpikeNet_transformer_interface)
        # TODO: I have assumed that the two directions are separated!!!
        # self._nest_to_tvb_transformers =
        #   self.__parameters.build_spikeNetToTVB_transformer_interfaces(
        #       self.__parameters.prepare_spikeNetToTVB_transformer_interface)

        self.__logger.info("Initialised")

    def spike_to_spiketrains(self, count, data_size, data_buffer):
        """Transforms the data from one format to another .
        Parameters
        ----------
        count: int
            counter of the number of time of the transformation
            (identify the timing of the simulation)
        data_size : int
            size of the data to be read from the buffer for transformation
        data_buffer: MPI shared memory window
            buffer contains id of devices, id of neurons and spike times
        Returns
        ------
            returns the spike trains from spikes
        """
        # return self.__elephant_delegator.spike_to_spiketrains(count, data_size, data_buffer)
        pass

    def rate_to_spikes(self, time_step, data_buffer):
        """Transforms the data from one format to another .
        Parameters
        ----------
        time_step: [int, int]
            time start and time stops of the current simulation step

        data_buffer: MPI shared memory window
            buffer contains the rate to be converted into spikes
        Returns
        ------
            returns the spike trains from rate
        """
        # return self.__elephant_delegator.rate_to_spikes(time_step, data_buffer)
        pass