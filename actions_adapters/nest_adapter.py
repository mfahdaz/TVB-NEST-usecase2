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
import os
import sys
import pickle
import base64

from actions_adapters.nest_simulator.utils_function import wait_transformation_modules
from actions_adapters.nest_simulator.utils_function import get_data
from common.utils.security_utils import check_integrity
from actions_adapters.parameters import Parameters
from EBRAINS_RichEndpoint.Application_Companion.common_enums import SteeringCommands
from EBRAINS_RichEndpoint.Application_Companion.common_enums import INTEGRATED_SIMULATOR_APPLICATION as SIMULATOR
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.configurations_manager import ConfigurationsManager
from EBRAINS_ConfigManager.workflow_configuraitons_manager.xml_parsers.xml2class_parser import Xml2ClassParser

import nest
import nest.raster_plot
import matplotlib.pyplot as plt


# BACKEND:
# Backend functions will be run upon runtime to build:
# - the NEST network model and simulator, based on a user provided script (build_nest_network.py),
# - and the NEST interfaces based on the configurations written to files by the FRONTEND.

def build_spikeNet_interfaces(nest_network, parameters):
    spikeNet_interface_builder = parameters.prepare_spikeNet_interface_builder(nest_network)[0]

    # Load spikeNet interfaces configurations
    spikeNet_interface_builder.load_all_interfaces()

    # Configure spikeNet interfaces' builder:
    spikeNet_interface_builder.configure()
    # spikeNet_interface_builder.print_summary_info_details(recursive=1)

    # Build spikeNet interfaces and attach them to spikeNet simulator
    nest_network = spikeNet_interface_builder.build()

    print("\n\noutput (TVB->NEST coupling) interfaces:\n")
    nest_network.output_interfaces.print_summary_info_details(recursive=2)

    print("\n\ninput (NEST->TVB update) interfaces:\n")
    nest_network.input_interfaces.print_summary_info_details(recursive=2)

    return nest_network


# This is the entrypoint for the EBRAINS Cosimulation platform:

def backEnd_spikeNet(parameters, nest=None):
    # Build the spikeNet simulator
    nest_network, nest_nodes_inds = parameters.build_nest_network(nest=nest)

    # Build the spikeNet interfaces and attach them to the spikeNet network
    nest_network = build_spikeNet_interfaces(nest_network, parameters)

    # Configure the interfaces
    if nest_network.input_interfaces:
        nest_network.input_interfaces.configure()
    if nest_network.output_interfaces:
        nest_network.output_interfaces.configure()

    # simulate_spikeNet(nest_network, simulation_length)

    return nest_network


class NESTAdapter:
    def __init__(self, p_configurations_manager=None, p_log_settings=None, sci_params_xml_path_filename=None):
        self._log_settings = p_log_settings
        self._configurations_manager = p_configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
            name="NEST_Adapter",
            log_configurations=self._log_settings,
            target_directory=DefaultDirectories.SIMULATION_RESULTS)
        self.__path_to_parameters_file = self._configurations_manager.get_directory(
            directory=DefaultDirectories.SIMULATION_RESULTS)

        # Loading scientific parameters into an object
        self.__sci_params = Xml2ClassParser(sci_params_xml_path_filename, self.__logger)

        self.__parameters = Parameters(self.__path_to_parameters_file)

        self.__logger.info("initialized")

    def __configure_nest(self, simulator):
        """
        configure NEST before the simulation
        modify example of https://simulator.simulator.readthedocs.io/en/stable/_downloads/482ad6e1da8dc084323e0a9fe6b2c7d1/brunel_alpha_simulator.py
        :param simulator: nest simulator
        :return:
        """

        # This is the function that plays the role of the Entrypoint:
        self._nest_network = backEnd_spikeNet(self.__parameters, nest=simulator)
        self._nest = self._nest_network.nest_instance

        # # The NEST input and output proxy devices can be accessed with integer indices i_interface and i_proxy_node as:
        # self._nest_network.input_interfaces.interfaces[i_interface].proxy.target[i_proxy_node]
        # self._nest_network.output_interfaces.interfaces[i_interface].proxy.source[i_proxy_node]
        #
        # # and their gids like for each proxy node:
        # self._nest_network.input_interfaces.interfaces[i_interface].proxy.target[i_proxy_node].global_id
        # self._nest_network.output_interfaces.interfaces[i_interface].proxy.source[i_proxy_node].global_id
        # # which return the gid as a tuple of integers

        # # or for all proxy nodes of an interface together:
        # self._nest_network.output_interfaces.interfaces[i_interface].proxy_gids
        # self._nest_network.input_interfaces.interfaces[i_interface].proxy_gids
        # # which return the gids as a numpy.ndarray of integers

        return self._nest_network.input_interfaces, self._nest_network.output_interfaces

    def execute_init_command(self):
        self.__logger.debug("executing INIT command")
        self._nest = nest
        self._nest.ResetKernel()
        self._nest.SetKernelStatus(
            {"data_path": self.__parameters.path + '/nest/', "overwrite_files": True, "print_time": True,
             "resolution": self.__parameters.resolution})

        self.__logger.info("configure the network")
        espikes, input_to_simulator, output_from_simulator = \
            self.__configure_nest(self._nest)

        self.__logger.info("establishing the connections")
        wait_transformation_modules(
            self._nest,
            self.__parameters.path,
            input_to_simulator,
            output_from_simulator,
            self.__logger)
        self.__logger.info("preparing the simulator")
        self._nest.Prepare()
        self.__logger.info("connections are made")
        self.__logger.debug("INIT command is executed")
        return self.__parameters.time_synch  # minimum step size for simulation

    def execute_start_command(self):
        self.__logger.debug("executing START command")
        count = 0.0
        self.__logger.debug('starting simulation')
        while count * self.__parameters.time_synch < self.__parameters.simulation_time:
            self._nest.Run(self.__parameters.time_synch)
            count += 1

        self.__logger.debug('nest simulation is finished')
        self.__logger.info("cleaning up NEST")
        self._nest.Cleanup()
        # self.execute_end_command()

    def execute_end_command(self):
        self.__logger.info("plotting the result")
        if self._nest.Rank() == 0:
            self._nest.raster_plot.from_data(get_data(self.__logger, self.__parameters.path + '/nest/'))
            plt.savefig(self.__parameters.path + "/figures/plot_nest.png")

        self.__logger.debug("post processing is done")


if __name__ == "__main__":

    # unpickle configurations_manager object
    configurations_manager = pickle.loads(base64.b64decode(sys.argv[2]))

    # unpickle log_settings
    log_settings = pickle.loads(base64.b64decode(sys.argv[3]))

    # security check of pickled objects
    # it raises an exception, if the integrity is compromised
    check_integrity(configurations_manager, ConfigurationsManager)
    check_integrity(log_settings, dict)

    # everything is fine, run simulation
    nest_adapter = NESTAdapter(configurations_manager, log_settings,
                               sci_params_xml_path_filename=sys.argv[4])

    local_minimum_step_size = nest_adapter.execute_init_command()

    # send local minimum step size to Application Manager as a response to INIT
    # NOTE Application Manager expects a string in the following format:
    # {'PID': <int>, 'LOCAL_MINIMUM_STEP_SIZE': <float>}
    pid_and_local_minimum_step_size = \
        {SIMULATOR.PID.name: os.getpid(),
         SIMULATOR.LOCAL_MINIMUM_STEP_SIZE.name: local_minimum_step_size}

    # Application Manager will read the stdout stream via PIPE
    # NOTE the communication with Application Manager via PIPES will be
    # changed to some other mechanism
    print(f'{pid_and_local_minimum_step_size}')

    user_action_command = input()
    # execute if steering command is START
    if SteeringCommands[user_action_command] == SteeringCommands.START:
        nest_adapter.execute_start_command()
        nest_adapter.execute_end_command()
        sys.exit(0)
    else:
        # TODO raise and log the exception with traceback and terminate with
        # error if received an unknown steering command
        print(f'unknown steering command: '
              f'{SteeringCommands[user_action_command]}',
              file=sys.stderr)
        sys.exit(1)
