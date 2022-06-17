#  Copyright 2020 Forschungszentrum Jülich GmbH and Aix-Marseille Université
# "Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements;
# and to You under the Apache License, Version 2.0. "
import numpy
import sys
import os
import pickle
import base64

from common.utils.security_utils import check_integrity
from actions_adapters.parameters import Parameters
import cosim_example_demos.TVB_NEST_demo.tvb_sim.wrapper_TVB_mpi as Wrapper
from EBRAINS_RichEndpoint.Application_Companion.common_enums import SteeringCommands
from EBRAINS_RichEndpoint.Application_Companion.common_enums import INTEGRATED_SIMULATOR_APPLICATION as SIMULATOR
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.default_directories_enum import DefaultDirectories
from EBRAINS_ConfigManager.global_configurations_manager.xml_parsers.configurations_manager import ConfigurationsManager
from EBRAINS_ConfigManager.workflow_configuraitons_manager.xml_parsers.xml2class_parser import Xml2ClassParser


import matplotlib.pyplot as plt

numpy.random.seed(125)


# BACKEND:
# Backend functions will be run upon runtime to build:
# - the TVB cosimulator, based on a user provided script (build_tvb_simulator.py),
# - and the TVB interfaces based on the configurations written to files by the FRONTEND.

def build_TVB_interfaces(simulator, parameters):
    tvb_interface_builder = parameters.prepare_TVB_interface(simulator=simulator)[0]

    # Load TVB interfaces configurations
    tvb_interface_builder.load_all_interfaces()

    # Configure TVB interfaces' builder:
    tvb_interface_builder.configure()
    # tvb_interface_builder.print_summary_info_details(recursive=1)

    # Build interfaces and attach them to TVB simulator
    simulator = tvb_interface_builder.build()

    # simulator.print_summary_info(recursive=3)
    # simulator.print_summary_info_details(recursive=3)

    print("\n\noutput (TVB-> coupling) interfaces:\n")
    simulator.output_interfaces.print_summary_info_details(recursive=2)

    print("\n\ninput (TVB<- update) interfaces:\n")
    simulator.input_interfaces.print_summary_info_details(recursive=2)

    return simulator


# This is the entrypoint for the EBRAINS Cosimulation platform:
def backEnd_TVB(parameters, simulator=None):
    if simulator is None:
        # Build TVB simulator
        simulator = parameters.build_tvb_simulator(parameters)

    simulator.simulation_length = parameters.simulation_time

    # Build TVB interfaces and attach them to TVB simulator
    simulator = build_TVB_interfaces(simulator)

    # Interfaces can be accessed with integer indicies i_interface by:
    # simulator.output_interfaces.interfaces[i_interface]
    # simulator.input_interfaces.interfaces[i_interface]

    # results = simulate_TVB(simulator, simulation_length)

    return simulator  # results


class TVBAdapter:

    def __init__(self, p_configurations_manager=None, p_log_settings=None, p_sci_params_xml_path_filename=None):
        self.__simulator = None
        self._log_settings = p_log_settings
        self._configurations_manager = p_configurations_manager
        self.__logger = self._configurations_manager.load_log_configurations(
            name="TVB_Adapter",
            log_configurations=self._log_settings,
            target_directory=DefaultDirectories.SIMULATION_RESULTS)
        self.__path_to_parameters_file = self._configurations_manager.get_directory(
            directory=DefaultDirectories.SIMULATION_RESULTS)

        # Loading scientific parameters into an object
        self.__sci_params = Xml2ClassParser(p_sci_params_xml_path_filename, self.__logger)

        self.__parameters = Parameters(self.__path_to_parameters_file)
        print(f'__ric__,self.__path_to_parameters_file={self.__path_to_parameters_file}')
        self.__logger.info("initialized")

    #
    # def __configure(self, time_synch=0.1, id_nest_region=None, dt=0.1):
    def __configure(self):
        """
        configure TVB before the simulation
        modify example of https://github.com/the-virtual-brain/tvb-root/blob/master/tvb_documentation/tutorials/tutorial_s1_region_simulation.ipynb
        based on: https://github.com/multiscale-cosim/TVB-NEST-demo/blob/main/nest_elephant_tvb/launcher/run.py
        :param: no parameters required
        :return: simulator
        """

        # This is the function that plays the role of the Entrypoint:
        simulator = backEnd_TVB(self.__parameters)
        simulator.configure()
        self.__logger.info(f'TVB simulator has been configured...')
        return simulator

    def execute_init_command(self):
        self.__logger.debug("executing INIT command")
        # self.__simulator = self.__configure(
        #     self.__parameters.time_synch,
        #     self.__parameters.id_nest_region,
        #     self.__parameters.resolution)
        self.__simulator = self.__configure()
        self.__logger.debug("INIT command is executed")
        return self.__parameters.time_synch  # minimum step size for simulation

    def execute_start_command(self):
        self.__logger.debug("executing START command")
        (r_raw_results,) = Wrapper.run_mpi(
            self.__simulator,
            self.__parameters.path,
            self.__logger)
        self.__logger.debug('TVB simulation is finished')
        return r_raw_results

    def execute_end_command(self, p_raw_results=None):
        self.__logger.info("plotting the result")
        plt.figure(1)
        plt.plot(p_raw_results[0], raw_results[1][:, 0, :, 0] + 3.0)
        plt.title("Raw -- State variable 0")
        plt.savefig(self.__parameters.path + "/figures/plot_tvb.png")

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
    tvb_adapter = TVBAdapter(p_configurations_manager=configurations_manager,
                             p_log_settings=log_settings,
                             p_sci_params_xml_path_filename=sys.argv[4])
    local_minimum_step_size = tvb_adapter.execute_init_command()
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
        raw_results = tvb_adapter.execute_start_command()
        tvb_adapter.execute_end_command(raw_results)
        sys.exit(0)
    else:
        # TODO raise and log the exception with traceback and terminate with
        # error if received an unknown steering command
        print(f'unknown steering command: '
              f'{SteeringCommands[user_action_command]}',
              file=sys.stderr)
        sys.exit(1)