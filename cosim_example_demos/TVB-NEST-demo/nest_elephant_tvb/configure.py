import os
import numpy as np


def configure():
    # Set up the environment
    from tvb.basic.profile import TvbProfile
    TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE)

    from tvb_multiscale.tvb_nest.config import Config

    # ----------- Simulation options ----------------
    SIM_MODE = "tvb-nest"  # "tvb-nest"  for multiscale cosimulation, "tvb" ("nest") for only TVB (NEST) simulation, respectively
    NEST_MODEL_BUILDERS = None  # only None will work!, "opinionated", "nonopinionated", None

    # For a minimal example, select:
    n_regions = 4  # total TVB brain regions
    nest_nodes_inds = np.array(
        [0, 1])  # the brain region nodes to place spiking networks from [0, n_regions-1] interval
    n_neurons = 10  # number of neurons per spiking population
    # -----------------------------------------------

    # Base paths
    work_path = os.getcwd()
    outputs_path = os.path.join(work_path, "outputs/WilsonCowanMin/Front_Back_End_Separated_Trans")
    if NEST_MODEL_BUILDERS is None:
        outputs_path += "NoNestBuilders"
    elif NEST_MODEL_BUILDERS == "opinionated":
        outputs_path += "OpinionBuilders"
    elif NEST_MODEL_BUILDERS == "nonopinionated":
        outputs_path += "NonOpinionBuilders"

    # Generate a configuration class instance
    config = Config(output_base=outputs_path)
    config.figures.SHOW_FLAG = True
    config.figures.SAVE_FLAG = True
    config.figures.FIG_FORMAT = 'png'
    config.figures.DEFAULT_SIZE = config.figures.NOTEBOOK_SIZE

    return config, n_regions, NEST_MODEL_BUILDERS, nest_nodes_inds, n_neurons
