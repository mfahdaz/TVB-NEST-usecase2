# ModularScience-Cosim TVB-NEST usecase
## TVB Multiscale
WIP
---

### INSTALL AND TEST RUN:
0. cd to a folder with bootstrap.sh, vagrantfile and a folder "shared"
1. vagrant up # to build the VM
2. vagrant ssh # to ssh into it
3. cd multiscale_cosim;
   source TVB-NEST-usecase2.source  # to setup the environment
4. cd cosim_example_demos/TVB-NEST-demo
   python3 WilsonCowanMinimal_EBRAINS.py  # to run the example
   python3 WilsonCowanMinimal_EBRAINS.py  test # to run the test

!!! Currently the test requires the flag plot=True into the run_example() function!!!


### TVB-Multiscale Introduction
[TVB-Multiscale](https://github.com/the-virtual-brain/tvb-multiscale) 
is a toolbox, part of the ecosystem of 
[TheVirtualBrain](https://github.com/the-virtual-brain) for:
- user-friendly configuration of a variety of multiscale brain models, 
employing TVB for the whole brain model, based on neural mass population models,
and a spiking network, implemented in one of 
NEST, ANNarchy or NetPyNE (NEURON) spiking simulators, 
for some selected brain regions and neuronal populations,
- for implementing and co-simulating such models, 
including all necessary transformations and data exchanges between the two simulators.

Main components (and classes thereof) of a TVB-Multiscale model:
- TVB CoSimulator,
- Spiking (NEST, ANNarchy or NetPyNE) network
- A set of Interfaces for each direction of communication (TVB <-> Spiking Network)

An Interface combines some or all of the following functions:
- some logic for inputing/outputing data (e.g., from a simulator or transformer)
and preparing them (reshaping etc),
- optionally, a Communicator class for sending/receiving the data to/from another process,
- optionally, a Transformer for transforming data based on some computation 
(e.g., from/to firing rates to/from spikes' trains).

In the case where all the main components reside to remote processes,
a Communicator class is necessary to send/receive the data. 
The default Communicator classes, meant only for testing, currently, 
are NPZWriter and NPZReader ones.

In the scenario implemented here, where TVB, SpikingNetwork (e.g., NESTNetwork)
and the Transformers of each direction run all on separate processes, 
and without specifying a TVB-multiscale Communicator,
we need the following kinds of Interfaces:
- TVBInterfaces (Output/Input) output/input data from/to TVB to/from TVBtoSpikeNetTransformerInterfaces.
- NESTInterfaces (Output/Input) output/input data from/to NEST to/from NESTtoTVBTransformerInterfaces.
- TVBtoSpikeNetTransformerInterfaces transforming mean field activity (e.g., rates) data
received from TVB, to spiking neuronal activity (e.g., spikes' trains) to be sent to NEST.
- SpikeNetToTVBTransformerInterfaces transforming spiking neuronal activity (e.g., spikes' trains)
received from NEST, to mean field activity (e.g., rates) data to be sent to TVB.

Several independent "pipelines" of such interfaces can be configured and then run at runtime, 
currently serially, but in the future also in parallel (for TransformerInterfaces),
as long as their relative order is preserved for each direction, 
e.g., TVBInterface_i will output some data towards TVBtoSpikeNetTransformerInterfaces_i,
which will transform them and output them to NESTInterface_i to get them into NEST,
where i = 0, 1, 2, ..., Ntvb->nest interfaces.
It goes similarlty for the other direction.

Currently, data from all interfaces are gathered at each component 
and "sent over the wall" to their destination,
following the structure:
- A list containing data of each (ith) interface: [data_0, data_1, ...., data_n]
- Each data_i is also a list with three elements:
  - an array of starting and ending time step (integers): numpy.array([T_t, T_t+t_sync]),
    where t_sync is the synchronization_time, 
    in integer multiples of the TVB integration time step (CoSimulator.integrator.dt),
  - a list of lists (in case of spikes' data), 
    or a multidimensional numpy.array, in case of mean field data),
    i.e, the actual data being communicated in each direction,
  - the integer i denoting the order of each interface, sent to for error checking.

Each of the main components can be built based on user provided scripts or related Builder classes
(e.g., TVB CoSimulatorBuilder, NESTNetworkBuilder, etc).
Interfaces can be configured by the user based on scripts or pickled configuration files,
which are then read by the TransformerInterfacesBuilders.

Finally, each component is managed by its correspondent App at a high level of abstraction
(i.e., TVBParallelApp, NESTParallelApp, TVBtoSpikeNetApp, SpikeNetToTVBApp).



### TVB-Multiscale in EBRAINS CoSimulation use case:

#### Requirements for a specific example:

In the present use case example:
- We assume that the user provides an example specific script
  [WilsonCowanMinimal_EBRAINS.py](cosim_example_demos/TVB-NEST-demo/WilsonCowanMinimal_EBRAINS.py),
which:
  - imports all necessary example specific functions:
    - [examples.parallel.wilson_cowan.config.configure](https://github.com/the-virtual-brain/tvb-multiscale/blob/parallel/examples/parallel/wilson_cowan/config.py): 
      it generates a Config class instance with all necessary configurations for this example,
    - [examples.parallel.tvb_nest.wilson_cowan.tvb_config.build_tvb_simulator](https://github.com/the-virtual-brain/tvb-multiscale/blob/parallel/examples/parallel/tvb_nest/wilson_cowan/tvb_config.py):
      it builds and configures the TVB CoSimulator for this example
    - [examples.parallel.tvb_nest.wilson_cowan.nest_config.build_nest_network](https://github.com/the-virtual-brain/tvb-multiscale/blob/parallel/examples/parallel/tvb_nest/wilson_cowan/nest_config.py)
      it builds and configures the NESTNetwork for this example
  - and calls the example unspecific backend functions for the example to run,
    by creating, and configuring all models' components with the corresponding Apps,
  - whereas the current example contains also a run_example() and a test() function,

- a set of configurations as pickled .pkl files in folder
[examples_data/WilsonCowanNoSpikeNetBuilders/config](cosim_example_demos/TVB-NEST-demo/examples_data/WilsonCowanNoSpikeNetBuilders/config):
  - TVB CoSimulator serialized: tvb_serial_cosimulator.pkl,
  - several .pkl files for configuring all necessary interfaces.

- or, alternatively, we can run the frontend() function, 
also included in [WilsonCowanMinimal_EBRAINS.py](cosim_example_demos/TVB-NEST-demo/WilsonCowanMinimal_EBRAINS.py), 
to generate them.
This function will run one user defined example specific script for each interface:
  - [examples.parallel.tvb_nest.wilson_cowan.tvb_interface_config.configure_TVB_interfaces](https://github.com/the-virtual-brain/tvb-multiscale/blob/parallel/examples/parallel/tvb_nest/wilson_cowan/tvb_interface_config.py)
  - [examples.parallel.tvb_nest.wilson_cowan.nest_interface_config.configure_NEST_interfaces](https://github.com/the-virtual-brain/tvb-multiscale/blob/parallel/examples/parallel/tvb_nest/wilson_cowan/nest_interface_config.py)
  - [examples.parallel.tvb_nest.wilson_cowan.transformers_config.configure_TVBtoNEST_transformer_interfaces](https://github.com/the-virtual-brain/tvb-multiscale/blob/parallel/examples/parallel/tvb_nest/wilson_cowan/transformers_config.py)
  - [examples.parallel.tvb_nest.wilson_cowan.transformers_config.configure_NESTtoTVB_transformer_interfaces](https://github.com/the-virtual-brain/tvb-multiscale/blob/parallel/examples/parallel/tvb_nest/wilson_cowan/transformers_config.py)
  which will generate the needed configurations files and will write them to the folder mentioned above.

In other words, the minimal requirement for a specific example is a script containing
imports of the basic configuration scripts, 
all necessary frontend configurations' functions,
(unless interfaces' configurations files are already generated and placed to the correct path),
as well as all backend model building functions (one for each of TVB CoSimulator and NESTNetwork).
The latter backend functions are attached to the generated Config class instance,
in order to be used by the example unspecific backend scripts.


#### Backend functions, unspecific to any example:

Those scripts are one for each component, and can be found in the respective folders:
- [nest_elephant_tvb.tvb.backend](cosim_example_demos/TVB-NEST-demo/nest_elephant_tvb/tvb/backend.py): 
  script to build, initialize, configure TVB CoSimulator, 
  as well as all TVBInterfaces,
  by consuming the user defined build_tvb_simulator script,
  by loading the TVBInterfaces configurations from the .pkl files,
  and by making use of the TVBParallelApp
- [nest_elephant_tvb.nest.backend](cosim_example_demos/TVB-NEST-demo/nest_elephant_tvb/nest/backend.py):
  script to build, initialize, configure NESTNetwork,
  as well as all NESTInterfaces,
  by consuming the user defined build_tvb_simulator script, 
  by loading the NESTInterfaces configurations from the .pkl files,
  and by making use of the NESTarallelApp
- [nest_elephant_tvb.Interscale_hub.backend](cosim_example_demos/TVB-NEST-demo/nest_elephant_tvb/Interscale_hub/backend.py):
  scripts to build, initialize, configure 
  TVBtoSpikeNetTransformerInterfaces and SpikeNetToTVBTransformerInterfaces,
  by loading the respective configurations from the .pkl files,
  and by making use of the TVBtoSpikeNetApp and  SpikeNetToTVBApp, respectively.

In each one of the above backend scripts you can also find functions that show how simulation
should run at runtime:
- initialization (including sending out the TVB initial condition, if any)
  and simulation loops are in run_cosimulation() functions,
- simulating for a single synchronization_time period is in run_for_synchronization_time() functions.

Finally, in each backend script, one can also find functions final() 
to be optionally used for plotting, and cleaning up, at the end of the co-simulation,
before deleting the Apps completely.

#### ENTRYPOINT for EBRAINS CoSimulation MPI communication and co-simulation management:

- The backend functions init(), and final() can be used as ENTRYPOINTS, for each component's process,
for initializing the co-simulation, and optionally cleaning up and plotting the results.

- The run_for_synchronization_time() functions 
  can be used as ENTRYPOINTS for runtime of co-simulation since they consist - for each component - of the same steps:
  1. get data "over the wall",
  2. process (simulate or transform),
  3. send data "over the wall",
always using the method run_for_synchronization_time() of the respective App.

In general, using the respective Apps, provides ENTRYPOINTS at the highest and most abstract possible level.

#### Important general co-simulation parameters:
The main parameter that determines the co-simulation is the synchronization_time. It follows two constraints:
- It has to be an integer multiple of the TVB's CoSimulator.integrator.dt time step of integration.
- It has to be smaller or equal to the minimum TVB time delay as determined by the TVB connectome's delays.
This is how it is set in the current example by default (i.e., 3.6 ms)

Other important parameters are:
- the CoSimulator.integrator.dt 
  (see above, 0.1 ms in this example leading to synchronization time being equal to 36 TVB time steps), 
  also needed for converting times into integer multiples of time steps at all components' processes,
- the NEST resolution time, in the current example chosen as half the integration time step of TVB,
- the weights and delays of the TVB connectome, 
  especially in case the NESTNetwork needs to implement one TVB "proxy" node 
  (usually somem spike generating device) for each TVB node that sends coupling to NEST.

TVB has priority in the current example for determining most of the parameters of the cosimulation.
This is why we run the build_tvb_simulator script before generating the frontend configurations,
so that we get the pickled serialized TVB CoSimulator file and make it available to all other components
(i.e., each App loads this dictionary of TVB CoSimulator properties).

For the same reason, it would be a safer practice 
1. to first start, build and configure the TVB App,
2. then, the NEST App,
3. and only then the Transformers' Apps.
 
TODO: Find a better way to determine these parameters, preconfigure them, and provide them to all processes, making priorities clear, and avoiding possible conflicts and errors.

