#!/bin/bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

#
# USAGE: 
#   a) using defaults <BASELINEPATH>=${HOME} <GITUSERNAME>=multiscale-cosim
#           sh ./TVB_NEST-usecase1_ubuntu_setting_up.sh
#    
#   b) specifiying the parameters   
#        sh ./TVB_NEST_usecase1_ubuntu_setting_up.sh <BASELINEPATH> <GITUSERNAME>
#       e.g. 
#           ./TVB_NEST_usace1_ubuntu_setting_up.sh /opt/MY_COSIM sontheimer



BASELINE_PATH="/home/vagrant"
BASELINE_PATH=${1:-${BASELINE_PATH}}

GIT_DEFAULT_NAME='multiscale-cosim'
GIT_DEFAULT_NAME=${2:-${GIT_DEFAULT_NAME}}

#
# STEP 1 - setting up folder locations
#
echo
echo "STEP 1 - setting up folder locations"
echo

[ -d ${BASELINE_PATH} ] \
	|| (echo "${BASELINE_PATH} does not exists"; exit 1;)

#
# Full base path where installation happends:
#
# CO_SIM_ROOT_PATH = /home/<user>/multiscale-cosim/
# or
# CO_SIM_ROOT_PATH = /home/<user>/<git_account_name>/
#
CO_SIM_ROOT_PATH=${BASELINE_PATH}/${GIT_DEFAULT_NAME}

mkdir -p ${CO_SIM_ROOT_PATH}
cd ${CO_SIM_ROOT_PATH}

# CO_SIM_REPOS=${CO_SIM_ROOT_PATH}/cosim-repos
CO_SIM_SITE_PACKAGES=${CO_SIM_ROOT_PATH}/site-packages
CO_SIM_NEST_BUILD=${CO_SIM_ROOT_PATH}/nest-build
CO_SIM_NEST=${CO_SIM_ROOT_PATH}/nest

#
# STEP 2 - installing linux packages
#
# STEP 2.1 - base packages
echo
echo "STEP 2.1 - installing base packages"
echo
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip
#
# STEP 2.2 - packages used by NEST, TVB and the use-case per se
echo
echo "STEP 2.2 - installing usecase base dependencies packages"
echo
sudo apt install -y doxygen
sudo apt install -y libboost-all-dev libgsl-dev libltdl-dev \
                    libncurses-dev libreadline-dev 
sudo apt install -y mpich


export PATH=/home/vagrant/.local/bin:$PATH
#
# STEP 2.3 - switching the default MPI installed packages to MPICH
echo
echo "STEP 2.3 - switching to MPICH"
echo
#   Selection    Path                     Priority   Status
#------------------------------------------------------------
#* 0            /usr/bin/mpirun.openmpi   50        auto mode
#  1            /usr/bin/mpirun.mpich     40        manual mode
#  2            /usr/bin/mpirun.openmpi   50        manual mode
echo "1" | sudo update-alternatives --config mpi 1>/dev/null 2>&1 # --> choosing mpich
echo "1" | sudo update-alternatives --config mpirun 1>/dev/null 2>&1 # --> choosing mpirun
 
#
# STEP 3 - install python packages for the TVB-NEST use-case
# NOTE the python packages are installed together with TVB in STEP 4
#
#
# STEP 4 - TVB
#

# NOTE: Specific versions are required for some packages
# NOTE The order is important here to let TVB finda and install correct dependencies such as numpy<1.24 for numba0.56
# TVB installation
echo
echo "STEP 3 - installing TVB and python dependencies (specific versions)"
echo
pip install --no-cache --target=${CO_SIM_SITE_PACKAGES} tvb-library==2.8 tvb-contrib==2.7.2 tvb-gdist==2.2 \
							cython elephant mpi4py numpy==1.23.4 pyzmq requests testresources pandas xarray


# jupyter notebook stuff
echo
echo "STEP 5 - installing jupyter"
echo
pip install jupyter markupsafe==2.0.1
#export PATH=/home/vagrant/.local/bin:$PATH

# 
# STEP 5 - cloning github repos
#
# git clone --recurse-submodules --depth 1 --shallow-submodules --jobs 4 https://github.com/${GIT_DEFAULT_NAME}/TVB-NEST-usecase2.git
git clone --recurse-submodules --jobs 4 https://github.com/${GIT_DEFAULT_NAME}/TVB-NEST-usecase2.git

#
# STEP 6 - NEST compilation
# International Neuroinformatics Coordinating Facility (INCF) 
# https://github.com/INCF/MUSIC
# https://github.com/INCF/libneurosim

# Cython
export PATH=${CO_SIM_SITE_PACKAGES}/bin:${PATH}
export PYTHONPATH=${CO_SIM_SITE_PACKAGES}:${PYTHONPATH:+:$PYTHONPATH}

mkdir -p ${CO_SIM_NEST_BUILD}
mkdir -p ${CO_SIM_NEST}

cd ${CO_SIM_NEST_BUILD}
cmake \
    -DCMAKE_INSTALL_PREFIX:PATH=${CO_SIM_NEST} \
    ${CO_SIM_ROOT_PATH}/TVB-NEST-usecase2/nest-simulator/ \
    -Dwith-mpi=ON \
    -Dwith-openmp=ON \
    -Dwith-readline=ON \
    -Dwith-ltdl=ON \
    -Dcythonize-pynest=ON \
    -DPYTHON_EXECUTABLE=/usr/bin/python3.10 \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.10 \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.10.so

make -j 3
make install
cd ${CO_SIM_ROOT_PATH}

#
# STEP 7 - WORK-AROUNDs (just in case)
#
# removing typing.py as work-around for pylab on run-time
rm -f ${CO_SIM_SITE_PACKAGES}/typing.py
#
# proper versions to be used by TVB
# removing (force) the installed versions 
# __? rm -Rf ${CO_SIM_SITE_PACKAGES}/numpy
# __? rm -Rf ${CO_SIM_SITE_PACKAGES}/gdist
# __? pip install --target=${CO_SIM_SITE_PACKAGES} --upgrade --no-deps --force-reinstall --no-cache matplotlib numpy==1.21
# __? pip install --target=${CO_SIM_SITE_PACKAGES} --upgrade --no-deps --force-reinstall gdist==1.0.2

# even though numpy==1.21 coud have been installed,
# other version could be still present and used

if false; then
continue_removing=1
while [ ${continue_removing} -eq 1 ]
do
        pip list | grep numpy | grep -v "1.21" 1>/dev/null 2>&1
        if [ $? -eq 0 ]
        then
                pip uninstall -y numpy 1>/dev/null 2>&1
        else
                continue_removing=0
        fi
done
fi

#
# STEP 8 - Generating the .source file based on ENV variables
#
NEST_PYTHON_PREFIX=`find ${CO_SIM_NEST} -name site-packages`
CO_SIM_USE_CASE_ROOT_PATH=${CO_SIM_ROOT_PATH}/TVB-NEST-usecase2
CO_SIM_MODULES_ROOT_PATH=${CO_SIM_ROOT_PATH}/TVB-NEST-usecase2

SUFFIX_PYTHONPATH="\${PYTHONPATH:+:\$PYTHONPATH}"

cat <<.EOSF > ${CO_SIM_ROOT_PATH}/TVB-NEST-usecase2.source
#!/bin/bash
export CO_SIM_ROOT_PATH=${CO_SIM_ROOT_PATH}
export CO_SIM_USE_CASE_ROOT_PATH=${CO_SIM_USE_CASE_ROOT_PATH}
export CO_SIM_MODULES_ROOT_PATH=${CO_SIM_MODULES_ROOT_PATH}
export PYTHONPATH=${CO_SIM_MODULES_ROOT_PATH}:${CO_SIM_SITE_PACKAGES}:${NEST_PYTHON_PREFIX}${SUFFIX_PYTHONPATH}
export PATH=${CO_SIM_NEST}/bin:${PATH}
.EOSF


# source TVB_NEST_usecase2.source
echo
echo "STEP 9 - sourcing TVB-NEST-usecase2.source"
echo
source ${CO_SIM_ROOT_PATH}/TVB-NEST-usecase2.source

echo
echo "STEP 9 - installing TVB_Multiscale"
echo
python3 ${CO_SIM_USE_CASE_ROOT_PATH}/TVB-multiscale/setup.py develop --user

cd ${CO_SIM_ROOT_PATH}


# 
# STEP 9 - Generating the run_on_local.sh  
cat <<.EORF > ${CO_SIM_ROOT_PATH}/run_on_local.sh
# checking for already set CO_SIM_* env variables
CO_SIM_ROOT_PATH=\${CO_SIM_ROOT_PATH:-${CO_SIM_ROOT_PATH}}
CO_SIM_USE_CASE_ROOT_PATH=\${CO_SIM_USE_CASE_ROOT_PATH:-${CO_SIM_USE_CASE_ROOT_PATH}}
CO_SIM_MODULES_ROOT_PATH=\${CO_SIM_MODULES_ROOT_PATH:-${CO_SIM_MODULES_ROOT_PATH}}
# exporting CO_SIM_* env variables either case
export CO_SIM_ROOT_PATH=\${CO_SIM_ROOT_PATH}
export CO_SIM_USE_CASE_ROOT_PATH=\${CO_SIM_USE_CASE_ROOT_PATH}
export CO_SIM_MODULES_ROOT_PATH=\${CO_SIM_MODULES_ROOT_PATH}
# CO_SIM_ site-packages for PYTHONPATH
export CO_SIM_PYTHONPATH=${CO_SIM_MODULES_ROOT_PATH}:${CO_SIM_SITE_PACKAGES}:${NEST_PYTHON_PREFIX}
# adding EBRAIN_*, site-packages to PYTHONPATH (if needed)
PYTHONPATH=\${PYTHONPATH:-\$CO_SIM_PYTHONPATH}
echo \$PYTHONPATH | grep ${CO_SIM_SITE_PACKAGES} 1>/dev/null 2>&1
[ \$? -eq 0 ] || PYTHONPATH=\${CO_SIM_PYTHONPATH}:\$PYTHONPATH
export PYTHONPATH=\${PYTHONPATH}
# making nest binary reachable
# __ric__? PATH=\${PATH:-$CO_SIM_NEST/bin}
echo \$PATH | grep ${CO_SIM_NEST}/bin 1>/dev/null 2>&1
[ \$? -eq 0 ] || export PATH=$CO_SIM_NEST/bin:\${PATH}
python3 \${CO_SIM_USE_CASE_ROOT_PATH}/main.py \\
    --global-settings \${CO_SIM_MODULES_ROOT_PATH}/EBRAINS_WorkflowConfigurations/general/global_settings.xml \\
    --action-plan \${CO_SIM_MODULES_ROOT_PATH}/EBRAINS_WorkflowConfigurations/usecase/local/plans/cosim_alpha_brunel_local.xml
.EORF

cat <<.EOKF >${CO_SIM_ROOT_PATH}/kill_co_sim_PIDs.sh
for co_sim_PID in \`ps aux | grep TVB-NEST-usecase2 | sed 's/user//g' | sed 's/^ *//g' | cut -d" " -f 1\`; do kill -9 \$co_sim_PID; done
.EOKF

#
# STEP 10 - THIS IS THE END!
#
echo "SETUP DONE!"
