#!/bin/bash
#
# Sept. 6, 2022
#
# Author : Joe Schoonover ( joe@fluidnumerics.com )
#
# This script downloads the empirical roofline toolkit and runs
# the benchmarks defined by the configuration file in this directory
#
# The ERT is only run with post-processing, but plots are not created
# since gnuplot is not avaiable on Mulan (ERT uses gnuplot to create plots)
#
# You should plan to download the results directory to your local workstation
# and plot the results by running ert on your local workstation with
#
#   /path/to/ert --no-build --no-run --gnuplot ./Mulan-MI100
#
# /////////////////////////////////////////////////////////////////////// #
#
#SBATCH -n 32
#SBATCH --sockets-per-node=1 
#SBATCH --gres=gpu:1
#SBATCH --partition=workq 
#SBATCH --time=01:00:00

# Get the current working directory
cwd=$(pwd)

# Download ert
git clone git@bitbucket.org:berkeleylab/cs-roofline-toolkit.git ${cwd}/cs-roofline-toolkit

${cwd}/cs-roofline-toolkit/Empirical_Roofline_Tool-1.1.0/ert --no-gnuplot Mulan-MI100
