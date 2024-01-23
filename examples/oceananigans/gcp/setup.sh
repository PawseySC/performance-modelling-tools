#!/bin/bash
#
# Run all of this on a compute node

cwd=$(pwd)

# Install julia
curl -fsSL https://install.julialang.org | sh

# The julia installation updates environment variable definitions in .bashrc
# Source the .bashrc to update environment
source ~/.bashrc

# Install OceanScalingTests.jl
git clone https://github.com/simone-silvestri/OceanScalingTests.jl.git ${cwd}/OceanScalingTests

# Instantiate Oceananigans dependencies
cd ${cwd}/OceanScalingTests/
git checkout 
julia --project=. -e "import Pkg; Pkg.instantiate()"
