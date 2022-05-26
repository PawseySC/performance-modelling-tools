#!/bin/bash

set -x 
mkdir -p $2

rocprof -o $2/rocprof-$SLURM_PROCID.csv $3 ./exess $1 > $2/log-$SLURM_PROCID.txt
