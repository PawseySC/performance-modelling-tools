#!/bin/bash

set -x 
cp rocprof-template.txt profile-$SLURM_PROCID.txt
rocprof -i profile-$SLURM_PROCID.txt --stats --sys-trace ./demo > log-$SLURM_PROCID.txt