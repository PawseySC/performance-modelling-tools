#!/bin/bash

set -x 
cp rocprof-template.txt profile-${SLURM_PROCID}.txt
rocprof -i profile-${SLURM_PROCID}.txt ./s3d.STXgpu.x > log-${SLURM_PROCID}.txt
rocprof -o profile-${SLURM_PROCID}.csv --stats --sys-trace ./s3d.STXgpu.x