#!/bin/bash

set -x 
cp ../rocprof-basic.txt rocprof-$SLURM_PROCID.txt
mkdir -p $2

rocprof -i rocprof-$SLURM_PROCID.txt --stats --sys-trace ./exess $1 > $2/log-$SLURM_PROCID.txt

mv rocprof-$SLURM_PROCID* $2/
