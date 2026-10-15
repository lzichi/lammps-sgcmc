#!/usr/bin/env bash


# Modules
module purge
module load gcc/14.2.0-fasrc01
module load cuda/12.9.1-fasrc01
module load cmake cudnn openmpi
module list

eval "$(micromamba shell hook -s bash)"
micromamba activate /n/netscratch/kozinsky_lab/Lab/lzichi/Oct/lammps-custom-output/env

../../../build-a100/lmp -k on g 1 -pk kokkos neigh half -sf kk -in in.lj
