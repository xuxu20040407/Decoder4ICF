#!/bin/bash

#SBATCH --job-name=R10-Simu       # Name
#SBATCH --partition=64c512g          # cpu
#SBATCH -n 50                # total node
#SBATCH --ntasks-per-node=50     # core per node
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

module load miniconda3/4.10.3
source activate multinet
which python
nproc
python ReinforceLearning_new.py
