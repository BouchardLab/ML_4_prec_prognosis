#!/bin/bash
#SBATCH --qos=debug
#SBATCH --nodes=4
#SBATCH --time=10:00
#SBATCH --licenses=cfs,cscratch1
#SBATCH --constraint=haswell

################################################################
# Haswell has 32 physical cores and 64 logical cores per node
################################################################

ACTIV_DIR="$HOME/projects/activ"

SCRIPT="$ACTIV_DIR/mmda.git/scripts/run_cca_perm.py"
N_PERM=1024         # 16 nodes 
N_PERM=128
SEED=1001

srun -n $N_PERM python $SCRIPT $INPUT $OUTPUT $N_PERM $SEED
