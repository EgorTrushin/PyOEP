#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --job-name=pyscf_fci
#SBATCH --time=1000:00:00
#SBATCH --partition=mem384
#SBATCH --mail-user=egor.trushin@fau.de

export I_MPI_DEBUG=5
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_THREAD_LIMIT=$SLURM_CPUS_PER_TASK
export OMP_STACKSIZE=3000M
ulimit -s unlimited
export MV2_CPU_MAPPING=0-$(( ${num} - 1))

cd $SLURM_SUBMIT_DIR

conda activate basic
python b_fci.py > output
