#!/bin/bash
#SBATCH --job-name=NGC4030
#SBATCH --output=job_NGC4030.log
#SBATCH --error=job_NGC4030.error.log
#SBATCH --ntasks=50
#SBATCH --time=5-00:00:00
#SBATCH --export=ALL

cd $SLURM_SUBMIT_DIR

# Note #1: Requires mpi4py > 3.0.0.
# Note #2: It is a little tricky to get slurm and mpi4py to play along.
#          You should compile mpi4py manually and link to the same mpi implementation as used by slurm.
# Note #3: DONT REMOVE this part: "-m mpi4py.futures".
#          This is needed for MPI to separate the master/workers properly.

srun --mpi=pmi2 python -m mpi4py.futures $HOME/pycasso2/scripts/pycasso_starlight_mpi.py \
	--config muse.v1.cfg --name=NGC4030.v1.syn --max-workers=${SLURM_NPROCS} \
	--out cubes/NGC4030.v1.CB17.fits cubes/NGC4030.v1.fits
