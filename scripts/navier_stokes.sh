#!/bin/bash -l
#SBATCH --job-name="Navier3D"
#SBATCH --output=Navier3D.%j.o
#SBATCH --error=Navier3D.%j.e
#SBATCH --time=24:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

module load daint-gpu
module load Julia/1.9.3-CrayGNU-21.09-cuda

export MPICH_RDMA_ENABLED_CUDA=0
export IGG_CUDAAWARE_MPI=0

srun -n8 bash -c 'julia Navier_Stokes_3d_multixpu_sphere.jl'