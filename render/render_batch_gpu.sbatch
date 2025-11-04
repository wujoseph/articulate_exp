#!/usr/bin/env bash

#SBATCH --account=MST114029
#SBATCH --partition=gp1d
#SBATCH --job-name=gpu_render_job
#SBATCH --output=logs/gpu_output_%j.out
#SBATCH --error=logs/gpu_error_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1 
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH --mail-user=wujoseph12345@gmail.com
#SBATCH --mail-type=ALL

# twnia3 has 28core per node


srun blender -b --python render_cycle_class.py -- Faucet