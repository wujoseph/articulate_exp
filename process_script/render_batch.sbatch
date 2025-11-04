#!/usr/bin/env bash

#SBATCH --account=MST114029
#SBATCH --partition=gp1d
#SBATCH --job-name=render_job
#SBATCH --output=logs/output.out
#SBATCH --error=logs/error.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=40G
#SBATCH --mail-user=wujoseph12345@gmail.com
#SBATCH --mail-type=ALL


srun blender -b --python render_cycle.py -- /work/u9497859/shared_data/partnet-mobility-v0/dataset/1028/textured_objs /work/u9497859/shared_data/partnet-mobility-v0/dataset/1028/multiview
