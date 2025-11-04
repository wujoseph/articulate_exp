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

cd /work/u9497859/projects/vggt
srun pixi run python ./glb_converter/vggt_to_glb.py --image_folder /work/u9497859/shared_data/partnet-mobility-v0/dataset/1028/multiview --output_dir /work/u9497859/shared_data/partnet-mobility-v0/dataset/1028/vggt_infer \
    --conf_thres 30.0 \
    --show_cam False \
    --mask_white_bg \
    --output_name custom_scene.glb