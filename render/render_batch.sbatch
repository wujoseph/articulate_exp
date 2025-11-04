#!/usr/bin/env bash

#SBATCH --account=MST114029
#SBATCH --partition=ct56
#SBATCH --job-name=render_job
#SBATCH --output=logs/output.out
#SBATCH --error=logs/error.err
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=80G
#SBATCH --mail-user=wujoseph12345@gmail.com
#SBATCH --mail-type=ALL

# twnia3 has 28core per node


srun blender -b --python render_cycle.py -- /work/u9497859/shared_data/partnet-mobility-v0/dataset/1028/textured_objs /work/u9497859/shared_data/partnet-mobility-v0/dataset/1028/multiview
srun blender -b --python render_cycle.py -- /work/u9497859/shared_data/partnet-mobility-v0/dataset/1034/textured_objs /work/u9497859/shared_data/partnet-mobility-v0/dataset/1034/multiview
srun blender -b --python render_cycle.py -- /work/u9497859/shared_data/partnet-mobility-v0/dataset/1052/textured_objs /work/u9497859/shared_data/partnet-mobility-v0/dataset/1052/multiview
srun blender -b --python render_cycle.py -- /work/u9497859/shared_data/partnet-mobility-v0/dataset/1053/textured_objs /work/u9497859/shared_data/partnet-mobility-v0/dataset/1053/multiview
srun blender -b --python render_cycle.py -- /work/u9497859/shared_data/partnet-mobility-v0/dataset/1280/textured_objs /work/u9497859/shared_data/partnet-mobility-v0/dataset/1280/multiview
srun blender -b --python render_cycle.py -- /work/u9497859/shared_data/partnet-mobility-v0/dataset/1288/textured_objs /work/u9497859/shared_data/partnet-mobility-v0/dataset/1288/multiview
srun blender -b --python render_cycle.py -- /work/u9497859/shared_data/partnet-mobility-v0/dataset/1343/textured_objs /work/u9497859/shared_data/partnet-mobility-v0/dataset/1343/multiview
srun blender -b --python render_cycle.py -- /work/u9497859/shared_data/partnet-mobility-v0/dataset/1370/textured_objs /work/u9497859/shared_data/partnet-mobility-v0/dataset/1370/multiview
srun blender -b --python render_cycle.py -- /work/u9497859/shared_data/partnet-mobility-v0/dataset/1380/textured_objs /work/u9497859/shared_data/partnet-mobility-v0/dataset/1380/multiview
srun blender -b --python render_cycle.py -- /work/u9497859/shared_data/partnet-mobility-v0/dataset/1386/textured_objs /work/u9497859/shared_data/partnet-mobility-v0/dataset/1386/multiview
