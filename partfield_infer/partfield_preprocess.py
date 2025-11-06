

import sys
import os
import shutil

"""
run this file example:

conda activate articulate_exp
partfield_preprocess.py -- Faucet
"""

# Add the lib directory to the path
argv = sys.argv
argv = argv[argv.index("--") + 1:]
class_name = argv[0]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from dataset_utils import get_dataset_dict
id_dict = get_dataset_dict()
assert class_name in id_dict, f"Not Exist CLASS {class_name}"

target_path = f"/work/u9497859/shared_data/partnet-mobility-pointcloud/{class_name}"
# Check if target_path directory exists, if not, create it
os.makedirs(target_path, exist_ok=True)

for id in id_dict[class_name][:]:
    print(f"start process id:{id} object")
    data_file = f"/work/u9497859/shared_data/partnet-mobility-v0/dataset/{id}/point_sample/ply-10000.ply"
    # Copy file from data_file to target_path
    if os.path.exists(data_file):
        shutil.copy2(data_file, os.path.join(target_path, f"{id}.ply"))
        print(f"  Copied {id}.ply to {target_path}")
    else:
        print(f"  Warning: {data_file} does not exist")
