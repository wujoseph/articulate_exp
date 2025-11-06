#!/usr/bin/env bash

# Move part_feat_*_0.npy files into dataset/<id>/partfield/
# Usage: run this script, it'll move to partfield_dir (which is defined under partfield_infer)

partfield_dir="/work/u9497859/projects/PartField/exp_results/temp"
cd $partfield_dir
for file in part_feat_*_0.npy; do
    # Skip if no files match
    [ -e "$file" ] || continue
    # Extract id (the number between the second and third underscore)
    id=$(echo "$file" | awk -F'_' '{print $3}')
    target_dir="/work/u9497859/shared_data/partnet-mobility-v0/dataset/${id}/partfield"
    mkdir -p "$target_dir"
    mv "$file" "$target_dir/"
done

