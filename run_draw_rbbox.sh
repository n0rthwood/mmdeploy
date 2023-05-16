#!/bin/bash

source set_env.sh

# Parse command line arguments
input_image_path="$1"

# Create a base output path with "debugg_draw" and datetime string
current_datetime=$(date +"%Y%m%d_%H%M%S")
input_basename=$(basename "${input_image_path}")
base_output_path=$(dirname "${input_image_path}")"${input_basename}_debugg_draw_${current_datetime}"

echo base output path : $base_output_path
# Walk through the image_path and get the dirname of each of the sub_directories
find "${input_image_path}" -mindepth 1 -type d | while read -r sub_dir; do

    echo "working in $sub_dir"
    # Get the relative sub_directory path
    relative_sub_dir=${sub_dir#${input_image_path}}
    relative_sub_dir=${relative_sub_dir#/}

    # Create the corresponding output sub_directory
    output_sub_dir="${base_output_path}/${relative_sub_dir}"
    mkdir -p "${output_sub_dir}"

    # Call the Python script with the input_path and output_path parameters
    find "${sub_dir}" -type f -iname "*.jpg" -o -iname "*.bmp" -o -iname "*.png" | while read -r image; do
        python tools/draw_rrbox.py cuda \
        /opt/workspace/mmdeploy/work_dirs/rtmdet-tiny-ins-fullsize_single_cat_20230511 \
        "${image}" \
        "${output_sub_dir}"
    done
done
