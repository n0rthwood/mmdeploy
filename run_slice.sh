#!/bin/bash

source set_env.sh

input_image_path=$1

datetime_str=$(date +"%Y%m%d_%H%M%S")
output_base_path=$(dirname "${input_image_path}")/output_${datetime_str}

for dir in "${input_image_path}"/*/; do
  subdir_name=$(basename "${dir}")
  output_path="${output_base_path}/${subdir_name}"
  mkdir -p "${output_path}"
  python tools/slice_with_model.py \
  cuda \
  /opt/workspace/mmdeploy/work_dirs/rtmdet-tiny-ins-fullsize_single_cat_20230511 \
  "${dir}" \
  --output_path "${output_path}"
done
