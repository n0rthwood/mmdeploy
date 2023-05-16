
source set_env.sh

date_string=$(date +"%Y%m%d")
work_dir="work_dirs/rtmdet-tiny-palmdate_${date_string}"
model_config="/opt/workspace/mmdetection-1/work_dirs/rtmdet-tiny-palmdate/20230511_225047/vis_data/config.py"
model_file="/opt/workspace/mmdetection-1/work_dirs/rtmdet-tiny-palmdate/best_coco_bbox_mAP_epoch_260.pth"
cvt_config="/opt/workspace/mmdeploy/configs/mmdet/detection/detection_tensorrt-fp16_dynamic-196x196-224x224.py"

python tools/deploy.py \
$cvt_config \
$model_config \
$model_file \
"/opt/workspace/mmdetection-1/data/20230511_181415/DatasetId_1822463_1682699165/Images/1bb043ad-8544-21-11-11_18-59-33-692_0_sliced_image_r1_c3.jpg" \
--work-dir ${work_dir} \
--device cuda     \
--dump-info

cp $model_config ${work_dir}/model_config.py
echo "$date_string cvt_config:$cvt_config \
model_config:$model_config \
model_file:$model_file \
work_dir:$work_dir \
device:cuda \
" >> ${work_dir}/cvt_log.txt