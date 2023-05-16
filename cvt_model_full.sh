source set_env.sh
echo $PATH
date_string=$(date +"%Y%m%d")
cvt_config="/opt/workspace/mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py"
work_dir="work_dirs/rtmdet-tiny-ins-fullsize_single_cat_${date_string}"
model_config="/opt/workspace/mmdetection-1/work_dirs/rtmdet-tiny-ins-fullsize_single_cat/20230508_122358/vis_data/config.py"
model_file="/opt/workspace/mmdetection-1/work_dirs/rtmdet-tiny-ins-fullsize_single_cat/best_coco_segm_mAP_epoch_70.pth"

python tools/deploy.py \
$cvt_config \
$model_config \
$model_file \
"/nas/ai_image/sync_image/baidu_pan_download/UAE_PD/椰枣-王爷20230429/Muneif_loose_air/23-05-10 14-09-06-114.bmp" \
--work-dir ${work_dir} \
--device cuda     \
--dump-info
cp $model_config ${work_dir}/model_config.py
echo "cvt_config:$cvt_config \
model_config:$model_config \
model_file:$model_file \
work_dir:$work_dir \
device:cuda \
" >> ${work_dir}/cvt_log.txt
