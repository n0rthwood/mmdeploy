source set_env.sh
echo $PATH
date_string=$(date +"%Y%m%d")
cvt_config="/opt/workspace/mmdeploy/configs/mmpretrain/classification_tensorrt_static-224x224-32.py"
cvt_config="/opt/workspace/mmdeploy/configs/mmpretrain/classification_tensorrt-fp16_dynamic-224x224-224x224.py"
cvt_config="/opt/workspace/mmdeploy/configs/mmpretrain/classification_tensorrt-fp16_dynamic-256x256-256x256.py"
work_dir="work_dirs/mmpretrain_repvgg_cls_${date_string}"
model_config="/opt/workspace/mmpretrain-1/work_dirs/palmdate_repvgg/20230820_155531_5cat/vis_data/config.py"
model_file="/opt/workspace/mmpretrain-1/work_dirs/palmdate_repvgg/epoch_400.pth"

echo "$model_config"
python tools/deploy.py \
$cvt_config \
$model_config \
$model_file \
"/nas/win_essd/UAE_sliced_256/fardh-重量/B/slice/15348-21-11-17 11-53-05-849_0.png" \
--work-dir ${work_dir} \
--device cuda     \
--dump-info

echo "cvt_config:$cvt_config \
model_config:$model_config \
model_file:$model_file \
work_dir:$work_dir \
device:cuda \
" >> ${work_dir}/cvt_log.txt
cp $model_config ${work_dir}/model_config.py