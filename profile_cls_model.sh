date_string=$(date +"%Y%m%d")
cvt_config="/opt/workspace/mmdeploy/configs/mmpretrain/classification_tensorrt_static-224x224-32.py"
cvt_config="/opt/workspace/mmdeploy/configs/mmpretrain/classification_tensorrt-fp16_dynamic-256x256-256x256.py"

work_dir="work_dirs/mmpretrain_repvgg_cls_${date_string}"

model_config="/opt/workspace/mmpretrain-1/work_dirs/palmdate_repvgg/20230702_144150/vis_data/config.py"
model_config="/opt/workspace/mmpretrain-1/work_dirs/palmdate_repvgg/20230820_155531_5cat/vis_data/config.py"

model_file="/opt/workspace/mmpretrain-1/work_dirs/palmdate_repvgg/best_accuracy_top1_epoch_120.pth"
source set_env.sh

python tools/profiler.py \
$cvt_config \
$model_config \
"/nas/win_essd/UAE_sliced_256/fardh-重量/B/slice/" \
--model /opt/workspace/mmdeploy/work_dirs/mmpretrain_repvgg_cls_20230821/end2end.engine \
--device cuda \
--shape 256x256 \
--num-iter 10000 \
--warmup 10 \
--batch-size 32
