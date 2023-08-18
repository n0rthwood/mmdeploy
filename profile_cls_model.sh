date_string=$(date +"%Y%m%d")
cvt_config="/opt/workspace/mmdeploy/configs/mmpretrain/classification_tensorrt_static-224x224-32.py"
work_dir="work_dirs/mmpretrain_repvgg_cls_${date_string}"
model_config="/opt/workspace/mmpretrain-1/work_dirs/palmdate_repvgg/20230702_144150/vis_data/config.py"
model_file="/opt/workspace/mmpretrain-1/work_dirs/palmdate_repvgg/best_accuracy_top1_epoch_120.pth"
source set_env.sh

python tools/profiler.py \
$cvt_config \
$model_config \
"/opt/images/UAE/khalash/21-11-09_09-34-06/slice/" \
--model /opt/workspace/mmdeploy/work_dirs/mmpretrain_repvgg_cls_20230702/end2end.engine \
--device cuda \
--shape 224x224 \
--num-iter 10000 \
--warmup 10 \
--batch-size 32
