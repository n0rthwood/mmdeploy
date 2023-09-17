source set_env.sh
echo $PATH
date_string=$(date +"%Y%m%d")
cvt_config="/opt/workspace/mmdeploy/configs/mmpretrain/classification_torchscript.py"

work_dir="work_dirs/mmpretrain_repvgg_cls_medjool_${date_string}"
model_config="/opt/workspace/mmpretrain-1/work_dirs/medjool_cls_RepVGG-image_size_256_batch_size256-datasettrain_folder-batchsize_256-maxep_1000-joysort-ai-server/2023-08-31_18-36-21/out/config.py"
model_file="/opt/workspace/mmpretrain-1/work_dirs/medjool_cls_RepVGG-image_size_256_batch_size256-datasettrain_folder-batchsize_256-maxep_1000-joysort-ai-server/2023-08-31_18-36-21/out/epoch_290.pth"

model_config="/opt/workspace/mmpretrain-1/work_dirs/medjool_cls_RepVGG-image_size_256_batch_size256-datasettrain_folder-batchsize_256-maxep_1000-joysort-ai-server/2023-09-02_06-41-13/out/20230902_064138/vis_data/config.py"
model_file="/opt/workspace/mmpretrain-1/work_dirs/medjool_cls_RepVGG-image_size_256_batch_size256-datasettrain_folder-batchsize_256-maxep_1000-joysort-ai-server/2023-09-02_06-41-13/out/epoch_530.pth"

model_config="/opt/workspace/mmpretrain-1/work_dirs/medjool_cls_RepVGG-image_size_256_batch_size256-datasettrain5-batchsize_256-maxep_1000-joysort-ai-server/2023-09-08_21-16-13/out/config.py"
model_file="/opt/workspace/mmpretrain-1/work_dirs/medjool_cls_RepVGG-image_size_256_batch_size256-datasettrain5-batchsize_256-maxep_1000-joysort-ai-server/2023-09-08_21-16-13/out/epoch_460.pth"


echo "$model_config"
python tools/deploy.py \
$cvt_config \
$model_config \
$model_file \
"assets/animal_feed.png" \
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