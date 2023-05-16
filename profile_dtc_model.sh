python tools/profiler.py \
./configs/mmdet/detection/detection_tensorrt-fp16_dynamic-196x196-224x224.py \
./work_dirs/rtmdet-tiny-palmdate_20230512/model_config.py \
"/opt/workspace/mmdetection-1/data/20230511_181415/DatasetId_1822463_1682699165/Images/" \
--model ./work_dirs/rtmdet-tiny-palmdate_20230512/end2end.engine \
--device cuda \
--shape 224x224 \
--num-iter 1000 \
--warmup 10 \
--batch-size 1
