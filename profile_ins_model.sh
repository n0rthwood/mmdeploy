python tools/profiler.py \
configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py \
work_dirs/config.py \
"/nas/ai_image/sync_image/baidu_pan_download/UAE_PD/椰枣-王爷20230429/sukari_loose_air_skin/" \
--model work_dirs/end2end.engine \
--device cuda \
--shape 640x640 \
--num-iter 1000 \
--warmup 10 \
--batch-size 1
