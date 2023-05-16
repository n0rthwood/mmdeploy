conda create -n mmdp1 python=3.8 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install -U openmim
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

pip install mmdeploy-runtime==1.0.0
pip install mmdeploy-runtime-gpu==1.0.0
pip install TensorRT-8.2.3.0/python/tensorrt-8.2.3.0-cp38-none-linux_x86_64.whl
pip install pycuda
pip install onnxruntime-gpu==1.8.1
pip3 install -v -e .




