# install MMDeploy
wget https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0rc3/mmdeploy-1.0.0rc3-linux-x86_64-cuda11.1-tensorrt8.2.3.0.tar.gz
tar -zxvf mmdeploy-1.0.0rc3-linux-x86_64-cuda11.1-tensorrt8.2.3.0.tar.gz
cd mmdeploy-1.0.0rc3-linux-x86_64-cuda11.1-tensorrt8.2.3.0
pip install dist/mmdeploy-1.0.0rc3-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-1.0.0rc3-cp38-none-linux_x86_64.whl
cd ..
# install inference engine: TensorRT
# !!! Download TensorRT-8.2.3.0 CUDA 11.x tar package from NVIDIA, and extract it to the current directory
pip install TensorRT-8.2.3.0/python/tensorrt-8.2.3.0-cp38-none-linux_x86_64.whl
pip install pycuda
export TENSORRT_DIR=$(pwd)/TensorRT-8.2.3.0
export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH
# !!! Download cuDNN 8.2.1 CUDA 11.x tar package from NVIDIA, and extract it to the current directory
export CUDNN_DIR=$(pwd)/cuda
export LD_LIBRARY_PATH=$CUDNN_DIR/lib64:$LD_LIBRARY_PATH

sudo apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
