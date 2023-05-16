export MMDEPLOY_DIR="/opt/workspace/mmdeploy"
export PATH="$MMDEPLOY_DIR/mmdeploy-1.0.0-linux-x86_64-cuda11.3/lib":$PATH

export PPLCV_DIR="$MMDEPLOY_DIR/third_party/ppl.cv"

export CUDNN_DIR=/usr/lib/x86_64-linux-gnu

export ONNXRUNTIME_DIR="$MMDEPLOY_DIR/onnxruntime-linux-x64-gpu-1.8.1"

export TENSORRT_DIR="$MMDEPLOY_DIR/TensorRT-8.2.3.0"

export CUDA_DIR="/usr/local/cuda"

export PATH="$CUDA_DIR/bin:$PATH"

export PATH="$TENSORRT_DIR/lib:$PATH"

export PATH="$TENSORRT_DIR/bin:$PATH"

export PATH="$ONNXRUNTIME_DIR/bin:$PATH"



export LD_LIBRARY_PATH="$CUDA_DIR/lib:$LD_LIBRARY_PATH"

export LD_LIBRARY_PATH="$TENSORRT_DIR/lib:$LD_LIBRARY_PATH"

export LD_LIBRARY_PATH="$TENSORRT_DIR/bin:$LD_LIBRARY_PATH"

export LD_LIBRARY_PATH="$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH"

export LD_LIBRARY_PATH="$CUDA_DIR/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"




