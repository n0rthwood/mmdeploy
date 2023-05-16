
# install dependencie
# if gcc-7 is installed skip this

sudo apt-get update
sudo apt-get install gcc-7 -y
sudo apt-get install g++-7 -y
sudo apt-get install libopencv-dev -y

source set_env.sh
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=ort -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} ..
make -j$(nproc) && make install

cmake -DCMAKE_CXX_COMPILER=g++-7 -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} -DCUDNN_DIR=${CUDNN_DIR} ..
make -j$(nproc) && make install

cmake .. \
    -DCMAKE_CXX_COMPILER=g++-7 \
    -DMMDEPLOY_BUILD_SDK=ON \
    -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
    -DMMDEPLOY_BUILD_EXAMPLES=ON \
    -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
    -DMMDEPLOY_TARGET_BACKENDS=trt \
    -Dpplcv_DIR=${PPLCV_DIR}/cuda-build/install/lib/cmake/ppl \
    -DTENSORRT_DIR=${TENSORRT_DIR} \
    -DCUDNN_DIR=${CUDNN_DIR}

make -j$(nproc) && make install

