wget https://github.com/open-mmlab/mmdeploy/releases/download/v1.0.0rc3/mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1.tar.gz
tar -zxvf mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1.tar.gz
cd mmdeploy-1.0.0rc3-linux-x86_64-onnxruntime1.8.1
pip install dist/mmdeploy-1.0.0rc3-py3-none-linux_x86_64.whl
pip install sdk/python/mmdeploy_python-1.0.0rc3-cp38-none-linux_x86_64.whl
cd ..
# install inference engine: ONNX Runtime
pip install onnxruntime==1.8.1
wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz
tar -zxvf onnxruntime-linux-x64-1.8.1.tgz