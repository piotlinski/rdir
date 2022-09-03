#!/bin/bash
# OpenCV
cd /tmp && mkdir opencvbuild && cd opencvbuild
git clone https://github.com/opencv/opencv
git clone https://github.com/opencv/opencv_contrib
unzip opencv.zip && unzip opencv_contrib.zip
cd opencv && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D WITH_TBB=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D CUDA_ARCH_BIN=8.6 \
    -D WITH_CUBLAS=1 \
    -D WITH_OPENGL=ON \
    -D WITH_QT=ON \
    -D OpenGL_GL_PREFERENCE=LEGACY \
    -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencvbuild/opencv_contrib/modules \
    -D PYTHON_DEFAULT_EXECUTABLE=python3 \
    -D BUILD_EXAMPLES=ON ..
make -j $(nproc)
sudo make install -j $(nproc) && sudo ldconfig
cd && sudo rm -rf /tmp/opencvbuild

# Darknet
cd && git clone https://github.com/AlexeyAB/darknet && cd darknet
mkdir build_release && cd build_release
cmake ..
cmake --build . --target install --parallel $(nproc)
cd ..
