#!/bin/bash
# OpenCV
cd /tmp && mkdir opencvbuild && cd opencvbuild
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.3.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.3.zip
unzip opencv.zip && unzip opencv_contrib.zip
cd opencv-4.5.3 && mkdir build && cd build
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
    -D CUDA_ARCH_BIN=8.0 \
    -D WITH_CUBLAS=1 \
    -D WITH_OPENGL=ON \
    -D WITH_QT=ON \
    -D OpenGL_GL_PREFERENCE=LEGACY \
    -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencvbuild/opencv_contrib-4.5.3/modules \
    -D PYTHON_DEFAULT_EXECUTABLE=python3 \
    -D BUILD_EXAMPLES=ON ..
sudo make install -j $(nproc) && sudo ldconfig
cd && sudo rm -rf /tmp/opencvbuild

# Darknet
cd && git clone https://github.com/AlexeyAB/darknet && cd darknet
mkdir build_release && cd build_release
cmake ..
cmake --build . --target install --parallel $(nproc)
cd ..
