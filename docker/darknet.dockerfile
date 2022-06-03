FROM nvcr.io/nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# up-to-date CMake
RUN apt-get update \
    && apt-get install --no-install-recommends -yqq gpg wget \
    && rm -rf /var/lib/apt/lists/* \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null

# OpenCV
RUN apt-get update \
    && apt-get install --no-install-recommends -yqq \
    python3 python3-dev python3-pip python3-numpy \
    build-essential cmake pkg-config unzip yasm git checkinstall wget curl \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libavresample-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev \
    libfaac-dev libmp3lame-dev libvorbis-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev \
    libgtk-3-dev libtbb-dev libatlas-base-dev gfortran \
    libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev \
    libgphoto2-dev libeigen3-dev libhdf5-dev doxygen \
    libgtkglext1 libgtkglext1-dev libopenblas-dev liblapacke-dev \
    libva-dev libopenjp2-tools libopenjpip-dec-server libopenjpip-server \
    libqt5opengl5-dev libtesseract-dev libsuitesparse-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp && git clone https://ceres-solver.googlesource.com/ceres-solver \
    && cd ceres-solver && mkdir build && cd build \
    && cmake .. && make -j $(nproc) && make install && cd .. && rm -rf /tmp/ceres-solver

# entrypoint
RUN apt-get update \
    && apt-get install --no-install-recommends -yqq gosu sudo \
    && rm -rf /var/lib/apt/lists/*

RUN printf '#!/bin/bash \n\
USER_ID=${LOCAL_USER_ID:-9001} \n\
GROUP_ID=${LOCAL_GROUP_ID:-$USER_ID} \n\
groupadd -f -g $GROUP_ID thegroup \n\
useradd --shell /bin/bash -u $USER_ID -g thegroup -o -c "" -m user  || true \n\
export HOME=/home/user \n\
echo user ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/user \n\
chmod 0440 /etc/sudoers.d/user \n\
exec gosu user:thegroup $@' > /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
RUN ln -s /usr/local/bin/entrypoint.sh /
ENTRYPOINT ["entrypoint.sh"]
EXPOSE 8090
