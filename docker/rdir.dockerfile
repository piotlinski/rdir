FROM nvcr.io/nvidia/pytorch:22.06-py3

ARG WANDB_API_KEY

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=$PYTHONPATH:$PWD/src/vendor/yolov4 \
    WANDB_API_KEY=${WANDB_API_KEY}

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
CMD ["/bin/bash"]

RUN apt-get update \
    && apt-get install --no-install-recommends -yqq libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt \
    && rm requirements.txt
