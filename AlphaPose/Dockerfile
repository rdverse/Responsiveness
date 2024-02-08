#FROM nvidia/cuda:11.6.0-devel-ubuntu20.04
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04
RUN apt clean

RUN DEBIAN_FRONTEND=noninteractive apt update && DEBIAN_FRONTEND=noninteractive apt install -y --allow-change-held-packages \
    ffmpeg \
    git \
    build-essential \
    ninja-build \
    cuda-libraries-11-6 \
    mesa-common-dev \
    libosmesa6 libosmesa6-dev \
    libgles2-mesa-dev \
    libglu1-mesa-dev \
    libgles2-mesa-dev \
    libcublas-11-6 \
    libcublas-dev-11-6 \
    libcusparse-dev-11-6 \
    cuda-nvcc-11-6 \
    libcusolver-dev-11-6 \
    cuda-nvrtc-dev-11-6 \
    libcurand-dev-11-6 \
    cuda-nvml-dev-11-6 \
    libcufft-dev-11-6 \
    cuda-toolkit-11-6 \
    nvidia-cuda-toolkit \
    libyaml-dev

ENV CUDA_HOME='/usr/local/cuda'

RUN DEBIAN_FRONTEND=noninteractive apt -y install python3 python3-pip python3-tk

RUN pip install "torch==1.12.1+cu116" "torchvision==0.13.1" "torchaudio==0.12.1" --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install opencv-contrib-python-headless
RUN pip install pip --upgrade

RUN pip install cython pycocotools

RUN python3 -c "import torch; print(torch.version.cuda)" 

RUN mkdir /build
RUN cd /build && git clone https://github.com/HaoyiZhu/HalpeCOCOAPI.git
RUN cd /build && git clone https://github.com/MVIG-SJTU/AlphaPose.git


RUN cd /build/HalpeCOCOAPI/PythonAPI && python3 setup.py build develop --user

WORKDIR /build/AlphaPose

RUN TORCH_CUDA_ARCH_LIST="6.1;7.5;8.6" \
    PATH=/usr/local/cuda/bin/:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH  \
    python3 setup.py build develop --user

# Fix compatibility issue with numpy >= 1.20
RUN pip install git+https://github.com/H4dr1en/cython_bbox.git@patch-1

RUN apt install bc -y && \
    pip install boto3 && \
    python3 -c "import torchvision.models as tm; tm.resnet152(pretrained=True)"

# Download weights, adapt to what you need
RUN pip install gdown && \
    mkdir -p detector/yolo/data && \
    gdown 1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC -O detector/yolo/data/yolov3-spp.weights && \
    gdown 1S-ROA28de-1zvLv-hVfPFJ5tFBYOSITb -O pretrained_models/halpe26_fast_res50_256x192.pth