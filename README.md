# NMLK CV project
Consists of multithread multistream dataloader, people detection via YOLOv8 and positioning via ARUCO.

## Installation

```bash
git clone git@github.com:nkb-tech/nlmk-cv.git
cd nlmk-cv
git submodule update --init --recursive
```

### Install via env
```bash
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt

# Install torch, ultralitics by yourself
python3 -m pip install torch
git clone https://github.com/nkb-tech/ultralytics.git && \
cd ultralytics && \
python3 -m pip install \
    --no-cache \
    --editable \
        ".[export]" \
        albumentations \
        comet \
        pycocotools \
        pytest-cov && \
yolo export model=tmp/yolov8n.pt format=edgetpu imgsz=32 && \
yolo export model=tmp/yolov8n.pt format=ncnn imgsz=32 && \
python3 -m pip install install \
    --no-cache \
        # Requires <= Python 3.10, bug with paddlepaddle==2.5.0 https://github.com/PaddlePaddle/X2Paddle/issues/991
        paddlepaddle==2.4.2 \
        x2paddle \
        # Fix error: `np.bool` was a deprecated alias for the builtin `bool` segmentation error in Tests
        numpy==1.23.5 && \
rm -rf tmp
```

### Install via docker
```bash
sh build.sh
```

## Test multithread dataloader
```bash
cd src
# to show help
python3 dataloader.py --help
```
