# Benchmark of opensource Platforms
Machine:

- Server: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz, 2 Sockets, 20 Cores per socket

- CPU environment
System: Ubuntu 16.04.3 LTS, Docker 17.05.0-ce, build 89658be
- GPU environment
System: Ubuntu 16.04.3 LTS, NVIDIA-Docker 17.05.0-ce, build 89658be
NVIDIA Docker image: nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

PaddlePaddle: (TODO: will rerun after 0.11.0)
- paddlepaddle/paddle:latest (for MKLML and MKL-DNN)
  - MKL-DNN tag v0.11
  - MKLML 2018.0.1.20171007
- paddlepaddle/paddle:latest-openblas (for OpenBLAS)
  - OpenBLAS v0.2.20

## Benchmark Model

### Server
Test on batch size 64, 128, 256 on Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz

Input image size - 3 * 224 * 224, Time: images/second

- VGG-19

| BatchSize    | 64    | 128  | 256     |
|--------------|-------| -----| --------|
| OpenBLAS     | 7.80  | 9.00  | 10.80  | 
| MKLML        | 12.12 | 13.70 | 16.18  |
| MKL-DNN      | 28.46 | 29.83 | 30.44  |

<img src="figs/vgg-cpu-train.png" width="500">

 - ResNet-50

| BatchSize    | 64    | 128   | 256    |
|--------------|-------| ------| -------|
| OpenBLAS     | 25.22 | 25.68 | 27.12  | 
| MKLML        | 32.52 | 31.89 | 33.12  |
| MKL-DNN      | 81.69 | 82.35 | 84.08  |

<img src="figs/resnet-cpu-train.png" width="500">

 - GoogLeNet

| BatchSize    | 64    | 128   | 256    |
|--------------|-------| ------| -------|
| OpenBLAS     | 89.52 | 96.97 | 108.25 | 
| MKLML        | 128.46| 137.89| 158.63 |
| MKL-DNN      | 250.46| 264.83| 269.50 |

<img src="figs/googlenet-cpu-train.png" width="500">

### Laptop
TBD
