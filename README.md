# Benchmark of opensource Platforms
Machine:

- Server: Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz, 2 Sockets, 20 Cores per socket

- CPU environment
  System: Ubuntu 16.04.3 LTS, Docker 17.05.0-ce, build 89658be
- GPU environment
  System: Ubuntu 16.04.3 LTS, NVIDIA-Docker 17.05.0-ce, build 89658be
  NVIDIA Docker image: nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

PaddlePaddle: 0.11.0(Fluid) 
- paddlepaddle/paddle:latest

TensorFlow: 1.4.0
- tensorflow/tensorflow:latest

## Benchmark Model

### selected models PaddlePaddle Fluid vs TensorFlow 
We selected some classic models, compare the performance and speed with TensorFlow. 

|              | train cost | train accuracy | test accuracy | samples/sec | train cost | train accuracy | test accuracy | samples/sec |
| ------------ | ---------- | -------------- | ------------- | ----------- | ---------- | -------------- | ------------- | ----------- |
| MNIST CNN    |            |                |               |             |            |                |               |             |
| VGG-19        |            |                |               |             |            |                |               |             |
| RESNET-101    |            |                |               |             |            |                |               |             |
| Stacked LSTM |            |                |               |             |            |                |               |             |

- TBD
add charts compare here

- VGG-19
input image size - 3 * 224 * 224, Time: images/second

| BatchSize    | 64    | 128   | 256    |
|--------------|-------| ------| -------|
| PaddlePaddle Fluid| | | |
| TensorFlow| | | |

- TBD
add charts compare here

- RESNET-101

| BatchSize    | 64    | 128  | 256     |
|--------------|-------| -----| --------|
| PaddlePaddle Fluid| | | |
| TensorFlow| | | |

- TBD

add charts here

- Stacked LSTM

| BatchSize    | 64    | 128  | 256     |
|--------------|-------| -----| --------|
| PaddlePaddle Fluid| | | |
| TensorFlow| | | |

- TBD

add charts here


### PaddlePaddle books Fluid vs Paddle 0.10.0 
To validate the Fluid performance on general models, we choose the models in book chapter, compare the performance and speed with Paddle 0.10.0.

|                         | train cost | train accuracy | test accuracy | samples/sec | train cost | train accuracy | test accuracy | samples/sec |
| ----------------------- | ---------- | -------------- | ------------- | ----------- | ---------- | -------------- | ------------- | ----------- |
| 01.fit_a_line           |            |                |               |             |            |                |               |             |
| 02.recognize_digits     |            |                |               |             |            |                |               |             |
| 03.image_classification |            |                |               |             |            |                |               |             |
| 04.word2vec             |            |                |               |             |            |                |               |             |
| 05.recommender_system   |            |                |               |             |            |                |               |             |
| 06.understand_sentiment |            |                |               |             |            |                |               |             |
| 07.label_semantic_roles |            |                |               |             |            |                |               |             |
| 08.machine_translation  |            |                |               |             |            |                |               |             |

