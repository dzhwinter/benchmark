## Overview
本文介绍如何给PaddlePaddle Fluid做benchmark. [Fluid](https://github.com/PaddlePaddle/Paddle/releases/tag/v0.11.0) 是PaddlePaddle 0.11.0引入的新设计, 用来让用户像Pytorch和Tensorflow Eager Execution一样执行程序.
为了验证Fluid的训练效果和性能, 从经典模型和books 两方面对Fluid进行基准测试.

### 单机单卡
下文中的对比实验都在相同的硬件条件和软件条件下进行. 本教程的测试环境为

Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHz. TITAN X (Pascal) 12G x 1

System: Ubuntu 16.04.3 LTS, Nvidia-Docker 17.05.0-ce, build 89658be. Nvidia-Driver 384.90.

### 经典模型在Fluid(0.11.0) 和TensorFlow上的对比

benchmark需要兼顾大小模型，不同训练任务下的表现. 其中mnist, VGG, Resnet属于CNN模型, stacked-lstm代表RNN模型. 
选取了train cost, test accuracy, train speed几个维度来度量Fluid.

Note: 

TensorFlow使用了随机算法[philox](https://github.com/tensorflow/tensorflow/blob/52dcb2590bb9274262656c958c105cb5e5cc1300/tensorflow/core/lib/random/philox_random.h#L16)初始化，网络结构相同下，随机初始化引入的差异也很大. 因此Fluid单个batch对齐比较困难, 只对比固定轮数后的收敛结果.

TensorFlow conv2d 默认格式为“NHWC”, 对“NCHW”只支持预测. Fluid conv2d默认格式为“NCHW”.

如果需要测试TensorFlow [eager Execution](https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html), 需要安装pip install tf-nightly-gpu, 目前只有tf-nightly-gpu 1.5.0.


为了简化流程，本教程目前提供了基准测试镜像 dzhwinter/benchmark:latest, 包含Fluid, TensorFlow以及GPU环境. 您也可以通过dockerhub获取对应的测试基准镜像.

Fluid
```bash
docker pull paddlepaddle/paddle:0.11.0-gpu
```
TensorFlow
```bash
docker pull tensorflow/tensorflow:1.4.0-gpu
```

- CPU

首先需要屏蔽GPU `export CUDA_VISIBLE_DEVICES=`;

在单机单卡的测试环境中,Fluid需要关闭OpenMP. 设置`export OMP_NUM_THREADS=1`. 设置Device=CPU, 或者在代码中设置CPUPlace().
TensorFlow需要关闭多线程, 设置 intra_op_parallelism_threads=1, inter_op_parallelism_threads=1.
运行过程中可以通过, `nvidia-smi`来校验是否有GPU被使用, 下文GPU同理.

```bash
nvidia-docker run -it --name CASE_NAME --security-opt seccomp=unconfined -v $PWD/benchmark:/benchmark -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu IMAGE_NAME /bin/bash
```
将其中的CASE_NAME和IMAGE_NAME换为对应的名字.


- GPU 

本节需要确认cudnn和cuda版本一致。本教程使用了cudnn5, cuda8. nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
```bash
nvidia-docker run -it --name CASE_NAME --security-opt seccomp=unconfined -v $PWD/benchmark:/benchmark -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu IMAGE_NAME /bin/bash
```

- mnist

- VGG-19

- Resnet-50

- Resnet-152

- stacked-lstm

- 图表生成

TBD

### PaddlePaddle book在Fluid(0.11.0) 和Paddle 0.10.0上的对比

[Paddle book](https://github.com/PaddlePaddle/book)呈现了几种典型应用场景, 为了在更通用的场景下验证Fluid的能力, 对比Fluid和Paddle 0.10.0训练精度和性能.

- Paddle book 0.10.0

Paddle book同样提供了镜像, 运行实例为 https://github.com/PaddlePaddle/book

```bash
docker pull paddlepaddle/book:latest-gpu
```
- Paddle Fluid
在本节测试中，Fluid仍使用前述镜像做基准测试. 运行实例为https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/v2/fluid/tests/book

- 01.fit_a_line

- 02.recognize_digits

- 03.image_classification

- 04.word2vec

- 05.recommender_system

- 06.understand_sentiment

- 07.label_semantic_roles

- 08.machine_translation

### 单机多卡

- TBD

Detail to be added.

### Reference

- PaddlePaddle Fluid [Paddle](https://github.com/PaddlePaddle/Paddle)

- TensorFlow [TensorFlow](https://github.com/tensorflow/tensorflow)
