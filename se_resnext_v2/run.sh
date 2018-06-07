#!/bin/bash
# This script benchmarking the PaddlePaddle Fluid on
# single thread single GPU.

#export FLAGS_fraction_of_gpu_memory_to_use=0.0
export CUDNN_PATH=/paddle/cudnn_v5

# disable openmp and mkl parallel
#https://github.com/PaddlePaddle/Paddle/issues/7199
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
ht=`lscpu |grep "per core"|awk -F':' '{print $2}'|xargs`
if [ $ht -eq 1 ]; then # HT is OFF
    if [ -z "$KMP_AFFINITY" ]; then
        export KMP_AFFINITY="granularity=fine,compact,0,0"
    fi
    if [ -z "$OMP_DYNAMIC" ]; then
        export OMP_DYNAMIC="FALSE"
    fi
else # HT is ON
    if [ -z "$KMP_AFFINITY" ]; then
        export KMP_AFFINITY="granularity=fine,compact,1,0"
    fi
fi
# disable multi-gpu if have more than one
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH
export PYTHONPATH=/paddle/Paddle/build/python/build/lib-python:$PYTHONPATH
export FLAGS_fraction_of_gpu_memory_to_use=0.0

sudo rm -y train.log mem.log
# only query the gpu used
nohup stdbuf -oL nvidia-smi \
      --id=${CUDA_VISIBLE_DEVICES} \
      --query-compute-apps=pid,used_memory \
      --format=csv \
      --filename=mem.log  \
      -l 1 &

stdbuf -oL python train.py \
               --iterations=10 \
               2>&1 | tee -a train.log
