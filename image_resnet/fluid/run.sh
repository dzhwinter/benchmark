alias python=./python-gcc482-paddle/bin/python
export LD_LIBRARY_PATH=/home/work/cuda-8.0/lib64/:/home/work/cudnn/cudnn_v5.1/cuda/lib64/:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0
python se_resnext152_parallel.py
