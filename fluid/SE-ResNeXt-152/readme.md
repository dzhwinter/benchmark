# Benchmark SE-ResNeXt-152

## For single card:
```
env CUDA_VISIBLE_DEVICES=4 python train.py --use_parallel_mode=parallel_do --use_nccl=False --parallel=False --display_step=1
```

## For multi-card:
### use parallel_do
```
env CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --use_parallel_mode=parallel_do --use_nccl=True --parallel=True --display_step=1
```
###  use parallel_exe
```
env CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --use_parallel_mode=parallel_exe --use_nccl=True --parallel=True --display_step=1
```
