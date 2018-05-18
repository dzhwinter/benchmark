# Benchmark SE-ResNeXt-152

## For single card:
```
env CUDA_VISIBLE_DEVICES=0 python train.py --parallel_mode=parallel_do --display_step=1
```

## For multi-card:
### use parallel_do (the data is in GPU side.)
```
env CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --parallel_mode=parallel_do --fix_data_in_gpu=true --display_step=1 
```
### use parallel_do (data_flow: CPU->GPU->Training)
```
env CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --parallel_mode=parallel_do --display_step=1
```

###  use parallel_exe (use C++ reader)
```
env CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --parallel_mode=parallel_exe --use_recordio=True --display_step=1
```
#### use parallel_exe (the data is in GPU side) 
```
env CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --parallel_mode=parallel_exe --fix_data_in_gpu=true --display_step=1 
```
#### use parallel_exe (data flow: CPU->GPU->Training)  
```
env CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --parallel_mode=parallel_exe --display_step=1 
```
