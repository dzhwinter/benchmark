# Benchmark SE-ResNeXt-152

## For single card:
```
env CUDA_VISIBLE_DEVICES=0 python train.py --use_parallel_mode=parallel_do --do_profile=False --use_nccl=False --parallel=False --display_step=1
```

## For multi-card:
### use parallel_do (the data is in GPU side.)
```
env CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --use_parallel_mode=parallel_do --do_profile=False  --use_nccl=True --parallel=True --use_python_reader=false  --display_step=1
```
### use parallel_do (use_python_reader, data_flow: CPU->GPU->Training)
```
env CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --use_parallel_mode=parallel_do --do_profile=False  --use_nccl=True --parallel=True --use_python_reader=true --display_step=1
```

###  use parallel_exe (use C++ reader)
```
env CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --use_parallel_mode=parallel_exe --do_profile=False --display_step=1
```
#### use parallel_exe (use feeder, the data is in GPU side) 
```
env CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --use_parallel_mode=parallel_exe --use_feeder=true --use_python_reader=false --do_profile=false --display_step=1
```
#### use parallel_exe (use feeder and use_python_readear, data flow: CPU->GPU->Training)  
```
env CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --use_parallel_mode=parallel_exe --use_feeder=true --use_python_reader=true --do_profile=false --display_step=1
```
