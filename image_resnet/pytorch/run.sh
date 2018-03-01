export PATH=/home/vis/ssd1/yancanxiang/code_pytorch/Anaconda2-4.3/bin/:$PATH
#export CUDA_VISBILE_DEVICES=0
export CUDA_VISBILE_DEVICES=0,1,2,3,4,5,6,7
python train_resnet.py --batch-size=384
