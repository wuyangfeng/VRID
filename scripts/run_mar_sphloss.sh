#!/bin/sh
export TORCH_HOME=/workspace/mnt/group/video/chenshuaijun/pytorch
cd reid/deep-person-reid
/workspace/mnt/group/video/chenshuaijun/pkg/anaconda3/envs/python36/bin/python -u train_sphloss.py -d market1501 -a resnet50 --max-epoch 400 --train-batch 128 --test-batch 128 --stepsize 20 --eval-step 10 --save-dir log/resnet50-sphloss-market1501 --gpu-devices 0,1 -j 8 --root ../datasets/ --height 256 --width 128 --warmup
