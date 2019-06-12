#!/bin/sh
export TORCH_HOME=/workspace/mnt/group/video/chenshuaijun/pytorch
cd reid/deep-person-reid
/workspace/mnt/group/video/chenshuaijun/pkg/anaconda3/envs/python36/bin/python -u train_soft_htri_warmup.py -d market1501 -a resnet50 --max-epoch 200 --train-batch 128 --test-batch 128 --stepsize 60 --eval-step 10 --save-dir log/resnet50-xent-htri-market1501-nowp-h384 --gpu-devices 0,1 -j 8 --root ../datasets/ --height 384 --width 192 --lr 0.003
