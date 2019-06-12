#!/bin/sh
export TORCH_HOME=/workspace/mnt/group/video/chenshuaijun/pytorch
cd reid/deep-person-reid
/workspace/mnt/group/video/chenshuaijun/pkg/anaconda3/envs/python36/bin/python -u train_img_model_xent.py -d market1501 --root ../datasets/ -a resnet50 --max-epoch 100 --train-batch 64 --test-batch 64 --stepsize 20 --eval-step 20 --save-dir log/resnet50-xent-warm-rmpl-market1501 --gpu-devices 0,1 -j 8 --warmup --lr 0.003
