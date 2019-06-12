#!/bin/sh
export TORCH_HOME=/media/SSD/chensj/pytorch
/media/data/chensj/anaconda3/envs/python36/bin/python -u trainonly_soft_htri_warmup.py -d market1501 -a resnet50 --max-epoch 400 --train-batch 64 --test-batch 128 --stepsize 20 --eval-step 10 --save-dir log/resnet50-xent-htri-ibn-randomer-market1501-h384 --gpu-devices 1 -j 8 --root ../datasets/ --height 384 --width 192 --warmup --margin 0.3 --loss xent,htri --random-erasing
