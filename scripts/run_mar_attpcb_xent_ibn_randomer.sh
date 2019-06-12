#!/bin/sh
export TORCH_HOME=/media/SSD/chensj/pytorch
/media/data/chensj/anaconda3/envs/python36/bin/python -u train_attpcb_soft_htri_warmup.py -d market1501 \
 -a attpcb --max-epoch 400 --train-batch 64 --test-batch 128 --stepsize 20 --eval-step 10 --save-dir log/attpcb-xent-ibn-randomer-market-h384 --gpu-devices 0,1 -j 8 --root ../datasets/ --height 384 --width 192 --warmup --margin 0.3 --loss xent --random-erasing
