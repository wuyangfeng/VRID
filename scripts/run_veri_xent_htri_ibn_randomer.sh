#!/bin/sh
python -u train_soft_htri_warmup.py -d veri \
 -a resnet50 --max-epoch 400 --train-batch 64 --test-batch 128 --stepsize 20 --eval-step 10 \
 --save-dir log/resnet50-xent-htri-ibn-randomer-veri-h384 --gpu-devices 0 -j 8 \
 --root ../datasets/ --height 224 --width 224 --warmup --margin 0.3 --loss xent,htri --random-erasing
