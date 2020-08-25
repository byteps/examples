#!/bin/bash

export DISTRIBUTED_FRAMEWORK=byteps

# you can also test other networks (e.g., ResNet) with different number of layers
bpslaunch python3 train_imagenet.py --network vgg --num-layers 16 --batch-size 32 $@