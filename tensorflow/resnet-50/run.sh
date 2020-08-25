#!/bin/bash

export DISTRIBUTED_FRAMEWORK=byteps

# by default the network is ResNet50, but you can also test VGG16
bpslaunch python3 benchmark.py $@