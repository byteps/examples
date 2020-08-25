#!/bin/bash

export DISTRIBUTED_FRAMEWORK=byteps

bpslaunch python3 train_imagenet.py $@