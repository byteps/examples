#!/bin/bash

# dataset can be downloaded from 
# https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view

export DISTRIBUTED_FRAMEWORK=byteps

cd ./ugatit
bpslaunch python3 main.py $@