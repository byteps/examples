#!/bin/bash

# dataset can be downloaded from 
# https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view

export DISTRIBUTED_FRAMEWORK=torch_native

cd ./ugatit
torchrun --nproc_per_node=${ML_PLATFORM_WORKER_GPU} \
         --nnodes=${ML_PLATFORM_WORKER_NUM} \
         --node_rank=${RANK} \
         --master_addr="${MASTER_ADDR}" \
         --master_port=${MASTER_PORT} main.py $@

# https://github.com/pytorch/pytorch/blob/v1.10.0/torch/distributed/run.py
