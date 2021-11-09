#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# dataset can be downloaded from
# https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view

SHOULD_DOWNLOAD_DATASET=${SHOULD_DOWNLOAD_DATASET:-1}
OMPI_COMM_WORLD_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
if [[ "$SHOULD_DOWNLOAD_DATASET" == "1" ]] && [[ "$OMPI_COMM_WORLD_LOCAL_RANK" == "0" ]]; then
  cd
  [[ -f ./selfie2anime.zip ]] || wget -nv https://byteps.tos-cn-qingdao.volces.com/datasets/selfie2anime.zip
  unzip -qn ./selfie2anime.zip -d selfie2anime
  cd -
fi

export DISTRIBUTED_FRAMEWORK=${DISTRIBUTED_FRAMEWORK:-byteps}

if [[ "$DISTRIBUTED_FRAMEWORK" == "byteps" ]]; then
  bytepsrun python3 ${this_dir}/ugatit/main.py $@
elif [[ "$DISTRIBUTED_FRAMEWORK" == "horovod" ]]; then
  python3 ${this_dir}/ugatit/main.py $@
elif [[ "$DISTRIBUTED_FRAMEWORK" == "torch_native" ]]; then
  torchrun --nproc_per_node=${ML_PLATFORM_WORKER_GPU} \
           --nnodes=${ML_PLATFORM_WORKER_NUM} \
           --node_rank=${RANK} \
           --master_addr="${MASTER_ADDR}" \
           --master_port=${MASTER_PORT} ${this_dir}/ugatit/main.py $@
else
  echo "Unsupported distributed training framework: $DISTRIBUTED_FRAMEWORK"
  echo "Please choose from: byteps, horovod and torch_native"
fi
