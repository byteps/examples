#!/bin/bash

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd
wget https://byteps.tos-cn-qingdao.volces.com/byteps-examples.tar.gz
mkdir byteps-examples
tar xvzf byteps-examples.tar.gz --strip-components=1 -C byteps-examples

# dataset can be downloaded from
# https://drive.google.com/file/d/1xOWj1UVgp6NKMT3HbPhBbtq2A4EDkghF/view

SHOULD_DOWNLOAD_DATASET=${SHOULD_DOWNLOAD_DATASET:-1}
if [[ "$SHOULD_DOWNLOAD_DATASET" == "1" ]]; then
  cd
  wget https://byteps.tos-cn-qingdao.volces.com/datasets/selfie2anime.zip
  unzip ./selfie2anime.zip -d selfie2anime
  cd -
fi

export DISTRIBUTED_FRAMEWORK=${DISTRIBUTED_FRAMEWORK:-byteps}

if [[ "$DISTRIBUTED_FRAMEWORK" != "horovod" ]]; then
  echo "This script can only be used to launch training uisng horovod."
  exit 1
fi
python3 ${this_dir}/ugatit/main.py $@
