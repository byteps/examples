#!/bin/bash

# install gluon-nlp under this dir
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $THIS_DIR/gluon-nlp
python3 setup.py install

# prepare dict 
mkdir -p ~/.mxnet/models
cd ~/.mxnet/models 
wget https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/vocab/book_corpus_wiki_en_uncased-a6607397.zip
unzip *.zip
       
# below params are used in V100-32GB, with synthetic data
export OPTIONS=--synthetic_data\ --eval_use_npz; 
export TRUNCATE_NORM="${TRUNCATE_NORM:-1}"
export LAMB_BULK="${LAMB_BULK:-30}"
export EPS_AFTER_SQRT="${EPS_AFTER_SQRT:-1}"
export NUMSTEPS="${NUMSTEPS:-281250}"
export DTYPE="${DTYPE:-float16}"
export ACC="${ACC:-1}"
export MODEL="${MODEL:-bert_24_1024_16}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
export MAX_PREDICTIONS_PER_SEQ="${MAX_PREDICTIONS_PER_SEQ:-20}"
export LR="${LR:-0.00354}"
export LOGINTERVAL="${LOGINTERVAL:-10}"
export CKPTDIR="${CKPTDIR:-ckpt_stage1_lamb_16k-682a361-c5fd6fc-0412-cu90}"
export CKPTINTERVAL="${CKPTINTERVAL:-300000000}"
export OPTIMIZER="${OPTIMIZER:-bertadam}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD:-120}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD:-120}"
export MXNET_SAFE_ACCUMULATION="${MXNET_SAFE_ACCUMULATION:-1}"
export DATA="${DATA:-/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train}"
export DATAEVAL="${DATAEVAL:-/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test}"
export BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-64}
export NUM_GPU=${NUM_GPU:-1}
export TOTAL_BATCH_SIZE=$(($NUM_GPU*$BATCH_SIZE_PER_GPU))

export COMM_BACKEND=${COMM_BACKEND:-byteps}

bpslaunch python3 $THIS_DIR/gluon-nlp/scripts/bert/run_pretraining.py \
    --data=$DATA \
    --data_eval=$DATAEVAL \
    --optimizer $OPTIMIZER \
    --warmup_ratio $WARMUP_RATIO \
    --num_steps $NUMSTEPS \
    --ckpt_interval $CKPTINTERVAL \
    --dtype $DTYPE \
    --ckpt_dir $CKPTDIR \
    --lr $LR \
    --accumulate $ACC \
    --model $MODEL \
    --max_seq_length $MAX_SEQ_LENGTH \
    --max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
    --num_data_workers 4 \
    --no_compute_acc \
    --comm_backend $COMM_BACKEND \
    --log_interval $LOGINTERVAL \
    --total_batch_size $TOTAL_BATCH_SIZE \
    --total_batch_size_eval $TOTAL_BATCH_SIZE \
    $OPTIONS 

	   
