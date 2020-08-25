
export BYTEPS_PATH=${BYTEPS_PATH:-/usr/local/byteps}
if [ "$DMLC_ROLE" != "worker" ]; then
    python3 $BYTEPS_PATH/launcher/launch.py 
fi

cd /root && git clone -b bert-byteps https://github.com/ymjiang/gluon-nlp.git 
cd gluon-nlp && python3 setup.py install

apt-get update && apt-get install -y zip
mkdir -p /root/.mxnet/models
cd /root/.mxnet/models 
wget https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/vocab/book_corpus_wiki_en_uncased-a6607397.zip
unzip *.zip
cd /root
       
export LR=0.00354;   
export OPTIONS=--synthetic_data\ --eval_use_npz; 
export WARMUP_RATIO=0.1;          
export NUMSTEPS=281250;   
export CKPTDIR=ckpt_stage1_lamb_16k-682a361-c5fd6fc-0412-cu90; 
export ACC=1;         
export GPUS=$NVIDIA_VISIBLE_DEVICES
export BPS_HOME=$BYTEPS_PATH

# start
export TRUNCATE_NORM="${TRUNCATE_NORM:-1}"
export LAMB_BULK="${LAMB_BULK:-30}"
export EPS_AFTER_SQRT="${EPS_AFTER_SQRT:-1}"
export NUMSTEPS="${NUMSTEPS:-900000}"
export DTYPE="${DTYPE:-float16}"
export ACC="${ACC:-1}"
export MODEL="${MODEL:-bert_24_1024_16}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
export MAX_PREDICTIONS_PER_SEQ="${MAX_PREDICTIONS_PER_SEQ:-20}"
export LR="${LR:-0.000625}"
export LOGINTERVAL="${LOGINTERVAL:-10}"
export CKPTDIR="${CKPTDIR:-ckpt_stage1_lamb}"
export CKPTINTERVAL="${CKPTINTERVAL:-300000000}"
export OPTIMIZER="${OPTIMIZER:-bertadam}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.003125}"
export BYTEPS_PARTITION_BYTES="${BYTEPS_PARTITION_BYTES:-4096000}"
export BYTEPS_NCCL_GROUP_SIZE="${BYTEPS_NCCL_GROUP_SIZE:-16}"
export BPS_HOME="${BPS_HOME:-/usr/local/byteps}"
export NVIDIA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
export DMLC_WORKER_ID="${DMLC_WORKER_ID:-0}"
export DMLC_NUM_WORKER="${DMLC_NUM_WORKER:-1}"
export DMLC_ROLE=worker
export NCCL_MIN_NRINGS="${NCCL_MIN_NRINGS:-16}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD:-120}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD:-120}"
export MXNET_SAFE_ACCUMULATION="${MXNET_SAFE_ACCUMULATION:-1}"
export OPTIONS="${OPTIONS:- }"
export DATA="${DATA:-/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train}"
export DATAEVAL="${DATAEVAL:-/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-16384}"

echo $NVIDIA_VISIBLE_DEVICES

env | grep BYTEPS

python3 $BPS_HOME/launcher/launch.py \
	python3 /root/gluon-nlp/scripts/bert/run_pretraining.py \
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
	    --comm_backend byteps --log_interval $LOGINTERVAL \
		--total_batch_size $TOTAL_BATCH_SIZE \
		--total_batch_size_eval $TOTAL_BATCH_SIZE \
		$OPTIONS 

