pkill python

	    #-x HOROVOD_CYCLE_TIME=0.1 \
	    #--verbose \
	    #-x HOROVOD_TIMELINE=timeline.json \
	    #-x MXNET_HOROVOD_NUM_GROUPS=4 \
	    #--synthetic_data --eval_use_npz \

mpirun -np $NP --allow-run-as-root -mca pml ob1 -mca btl ^openib \
            -mca btl_tcp_if_exclude docker0,lo --bind-to none \
            --mca plm_rsh_agent 'ssh -q -o StrictHostKeyChecking=no' \
	    -x NCCL_MIN_NRINGS=$NCCLMINNRINGS -x NCCL_DEBUG=INFO \
	    -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
	    -x HOROVOD_CYCLE_TIME=1 \
	    -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=120 \
	    -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=120 \
	    -x MXNET_SAFE_ACCUMULATION=1 \
	    --tag-output ompi_bind_DGX1.sh python run_pretraining.py \
	    --data='/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train' \
	    --data_eval='/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test' \
	    --optimizer $OPTIMIZER \
	    --warmup_ratio $WARMUP_RATIO \
	    --num_steps $NUMSTEPS \
	    --ckpt_interval $CKPTINTERVAL \
	    --dtype $DTYPE \
	    --ckpt_dir $CKPTDIR \
	    --lr $LR \
	    --total_batch_size $BS \
	    --total_batch_size_eval $BS \
	    --accumulate $ACC \
	    --model $MODEL \
	    --max_seq_length $MAX_SEQ_LENGTH \
	    --max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
	    --num_data_workers 4 \
	    --no_compute_acc --raw \
	    --comm_backend horovod --log_interval $LOGINTERVAL
