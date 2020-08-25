#!/bin/bash

# if you use horovod, you should replace bpslaunch with mpi-based command
bpslaunch python3 ./official/r1/transformer/transformer_main.py --ds=byteps $@
