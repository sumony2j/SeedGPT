#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/corex-4.2.0/lib64:/usr/local/openmpi/lib:/usr/local/lib
export CUDA_VISIBLE_DEVICE="0,1,2,3"
export NCCL_DEBUG=INFO

python3 SeedGPT.py                       \
    --batch_size 64                     \
    --iteration 10000                    \
    --dataset "/root/llm/Data.txt"  \
    --context 256                        \
    --emb_size 384                       \
    --n_layers 6                         \
    --lr 3e-4                            \
    --n_head 6                           \
    --eval_itr 100
