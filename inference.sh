#!/bin/bash

export CUDA_VISIBLE_DEVICE="0,1,2,3"
export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME = "ens2"

python3 Inference.py                                           \
    --model_path "./SeedGPT.pt"                   \
    --tokenizer_path "./tokenizer.json"           \
    --input "Love "                                           \
    --max_token 10000                                          \
    --output_file "./llm_output.txt"              \
    --show
