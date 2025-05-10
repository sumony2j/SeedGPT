#!/bin/bash

export CUDA_VISIBLE_DEVICE="0,1,2,3"
export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME = "ens2"

python3 Inference.py                                           \
    --model_path "/root/llm/SeedGPT.pt"                   \
    --tokenizer_path "/root/llm/tokenizer.json"           \
    --input "Where is"                                           \
    --max_token 10000                                          \
    --output_file "/root/llm/llm_output.txt"              \
    --show
