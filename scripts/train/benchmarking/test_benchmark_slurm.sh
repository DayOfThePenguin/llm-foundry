#!/bin/bash

PROJECT="tput"
GPU_MAX_MEMORY_GB="80"

# seqlen 2048
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  32 --accum  2 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB