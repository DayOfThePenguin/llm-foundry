#!/bin/bash

PROJECT="tput"
GPU_MAX_MEMORY_GB="80"

# A100 80GB

# seqlen 2048
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  32 --accum  2 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  32 --accum  2 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  24 --accum  2 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size  14 --accum  4 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size  10 --accum  6 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  32 --accum  2 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  20 --accum  3 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   3 --accum 21 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB

# INCREASE GPU COUNT
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 16 32 64 --microbatch_size 32 --accum  1 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 16 32 64 --microbatch_size 32 --accum  1 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 16 32 64 --microbatch_size 24 --accum  1 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 16 32 64 --microbatch_size 20 --accum  1 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 16       --microbatch_size 10 --accum  3 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 16 32 64 --microbatch_size 32 --accum  1 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 16       --microbatch_size 24 --accum  1 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 16       --microbatch_size 10 --accum  3 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g    32 64 --microbatch_size 12 --accum  3 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g    32 64 --microbatch_size 32 --accum  1 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g    32    --microbatch_size 14 --accum  3 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  70b.yaml -g    32    --microbatch_size  2 --accum 16 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g       64 --microbatch_size 16 --accum  3 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  70b.yaml -g       64 --microbatch_size  8 --accum  4 -s 11 11 --gpu_max_memory $GPU_MAX_MEMORY_GB

# SCALE SEQUENCE LENGTH
# seqlen 512
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size 128 --accum  2 -s  9  9 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size 128 --accum  2 -s  9  9 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  96 --accum  2 -s  9  9 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size  56 --accum  4 -s  9  9 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size  40 --accum  6 -s  9  9 --gpu_max_memory $GPU_MAX_MEMORY_GB --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size 128 --accum  2 -s  9  9 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  80 --accum  3 -s  9  9 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size  12 --accum 21 -s  9  9 --gpu_max_memory $GPU_MAX_MEMORY_GB
# seqlen 1024
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  64 --accum  2 -s 10 10 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  64 --accum  2 -s 10 10 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  48 --accum  2 -s 10 10 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size  18 --accum  4 -s 10 10 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size  20 --accum  6 -s 10 10 --gpu_max_memory $GPU_MAX_MEMORY_GB --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  64 --accum  2 -s 10 10 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  40 --accum  3 -s 10 10 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   6 --accum 21 -s 10 10 --gpu_max_memory $GPU_MAX_MEMORY_GB
# seqlen 4096
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size  16 --accum  2 -s 12 12 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size  16 --accum  2 -s 12 12 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size  12 --accum  2 -s 12 12 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   7 --accum  4 -s 12 12 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   5 --accum  6 -s 12 12 --gpu_max_memory $GPU_MAX_MEMORY_GB --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size  16 --accum  2 -s 12 12 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size  10 --accum  3 -s 12 12 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   1 --accum 21 -s 12 12 --gpu_max_memory $GPU_MAX_MEMORY_GB
# seqlen 8192
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   8 --accum  2 -s 13 13 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   8 --accum  2 -s 13 13 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   6 --accum  2 -s 13 13 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   3 --accum  4 -s 13 13 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   3 --accum  6 -s 13 13 --gpu_max_memory $GPU_MAX_MEMORY_GB --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   8 --accum  2 -s 13 13 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size   5 --accum  3 -s 13 13 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  30b.yaml -g 8 --microbatch_size   1 --accum 21 -s 13 13 --gpu_max_memory $GPU_MAX_MEMORY_GB
# seqlen 16384
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   4 --accum  2 -s 14 14 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   4 --accum  2 -s 14 14 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   3 --accum  2 -s 14 14 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   2 --accum  4 -s 14 14 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   1 --accum  6 -s 14 14 --gpu_max_memory $GPU_MAX_MEMORY_GB --fsdp_config_activation_checkpointing false
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   4 --accum  2 -s 14 14 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size   3 --accum  3 -s 14 14 --gpu_max_memory $GPU_MAX_MEMORY_GB
# seqlen 32768
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   2 --accum  2 -s 15 15 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   2 --accum  2 -s 15 15 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   1 --accum  2 -s 15 15 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   1 --accum  4 -s 15 15 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   3 --accum  6 -s 15 15 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   2 --accum  2 -s 15 15 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m  13b.yaml -g 8 --microbatch_size   1 --accum  3 -s 15 15 --gpu_max_memory $GPU_MAX_MEMORY_GB
# seqlen 65536
python submit_benchmarks.py --project $PROJECT -m 125m.yaml -g 8 --microbatch_size   1 --accum  2 -s 16 16 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 350m.yaml -g 8 --microbatch_size   1 --accum  2 -s 16 16 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m 760m.yaml -g 8 --microbatch_size   1 --accum  2 -s 16 16 --gpu_max_memory $GPU_MAX_MEMORY_GB --fsdp_config_activation_checkpointing true
python submit_benchmarks.py --project $PROJECT -m   1b.yaml -g 8 --microbatch_size   1 --accum  2 -s 16 16 --gpu_max_memory $GPU_MAX_MEMORY_GB --fsdp_config_activation_checkpointing true
python submit_benchmarks.py --project $PROJECT -m   3b.yaml -g 8 --microbatch_size   1 --accum  2 -s 16 16 --gpu_max_memory $GPU_MAX_MEMORY_GB
python submit_benchmarks.py --project $PROJECT -m   7b.yaml -g 8 --microbatch_size   1 --accum  2 -s 16 16 --gpu_max_memory $GPU_MAX_MEMORY_GB
