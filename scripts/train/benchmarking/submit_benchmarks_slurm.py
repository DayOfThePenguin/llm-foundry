# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# A mash-up of https://github.com/mosaicml/composer/blob/dev/examples/training_with_submitit.ipynb
# and the LLM-Foundry benchmarking code that uses SLURM instead of the Mosaic MCLI (and Mosaic's cloud)
# to execute distributed training

import argparse
import logging
import math
import os

from typing import Union

import submitit

from omegaconf import OmegaConf as om
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

# local imports
from ..train import main

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s", level=logging.DEBUG
)


def str_to_bool(value: Union[bool, str]) -> bool:
    """Convert a string to a boolean. If already boolean, return it"""
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate and run configurations to test MPT training throughput on a Slurm cluster."
    )

    parser.add_argument("--project", type=str, default="tput")
    parser.add_argument("--fsdp_config_mixed_precision", type=str, default="PURE")
    parser.add_argument(
        "--fsdp_config_activation_checkpointing",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=None,
    )
    parser.add_argument(
        "-s",
        "--seq_len_exp",
        type=int,
        default=[11, 11],
        nargs=2,
        help="exponent of seq lengths to be tested (default: [11, 11] = 2048)",
    )
    parser.add_argument(
        "-b",
        "--batch_size_exp",
        type=int,
        default=None,
        nargs=2,
        help="exponent of batch size (in tokens) to be tested (default: [19, 23] = 2^19 to 2^23)",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[],
        help="batch sizes to run.",
    )
    parser.add_argument(
        "--accum",
        type=int,
        default=None,
        help="batch sizes multiplier (accumulations before step).",
    )
    parser.add_argument(
        "-m",
        "--model_yamls",
        type=str,
        default=[
            "125m.yaml",
            "350m.yaml",
            "760m.yaml",
            "1b.yaml",
            "3b.yaml",
            "7b.yaml",
            "13b.yaml",
            "30b.yaml",
            "70b.yaml",
        ],
        choices=[
            "125m.yaml",
            "350m.yaml",
            "760m.yaml",
            "1b.yaml",
            "3b.yaml",
            "7b.yaml",
            "13b.yaml",
            "30b.yaml",
            "70b.yaml",
        ],
        nargs="+",
        help="model sizes to test",
    )

    parser.add_argument("--attn_impl", type=str, default="triton")
    parser.add_argument(
        "-g",
        "--gpu_nums",
        type=int,
        default=[16],
        nargs="+",
    )

    parser.add_argument(
        "--microbatch_size", type=int, default=None, help="set microbatch_size"
    )

    parser.add_argument("--pad_vocab_multiple", type=int, default=None)

    parser.add_argument(
        "--data_remote",
        type=str,
        default=None,
        help="optional data remote path for streaming data",
    )

    # TODO: swap this for Tensorboard
    # parser.add_argument('--wandb',
    #                     type=str_to_bool,
    #                     nargs='?',
    #                     const=True,
    #                     default=True)

    parser.add_argument("--priority", type=str, default="low")

    parser.add_argument(
        "--gpu_max_memory",
        type=int,
        default=80,
        help="maximum memory, in GB, per GPU in the cluster",
    )

    return parser.parse_args()


def get_max_seq_lens(pows: list[int] = [9, 14]) -> list[int]:
    """Converts a list of exponents into a list of powers of 2

    This list is used to determine the maximum sequence lengths to iterate over

    Parameters:
    pows (list[int]): exponents to convert into powers of 2

    Returns:
    (list[int]): a list of the maximum sequence lengths
    """
    return [2**n for n in range(pows[0], pows[1] + 1)]


def get_global_train_batch_sizes(
    max_seq_len: int, pows: list[int], batch_sizes: list[int] = []
) -> list[int]:
    """Converts a list of global train batch size exponents into a list of powers of 2"""
    if pows:
        # global batch size in tokens (defualt: .5M thru 8M)
        global_train_token_counts = [2**n for n in range(pows[0], pows[1] + 1)]
        batch_sizes += [
            t // max_seq_len for t in global_train_token_counts
        ]  # global batch size in samples
    return batch_sizes


def create_job_name(cfg: Union[DictConfig, ListConfig]) -> str:
    """Figure out what the model name is and make a unique run name based on some key hyperparameters

    Parameters:
    cfg (DictConfig): model configuration
    """
    model_name = "-".join(model_yaml.split(".")[-2].split("/")[-2:]).replace("_", "-")
    model_name = model_name.split("-")
    if "mosaic" in model_name:
        model_name.pop(model_name.index("mosaic"))
    model_name = "".join(model_name)
    name = f"{cfg.project}-{model_name}-{gpu_num}x{cfg.gpu_max_memory}GB-s{max_seq_len}b{global_train_batch_size}"
    name = name.replace("_", "-")

    # unclear why this is here, but not going to mess with it
    name_len_lim = 54 - 7
    if len(name) > name_len_lim:
        _name = name
        name = name[:name_len_lim]
        print(f"Shortening {_name} to {name} ({name_len_lim} chars)")

    return name


def submit_job(cfg: Union[ListConfig, DictConfig]):
    """Adapted from the Mosaic submitit demo:
    https://github.com/mosaicml/composer/blob/dev/examples/training_with_submitit.ipynb

    Submit a job to a slurm cluster using submitit

    Parameters:
    name (str): name of the run. used to create a unique slurm log directory
    """
    slurm_ngpus = cfg.gpu_num

    # assuming 8x GPU per node, we need this many nodes
    slurm_nnodes = math.ceil(slurm_ngpus / 8)
    slurm_timeout = 1024
    # TODO: need to think about how to more smartly calculate this based on dataloader worker
    # CPUs + expected CPU overhead for GPU control
    workers = 10

    slurm_directory = f"logs-{cfg.name}"
    executor = submitit.AutoExecutor(folder=slurm_directory)

    executor.update_parameters(
        mem_gb=128 * slurm_ngpus,
        gpus_per_node=slurm_ngpus,
        tasks_per_node=slurm_ngpus,
        cpus_per_task=workers,
        nodes=slurm_nnodes,
        timeout_min=slurm_timeout,
        slurm_partition="gpu",
        # see submitit github repo for details
    )

    # TODO: my interpretation is that this is how you pass parameters to the python function you're submitting. We
    # can also hard-code some of these if it makes more sense to do that
    executor.submit(main, cfg)


def run_check_capacity(
    model_yaml: str, gpu_num: int, per_gpu_mem: int, p_multiplier: int = 16
):
    """Determine whether it's reasonable to run the given number of parameters on a given system configuration

    Parameters:
        model_yaml: name of model yaml file to read. Not a path, just a file name like '30b.yaml'
        gpu_num: how many GPUs across all nodes will be used
        per_gpu_mem: how much GPU memory (in GB) per device
        p_multiplier: fudge factor for roughly how much memory is required per billion parameters. this depends on a
            lot of things and doesn't take into account things like CPU offload or activation checkpointing.
    """
    _params = model_yaml.replace(".yaml", "")
    params, mult = int(_params[:-1]), _params[-1]
    if mult == "m":
        b_params = params / 1000
    elif mult == "b":
        b_params = params
    else:
        raise ValueError

    if p_multiplier * b_params > gpu_num * per_gpu_mem:
        print(
            f"WARNING: will not be running {model_yaml=} on {gpu_num=} {per_gpu_mem=}GB GPUs since it probably will not fit into memory"
        )
        return False
    return True


def run_check_batch_size(num_gpus: int, micro_batch_size: int, global_batch_size: int):
    """Check that the global batch size choice is reasonable.

    This function is opaquely called run_check_dtms in the parent codebase...unclear what
    "dtms" means and what the significance of this calculation is.

    Renamed dtms parameter to micro_batch_size because that's what's actually being passed
    to the function.
    """
    if num_gpus * micro_batch_size > global_batch_size:
        print(
            f"WARNING: Cannot run with {global_batch_size=} on {num_gpus=} with {micro_batch_size=} ({num_gpus*micro_batch_size=})."
        )
        return False
    return True


if __name__ == "__main__":
    args = parse_args()

    # jobs counter to keep track of how many permutations we're submitting to SLURM
    n_jobs = 0

    # create a separate job for each max sequence length we want to explore
    max_seq_lens = get_max_seq_lens(args.seq_len_exp)
    logging.debug(f"Creating jobs for maximum sequence lengths {max_seq_lens}")
    for max_seq_len in max_seq_lens:
        logging.debug(f"{max_seq_len=}")

        # create a separate job for each GPU count in the args.gpu_nums list
        logging.debug(f"Creating jobs for gpu counts {args.gpu_nums}")
        for gpu_num in args.gpu_nums:
            logging.debug(f"{gpu_num=}")
            global_train_batch_sizes = get_global_train_batch_sizes(
                max_seq_len, args.batch_size_exp, args.batch_sizes
            )
            if not global_train_batch_sizes and args.microbatch_size is not None:
                accum = args.accum or 1
                global_train_batch_sizes = [accum * gpu_num * args.microbatch_size]

            # create a separate job for each global train batch size
            logging.debug(
                f"Creating jobs for global train batch sizes {global_train_batch_sizes}"
            )
            for global_train_batch_size in global_train_batch_sizes:
                logging.debug(f"{global_train_batch_size=}")

                # create a separate job for each model yaml
                logging.debug(f"Creating jobs for model yamls {args.model_yamls}")
                for model_yaml in args.model_yamls:
                    logging.debug(f"{model_yaml=}")

                    # make sure there's roughly enough GPU memory across the cluster to run the
                    run = run_check_capacity(
                        model_yaml,
                        gpu_num,
                        per_gpu_mem=args.gpu_max_memory,
                        p_multiplier=4,
                    )
                    if args.microbatch_size is not None:
                        # make sure the microbatch size makes sense given the global batch size
                        run = run and run_check_batch_size(
                            gpu_num, args.microbatch_size, global_train_batch_size
                        )

                    if run:
                        # load the relevant model configuration yaml
                        yaml_path = os.path.join("../yamls/mpt", model_yaml)
                        with open(yaml_path) as f:
                            cfg = om.load(f)

                        # some of the default values set in the yaml will be overwritten
                        cfg.max_seq_len = max_seq_len
                        cfg.global_train_batch_size = global_train_batch_size
                        cfg.device_train_microbatch_size = args.microbatch_size
                        cfg.max_duration = "30ba"
                        cfg.fsdp_config.activation_checkpointing = (
                            args.fsdp_config_activation_checkpointing
                        )

                        # some of the command line args need to be added
                        cfg.model_yaml_name = model_yaml
                        cfg.project = args.project
                        cfg.gpu_max_memory = args.gpu_max_memory

                        # some other run-specific values need to be added
                        cfg.gpu_num = gpu_num
                        cfg.name = create_job_name(cfg)

                        print(cfg)

                        # submit the given config to slurm
                        submit_job(cfg)
                        n_jobs += 1

    print(f"{n_jobs=}")
