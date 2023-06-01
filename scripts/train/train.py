# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import logging
import os
import subprocess
import sys
import warnings

from typing import Union

import submitit
import torch

from composer import Trainer
from composer.core import Evaluator
from composer.utils import dist, get_device, reproducibility
from omegaconf import OmegaConf as om
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from llmfoundry import (
    COMPOSER_MODEL_REGISTRY,
    build_finetuning_dataloader,
    build_text_denoising_dataloader,
)
from llmfoundry.data.text_data import build_text_dataloader
from llmfoundry.models.utils import init_empty_weights
from llmfoundry.utils.builders import (
    build_algorithm,
    build_callback,
    build_icl_evaluators,
    build_logger,
    build_optimizer,
    build_scheduler,
    build_tokenizer,
)
from llmfoundry.utils.config_utils import log_config, update_batch_size_info

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO
)


def validate_config(cfg: Union[ListConfig, DictConfig]):
    """Validates compatible model and dataloader selection."""
    loaders = [cfg.train_loader]
    if "eval_loader" in cfg:
        loaders.append(cfg.eval_loader)
    for loader in loaders:
        if loader.name == "text":
            if cfg.model.name in ["hf_prefix_lm", "hf_t5"]:
                raise ValueError(
                    f'Model type "{cfg.model.name}" is not supported when using the "text " '
                    + f'dataloader. Please use the "text_denoising" dataloader to pre-train that model type.'
                )
        elif loader.name == "text_denoising":
            if cfg.model.name == "hf_causal_lm":
                raise ValueError(
                    f'Model type "{cfg.model.name}" is not supported when using the "text_denoising" '
                    + f'dataloader. Please use the "text" dataloader to pre-train that model type.'
                )
            if (
                loader.mixture_of_denoisers.decoder_only_format
                and cfg.model.name == "hf_t5"
            ):
                warnings.warn(
                    'Model type "hf_t5" requires `decoder_only_format` to be ``False``. '
                    + "Overriding `decoder_only_format` from ``True`` to ``False``."
                )
                loader.mixture_of_denoisers.decoder_only_format = False
            if (
                not loader.mixture_of_denoisers.decoder_only_format
            ) and cfg.model.name == "hf_prefix_lm":
                warnings.warn(
                    'Model type "hf_prefix_lm" requires `decoder_only_format` to be ``True``. '
                    + "Overriding `decoder_only_format` from ``False`` to ``True``."
                )
                loader.mixture_of_denoisers.decoder_only_format = True

    if "icl_tasks" in cfg:
        if cfg.model.name == "hf_t5":
            raise ValueError(
                'ICL evaluation does not currently support Encoder-Decoder models, such as "hf_t5".'
            )


def build_composer_model(model_cfg: DictConfig, tokenizer: Tokenizer):
    warnings.filterwarnings(
        action="ignore",
        message="Torchmetrics v0.9 introduced a new argument class property",
    )
    if model_cfg.name not in COMPOSER_MODEL_REGISTRY:
        raise ValueError(f"Not sure how to build model with name={model_cfg.name}")
    return COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer)


def build_dataloader(
    cfg: Union[ListConfig, DictConfig], tokenizer: Tokenizer, device_batch_size: int
):
    if cfg.name == "text":
        return build_text_dataloader(
            cfg,  # type: ignore
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == "text_denoising":
        return build_text_denoising_dataloader(
            cfg,  # type: ignore
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == "finetuning":
        return build_finetuning_dataloader(
            cfg,  # type: ignore
            tokenizer,
            device_batch_size,
        )

    else:
        raise ValueError(f"Not sure how to build dataloader with config: {cfg}")


def set_up_dist_env(cfg: Union[ListConfig, DictConfig]):
    # 1. RANK
    job_env = submitit.JobEnvironment()
    global_rank = job_env.global_rank
    logging.info(job_env)

    # 2. LOCAL_RANK
    local_rank = job_env.local_rank
    logging.info(local_rank)

    # 3. LOCAL_WORLD_SIZE
    ngpus_per_node = cfg.gpus_per_node

    # 4. WORLD_SIZE
    if os.getenv("SLURM_NNODES") is None:
        raise NameError("environment variable SLURM_NNODES is not set")
    else:
        world_size = int(os.getenv("SLURM_NNODES")) * ngpus_per_node  # type: ignore

    # 5. NODE_RANK
    if os.getenv("SLURM_NODEID") is None:
        raise NameError("environment variable SLURM_NODEID is not set")
    else:
        node_rank = int(os.getenv("SLURM_NODEID"))  # type: ignore

    # 6. MASTER_ADDR
    if os.getenv("SLURM_JOB_NODELIST") is None:
        raise NameError("environment variable SLURM_JOB_NODELIST is not set")
    cmd = "scontrol show hostnames " + os.getenv("SLURM_JOB_NODELIST")  # type: ignore
    stdout = subprocess.check_output(cmd.split())
    host_name = stdout.decode().splitlines()[0]

    # 7. MASTER_PORT
    port = 54321

    # Set All the Necessary Environment Variables!
    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(ngpus_per_node)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["NODE_RANK"] = str(node_rank)
    os.environ["MASTER_ADDR"] = host_name
    os.environ["MASTER_PORT"] = str(port)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_ASYNC_ERROR_HANDLIG"] = "1"

def main(cfg: DictConfig):
    # TODO: is this necessary now that we've fixed the shared memory collisions?
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Check for incompatibilities between the model and data loaders
    validate_config(cfg)

    set_up_dist_env(cfg)

    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message=f"torch.distributed.*_base is a private function and will be deprecated.*",
    )

    cfg.dist_timeout = cfg.get("dist_timeout", 600.0)

    reproducibility.seed_all(cfg.seed)
    dist.initialize_dist(get_device(None), timeout=cfg.dist_timeout)

    # Run Name
    if cfg.get("run_name") is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "llm")

    # Get batch size info
    cfg = update_batch_size_info(cfg)

    # Read FSDP Config as a dict
    fsdp_config = cfg.get("fsdp_config", None)
    fsdp_config = om.to_container(fsdp_config, resolve=True) if fsdp_config else None

    # Restrict model init_device to 'meta' and 'cpu',
    # using 'cuda' vs. 'cuda:id' is tricky and can lead to common user errors
    # when multiple GPUs are available.
    # Also 'meta' is only valid when using FSDP
    init_device = cfg.model.get("init_device", "cpu")
    assert init_device in ["meta", "cpu"]
    if fsdp_config is None and init_device == "meta":
        warnings.warn(
            "Using `cfg.model.init_device='meta'` is only valid when using FSDP! "
            + "Reverting to `cfg.model.init_device='cpu'`."
        )
        cfg.model.init_device = "cpu"

    # build tokenizer
    tokenizer = build_tokenizer(cfg.tokenizer)

    # Build Model
    print("Initializing model...")
    init_context = contextlib.nullcontext()
    if init_device == "meta":
        init_context = init_empty_weights()
    with init_context:
        model = build_composer_model(cfg.model, tokenizer)
    cfg.n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.n_params=:.2e}")

    # Dataloaders
    print("Building train loader...")
    train_loader = build_dataloader(
        cfg.train_loader,
        tokenizer,
        cfg.device_train_batch_size,
    )
    print("Building eval loader...")
    evaluators = []
    if "eval_loader" in cfg:
        eval_loader = Evaluator(
            label="eval",
            dataloader=build_dataloader(
                cfg.eval_loader, tokenizer, cfg.device_eval_batch_size
            ),
            metric_names=list(model.train_metrics.keys()),
        )
        evaluators.append(eval_loader)

    if "icl_tasks" in cfg:
        icl_evaluators, _ = build_icl_evaluators(
            cfg.icl_tasks, tokenizer, cfg.max_seq_len, cfg.device_eval_batch_size
        )
        evaluators.extend(icl_evaluators)

    # Optimizer
    optimizer = build_optimizer(cfg.optimizer, model)

    # Scheduler
    scheduler = build_scheduler(cfg.scheduler)

    # Loggers
    loggers = [
        build_logger(name, logger_cfg)
        for name, logger_cfg in (cfg.get("loggers") or {}).items()
    ]

    # Callbacks
    callbacks = [
        build_callback(name, callback_cfg)
        for name, callback_cfg in (cfg.get("callbacks") or {}).items()
    ]

    # Algorithms
    algorithms = [
        build_algorithm(name, algorithm_cfg)
        for name, algorithm_cfg in (cfg.get("algorithms") or {}).items()
    ]

    # Build the Trainer
    print("Building trainer...")
    trainer = Trainer(
        run_name=cfg.run_name,
        seed=cfg.seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=cfg.max_duration,
        eval_interval=cfg.eval_interval,
        eval_subset_num_batches=cfg.get("eval_subset_num_batches", -1),
        progress_bar=cfg.get("progress_bar", False),
        log_to_console=cfg.get("log_to_console", True),
        console_log_interval=cfg.get("console_log_interval", "1ba"),
        loggers=loggers,
        callbacks=callbacks,
        precision=cfg.precision,
        algorithms=algorithms,
        device_train_microbatch_size=cfg.get("device_train_microbatch_size", "auto"),
        fsdp_config=fsdp_config,  # type: ignore
        save_folder=cfg.get("save_folder", None),
        save_filename=cfg.get("save_filename", "ep{epoch}-ba{batch}-rank{rank}.pt"),
        save_latest_filename=cfg.get("save_latest_filename", "latest-rank{rank}.pt"),
        save_interval=cfg.get("save_interval", "1000ba"),
        save_num_checkpoints_to_keep=cfg.get("save_num_checkpoints_to_keep", -1),
        save_overwrite=cfg.get("save_overwrite", False),
        load_path=cfg.get("load_path", None),
        load_weights_only=cfg.get("load_weights_only", False),
        load_ignore_keys=cfg.get("load_ignore_keys", None),
        autoresume=cfg.get("autoresume", False),
        python_log_level=cfg.get("python_log_level", None),
        dist_timeout=cfg.dist_timeout,
    )

    print("Logging config...")
    log_config(cfg)

    if cfg.get("eval_first", False) and trainer.state.timestamp.batch.value == 0:
        trainer.eval()

    print("Starting training...")
    trainer.fit()

    print("Done.")


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    main(cfg)  # type: ignore
