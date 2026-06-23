#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brian_sphere_llm.data.dataloader import build_dataloader
from brian_sphere_llm.train.stage_runner import build_model_from_config, train_mode_for_stage
from brian_sphere_llm.train.trainer import _device, _wrap_distributed_model, evaluate
from brian_sphere_llm.utils import distributed as dist_utils
from brian_sphere_llm.utils.config import load_config
from brian_sphere_llm.utils.logging import write_json
from brian_sphere_llm.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Check eval invariance between single-process and DDP rank 0.")
    parser.add_argument("--config", required=True, help="Train config to instantiate.")
    parser.add_argument("--split", default=None, help="Eval split override, e.g. val_legacy.")
    parser.add_argument("--batch-size", type=int, default=None, help="Eval batch size override.")
    parser.add_argument("--max-batches", type=int, default=1, help="Number of eval batches for the smoke check.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    seed = int(config.get("seed", 1))
    set_seed(seed)

    device = _device(str(config.get("device", "auto")))
    distributed = dist_utils.init_distributed(device)
    model_config_path = (config_path.parent / str(config["model_config"])).resolve()
    data_config_path = (config_path.parent / str(config["data_config"])).resolve()
    data_config = load_config(data_config_path)
    model = build_model_from_config(model_config_path)
    model.to(device)
    model = _wrap_distributed_model(
        model,
        device,
        distributed=distributed,
        find_unused_parameters=bool(config.get("ddp_find_unused_parameters", True)),
    )

    split = str(args.split or config.get("eval_split", "val"))
    batch_size = int(args.batch_size or config["batch_size"])
    eval_config = {**config, "eval_max_batches": int(args.max_batches)}
    loader = build_dataloader(
        tokenized_dir=data_config["output_dir"],
        split=split,
        batch_size=batch_size,
        shuffle=False,
    )
    row = evaluate(
        model,
        loader,
        config=eval_config,
        device=device,
        route_mode=train_mode_for_stage(str(config["stage"])),
        global_step=0,
    )
    payload = {
        "config": str(config_path),
        "split": split,
        "batch_size": batch_size,
        "max_batches": int(args.max_batches),
        "distributed": distributed,
        "world_size": dist_utils.world_size(),
        "rank": dist_utils.rank(),
        "row": row,
    }
    if dist_utils.is_main_process():
        write_json(payload, args.output)
        print(json.dumps(payload, sort_keys=True))
    dist_utils.barrier()
    dist_utils.destroy_distributed()


if __name__ == "__main__":
    main()
