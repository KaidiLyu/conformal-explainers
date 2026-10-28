import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import write_splits
from .experiment import run_experiment
from .training import train_classifier, train_fastshap


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def main():
    p = argparse.ArgumentParser(
        description="Bamboo Steamer conformal explanation experiments"
    )
    sub = p.add_subparsers(dest="command", required=True)
    split = sub.add_parser("split")
    split.add_argument("--data-root", required=True)
    split.add_argument("--output", default="data/splits.json")
    split.add_argument("--seed", type=int, default=42)
    for name in ("train", "train-fastshap", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", default="configs/imagenette.yaml")
        cmd.add_argument(
            "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
        )
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--output", default="outputs/smoke")
    args = p.parse_args()
    # 下载的权重默认放在项目 data/ 内，避免污染组员的全局缓存。
    os.environ.setdefault("TORCH_HOME", str(Path("data/torch-cache").resolve()))
    if args.command == "split":
        print(
            json.dumps(write_splits(args.data_root, args.output, args.seed), indent=2)
        )
        return
    if args.command == "smoke":
        from .smoke import run_smoke

        run_smoke(Path(args.output))
        return
    cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = choose_device(args.device)
    print(f"Running {args.command} on {device}", flush=True)
    {
        "train": train_classifier,
        "train-fastshap": train_fastshap,
        "evaluate": run_experiment,
    }[args.command](cfg, device)


if __name__ == "__main__":
    main()
