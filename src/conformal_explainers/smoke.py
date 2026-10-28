"""用合成图片跑通五种方法、三种校准和四个置信度；无需下载数据或 GPU。"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import digest_file
from .experiment import run_experiment
from .models import SmokeClassifier
from .training import train_fastshap


def run_smoke(output):
    output = Path(output).resolve()
    assets = output.parent / (output.name + "_assets")
    if assets.exists() or output.exists():
        raise ValueError(
            "Smoke outputs already exist; choose a fresh --output directory"
        )
    assets.mkdir(parents=True)
    torch.manual_seed(42)
    torch.set_num_threads(2)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    splits = {}
    for split, n in [("train", 12), ("validation", 4), ("calibration", 8), ("test", 4)]:
        splits[split] = []
        for i in range(n):
            pixels = rng.integers(80, 180, size=(16, 16, 3), dtype=np.uint8)
            pixels[:4, :4, 0] = 220 if i % 2 else 20
            path = f"{split}_{i}.png"
            Image.fromarray(pixels).save(assets / path)
            splits[split].append(dict(path=path, label=i % 2))
    manifest = assets / "splits.json"
    manifest.write_text(
        json.dumps(dict(classes=["dark", "bright"], splits=splits, seed=42))
    )
    checkpoint = assets / "classifier.pt"
    torch.save(
        dict(
            state_dict=SmokeClassifier().state_dict(),
            classes=["dark", "bright"],
            image_size=16,
            synthetic=True,
            manifest_sha256=digest_file(manifest),
        ),
        checkpoint,
    )
    cfg = dict(
        seed=42,
        data_root=str(assets),
        manifest=str(manifest),
        image_size=16,
        classifier_checkpoint=str(checkpoint),
        fastshap_checkpoint=str(assets / "fastshap.pt"),
        output_dir=str(output),
        grid=2,
        samples=32,
        gradient_noise=0.0,
        methods=[
            "saliency",
            "input_x_gradient",
            "gradient_shap",
            "kernel_shap",
            "fastshap",
        ],
        variants=["pixelwise", "superpixels", "scaled"],
        confidences=[0.85, 0.90, 0.95, 0.99],
        quantile_rule="paper",
        rho=0.5,
        rho_mode="fraction",
        segments=8,
        threshold_steps=8,
        mask_batch_size=8,
        save_examples=1,
        fastshap=dict(
            epochs=2,
            batch_size=4,
            samples=4,
            validation_samples=4,
            learning_rate=0.001,
            lookback=2,
        ),
    )
    train_fastshap(cfg, torch.device("cpu"))
    rows = run_experiment(cfg, torch.device("cpu"))
    assert len(rows) == 60
    assert all(
        np.isfinite(r["fidelity"]) and 0 <= r["retained_fraction"] <= 1 for r in rows
    )
    print(f"Synthetic integration check passed: 60 configurations. Results: {output}")
