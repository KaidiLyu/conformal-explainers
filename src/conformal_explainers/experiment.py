"""实验主流程：校准集求阈值 → 固定阈值 → 独立测试集评估。"""

import csv
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from skimage.segmentation import slic

from .calibration import (
    MaskConfig,
    calibrate_threshold,
    conformity_score,
    make_mask,
    wilson_interval,
)
from .data import digest_file, to_rgb
from .explainers import Explainer
from .models import load_classifier
from .training import check_classifier_metadata, dataset, load_fastshap


def synchronize(device):
    """GPU 运算默认异步；计时前后同步，避免只测到任务提交时间。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_experiment(cfg, device):
    """输出逐图结果、汇总指标、阈值、示例图以及实验配置指纹。"""
    output = Path(cfg["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Choose a fresh output_dir to preserve existing experiments: {output}"
        )
    model, meta = load_classifier(cfg["classifier_checkpoint"], device)
    check_classifier_metadata(meta, cfg)
    fs = (
        load_fastshap(cfg, model, len(meta["classes"]), device)
        if "fastshap" in cfg["methods"]
        else None
    )
    configs = [MaskConfig(v, cfg["rho"], cfg["rho_mode"]) for v in cfg["variants"]]
    cal = dataset(cfg, "calibration", limit=cfg.get("calibration_limit"))
    test = dataset(cfg, "test", limit=cfg.get("test_limit"))
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        config=cfg,
        classifier_sha256=digest_file(cfg["classifier_checkpoint"]),
        manifest_sha256=digest_file(cfg["manifest"]),
        classes=meta["classes"],
        torch_version=str(torch.__version__),
        platform=platform.platform(),
        device=str(device),
        calibration_n=len(cal),
        test_n=len(test),
        status="running",
        synthetic=bool(meta.get("synthetic", False)),
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    all_rows, summaries, threshold_records = [], [], []
    needs_segments = any(c.variant != "pixelwise" for c in configs)
    segment_cache = {}

    def segmentation(split, i, x):
        if not needs_segments:
            return None
        key = (split, i)
        if key not in segment_cache:
            segment_cache[key] = slic(
                to_rgb(x),
                n_segments=cfg["segments"],
                compactness=10,
                sigma=1,
                start_label=0,
            )
        return segment_cache[key]

    for method in cfg["methods"]:
        explainer = Explainer(
            model,
            method,
            samples=cfg["samples"],
            grid=cfg["grid"],
            noise=cfg["gradient_noise"],
            fastshap=fs,
        )
        scores = {c.variant: [] for c in configs}
        synchronize(device)
        start = time.perf_counter()
        for i, (x, _) in enumerate(cal):
            x = x.to(device)
            attr = explainer(x)
            seg = segmentation("calibration", i, x)
            for c in configs:
                scores[c.variant].append(
                    conformity_score(
                        model,
                        x,
                        attr,
                        c,
                        seg,
                        cfg["threshold_steps"],
                        cfg["mask_batch_size"],
                    )
                )
        synchronize(device)
        calibration_seconds = time.perf_counter() - start
        thresholds = {}
        for c in configs:
            for confidence in cfg["confidences"]:
                t = calibrate_threshold(
                    scores[c.variant], confidence, cfg["quantile_rule"]
                )
                thresholds[c.variant, confidence] = t
                threshold_records.append(
                    dict(
                        method=method,
                        **asdict(c),
                        confidence=confidence,
                        threshold=None if t == -np.inf else t,
                        full_image_fallback=t == -np.inf,
                        quantile_rule=cfg["quantile_rule"],
                        n_calibration=len(cal),
                    )
                )
        for i, (x, label) in enumerate(test):
            x = x.to(device)
            with torch.no_grad():
                prediction = model(x[None]).argmax(1).item()
            synchronize(device)
            start = time.perf_counter()
            attr = explainer(x, prediction)
            synchronize(device)
            attribution_seconds = time.perf_counter() - start
            seg = segmentation("test", i, x)
            for c in configs:
                for confidence in cfg["confidences"]:
                    mask = make_mask(attr, thresholds[c.variant, confidence], c, seg)
                    masked = (
                        x * torch.as_tensor(mask, device=device, dtype=x.dtype)[None]
                    )
                    with torch.no_grad():
                        masked_prediction = model(masked[None]).argmax(1).item()
                    row = dict(
                        method=method,
                        variant=c.variant,
                        confidence=confidence,
                        sample_id=test.records[i]["path"],
                        label=label,
                        prediction=prediction,
                        masked_prediction=masked_prediction,
                        agreement=int(prediction == masked_prediction),
                        correct=int(prediction == label),
                        retained_fraction=float(mask.mean()),
                        attribution_seconds=attribution_seconds,
                    )
                    all_rows.append(row)
                    if i < cfg["save_examples"]:
                        name = f"{method}_{c.variant}_{confidence}_{i}.npz"
                        np.savez_compressed(
                            output / name,
                            image=to_rgb(x),
                            attribution=attr,
                            mask=mask,
                            masked_image=to_rgb(masked),
                            **row,
                        )
        for c in configs:
            for confidence in cfg["confidences"]:
                rows = [
                    r
                    for r in all_rows
                    if r["method"] == method
                    and r["variant"] == c.variant
                    and r["confidence"] == confidence
                ]
                successes = sum(r["agreement"] for r in rows)
                lo, hi = wilson_interval(successes, len(rows))
                summaries.append(
                    dict(
                        method=method,
                        variant=c.variant,
                        confidence=confidence,
                        n=len(rows),
                        fidelity=successes / len(rows),
                        fidelity_ci_low=lo,
                        fidelity_ci_high=hi,
                        retained_fraction=float(
                            np.mean([r["retained_fraction"] for r in rows])
                        ),
                        classifier_accuracy=float(
                            np.mean([r["correct"] for r in rows])
                        ),
                        attribution_seconds=float(
                            np.mean([r["attribution_seconds"] for r in rows])
                        ),
                        calibration_seconds=calibration_seconds,
                    )
                )
        print(
            f"Completed {method}: {len(cal)} calibration / {len(test)} test images",
            flush=True,
        )
        # 每完成一种方法就保存，避免后续出错丢失已完成的结果。
        for filename, rows in [("per_image.csv", all_rows), ("summary.csv", summaries)]:
            with (output / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (output / "thresholds.json").write_text(
            json.dumps(threshold_records, indent=2, allow_nan=False)
        )
        # JSON 不支持标准的负无穷值；用 null 表示全图回退分数。
        (output / f"{method}_calibration_scores.json").write_text(
            json.dumps(
                {
                    k: [None if v == -np.inf else v for v in vs]
                    for k, vs in scores.items()
                },
                indent=2,
            )
        )
    metadata["status"] = "complete"
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return summaries
