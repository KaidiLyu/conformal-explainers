"""真实 Imagenette CUDA 试跑：短训练、验证、归因与校准搜索计时。

只用 train/validation，不读取最终 calibration/test。试跑权重不能用于论文对照。
"""

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from skimage.segmentation import slic

from conformal_explainers.calibration import MaskConfig, conformity_score
from conformal_explainers.data import ImageDataset, digest_file, to_rgb
from conformal_explainers.explainers import Explainer
from conformal_explainers.models import build_classifier


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def timed(call):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, dict(
        seconds=time.perf_counter() - start,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
    )


def summarize_measurements(report):
    """排除训练首步预热；按方法/校准方式汇总，保留小样本试跑标签。"""
    groups = {"train": [r for r in report["train_steps"] if not r["warmup"]]}
    for row in report["attribution"]:
        groups.setdefault("attribution/" + row["method"], []).append(row)
    for row in report["calibration_search"]:
        groups.setdefault("search/" + row["method"] + "/" + row["variant"], []).append(
            row
        )
    return {
        name: dict(
            n=len(rows),
            mean_seconds=float(np.mean([r["seconds"] for r in rows])),
            median_seconds=float(np.median([r["seconds"] for r in rows])),
            max_peak_allocated_mib=max(r["peak_allocated_mib"] for r in rows),
            max_peak_reserved_mib=max(r["peak_reserved_mib"] for r in rows),
        )
        for name, rows in groups.items()
        if rows
    }


def write_summary(output, report):
    lines = [
        "# 真实图片 GPU 试跑",
        "",
        f"GPU：{report['gpu']}；精度：{report['precision']}。",
        "仅为小规模工程测试，不是正式模型准确率或论文复现结果。",
        "",
        "| 阶段 | 次数 | 平均秒 | 中位秒 | 已分配显存峰值 MiB | 保留显存峰值 MiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in report["summary"].items():
        lines.append(
            f"| {name} | {r['n']} | {r['mean_seconds']:.4f} | {r['median_seconds']:.4f} | "
            f"{r['max_peak_allocated_mib']:.1f} | {r['max_peak_reserved_mib']:.1f} |"
        )
    lines += [
        "",
        "训练首步计为预热并从汇总排除；归因各自预热一次。训练按批次计时，归因和搜索按单图计时。",
        "计时不包含图片读取、训练数据传输、checkpoint 写盘；分割耗时另记。KernelSHAP 包含 CPU 拟合。",
        "显存来自 PyTorch CUDA 分配器，包含模型与该阶段中仍存活的张量，不是 nvidia-smi 的整个设备占用。",
        "正式模型、不同采样预算或不同 batch size 的成本可能不同，不能直接当作最终基准。",
        "",
    ]
    (output / "summary.md").write_text("\n".join(lines))


def run(args):
    if min(args.batch_size, args.steps, args.images, args.mask_batch_size) < 1:
        raise ValueError("Batch sizes, steps and images must be positive")
    if args.samples < 2 or args.threshold_steps < 2:
        raise ValueError("At least two SHAP samples and threshold steps are required")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required; CPU/MPS is not a GPU pilot substitute"
        )
    audit = json.loads(Path(args.manifest).with_suffix(".audit.json").read_text())
    if digest_file(args.manifest) != audit["manifest_sha256"]:
        raise ValueError("Manifest no longer matches its frozen audit")
    saved = json.loads(Path(args.manifest).read_text())
    if args.steps * args.batch_size > len(
        saved["splits"]["train"]
    ) or args.images > len(saved["splits"]["validation"]):
        raise ValueError("Pilot requests more train/validation images than available")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    os.environ.setdefault("TORCH_HOME", str(Path("data/torch-cache").resolve()))
    props = torch.cuda.get_device_properties(0)
    report = dict(
        status="running",
        purpose="real-image engineering pilot, not paper reproduction",
        gpu=props.name,
        gpu_total_mib=props.total_memory / 2**20,
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        platform=platform.platform(),
        slurm_job_id=os.getenv("SLURM_JOB_ID"),
        manifest_sha256=digest_file(args.manifest),
        args=vars(args),
        initialization="ImageNet pretrained ResNet50; random 10-class head",
        normalization="ImageNet mean/std (current baseline, not paper training statistics)",
        precision="FP32",
        train_steps=[],
        attribution=[],
        calibration_search=[],
    )

    def save():
        write_json(output / "pilot.json", report)

    save()
    train = ImageDataset(
        args.data_root, args.manifest, "train", limit=args.batch_size * args.steps
    )
    val = ImageDataset(
        args.data_root, args.manifest, "validation", limit=max(10, args.images)
    )
    start = time.perf_counter()
    model = build_classifier(classes=10, pretrained=True).cuda()
    report["model_setup_seconds"] = time.perf_counter() - start
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(args.steps):
        # IO 和主机到 GPU 传输不计入纯训练步骤计时，另行记录。
        start = time.perf_counter()
        records = [train[step * args.batch_size + i] for i in range(args.batch_size)]
        x = torch.stack([r[0] for r in records]).cuda()
        y = torch.tensor([r[1] for r in records], device="cuda")
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start

        def train_step():
            model.train()
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            opt.step()
            return float(loss.detach())

        loss, measurement = timed(train_step)
        report["train_steps"].append(
            dict(
                step=step,
                loss=loss,
                warmup=step == 0,
                load_seconds=load_seconds,
                **measurement,
            )
        )
        # 每一步保存试跑权重，节点关闭时保留已完成成果；不宣称支持正式训练续跑。
        temporary = output / "pilot_checkpoint.tmp"
        torch.save(
            dict(
                state_dict=model.state_dict(),
                optimizer=opt.state_dict(),
                completed_steps=step + 1,
                pilot_only=True,
                manifest_sha256=report["manifest_sha256"],
            ),
            temporary,
        )
        temporary.replace(output / "pilot_checkpoint.pt")
        save()
        print("TRAIN", report["train_steps"][-1], flush=True)
    opt.zero_grad(set_to_none=True)
    opt, x, y = None, None, None
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        correct = sum(
            int(model(image[None].cuda()).argmax(1).item() == label)
            for image, label in val
        )
    report["validation"] = dict(
        n=len(val),
        accuracy=correct / len(val),
        note="Tiny development subset; no generalization claim",
    )
    for method in ("saliency", "kernel_shap"):
        explainer = Explainer(model, method, samples=args.samples, grid=7)
        # 每种归因方法预热一次，不使用最终测试图片。
        explainer(val[0][0].cuda())
        for i in range(args.images):
            image = val[i][0].cuda()
            attr, measurement = timed(lambda: explainer(image))
            if not np.isfinite(attr).all():
                raise ValueError("Non-finite attribution")
            report["attribution"].append(
                dict(method=method, sample_id=val.records[i]["path"], **measurement)
            )
            start = time.perf_counter()
            segments = slic(
                to_rgb(image), n_segments=50, compactness=10, sigma=1, start_label=0
            )
            segmentation_seconds = time.perf_counter() - start
            for variant in ("pixelwise", "superpixels", "scaled"):
                score, measurement = timed(
                    lambda: conformity_score(
                        model,
                        image,
                        attr,
                        MaskConfig(variant),
                        segments,
                        steps=args.threshold_steps,
                        batch_size=args.mask_batch_size,
                    )
                )
                report["calibration_search"].append(
                    dict(
                        method=method,
                        variant=variant,
                        sample_id=val.records[i]["path"],
                        score=None if score == -np.inf else float(score),
                        segmentation_seconds=segmentation_seconds,
                        **measurement,
                    )
                )
            save()
            print("EXPLAIN", report["attribution"][-1], flush=True)
    report["status"] = "complete"
    report["summary"] = summarize_measurements(report)
    write_summary(output, report)
    save()
    print("PILOT COMPLETE", output / "pilot.json", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/imagenette2-320")
    parser.add_argument("--manifest", default="manifests/imagenette-seed42.json")
    parser.add_argument("--output", default="outputs/gpu-pilot")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--threshold-steps", type=int, default=100)
    parser.add_argument("--mask-batch-size", type=int, default=16)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
