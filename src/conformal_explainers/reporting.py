"""从已有测试结果生成报告；只做统计与绘图，不重新校准或选择参数。"""

import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from .calibration import wilson_interval
from .data import digest_file


def read_results(folder):
    """核验完整配置矩阵及样本身份，防止重复/缺失记录改变统计权重。"""
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata.get("status") != "complete":
        raise ValueError("Report requires a complete run; finish evaluation first")
    if not isinstance(metadata.get("synthetic"), bool):
        raise ValueError("metadata must explicitly identify synthetic results")
    cfg = metadata["config"]
    expected = set(product(cfg["methods"], cfg["variants"], cfg["confidences"]))
    groups = defaultdict(list)
    identities, timings, seen = {}, {}, set()
    with (folder / "per_image.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            for key in ("confidence", "retained_fraction", "attribution_seconds"):
                row[key] = float(row[key])
                if not np.isfinite(row[key]):
                    raise ValueError(f"Non-finite {key}")
            for key in (
                "label",
                "prediction",
                "masked_prediction",
                "agreement",
                "correct",
            ):
                row[key] = int(row[key])
            if not 0 < row["confidence"] < 1:
                raise ValueError("Confidence must be in (0, 1)")
            if not 0 <= row["retained_fraction"] <= 1 or row["attribution_seconds"] < 0:
                raise ValueError("Invalid retained fraction or attribution time")
            if row["agreement"] != int(row["prediction"] == row["masked_prediction"]):
                raise ValueError("Agreement disagrees with saved predictions")
            if row["correct"] != int(row["prediction"] == row["label"]):
                raise ValueError("Correctness disagrees with saved label")
            key = (row["method"], row["variant"], row["confidence"])
            sample = row["sample_id"]
            if not sample or (key, sample) in seen:
                raise ValueError("Empty or duplicate sample in a configuration")
            seen.add((key, sample))
            identity = (row["label"], row["prediction"])
            if identities.setdefault(sample, identity) != identity:
                raise ValueError(
                    "Label or original prediction changes across configurations"
                )
            # 同一种方法对同一张图只计时一次，随后在多个配置中复用。
            timing_key = (row["method"], sample)
            if (
                timings.setdefault(timing_key, row["attribution_seconds"])
                != row["attribution_seconds"]
            ):
                raise ValueError("Attribution time changes across configurations")
            groups[key].append(row)
    if not expected or set(groups) != expected:
        raise ValueError("Results do not match the configured experiment matrix")
    sample_sets = [{r["sample_id"] for r in rows} for rows in groups.values()]
    if any(s != sample_sets[0] or len(s) != metadata["test_n"] for s in sample_sets):
        raise ValueError("Configurations must contain the same complete test set")
    return metadata, groups


def summarize(groups):
    """直接由逐图记录计算指标，避免报告误用过期的 summary.csv。"""
    summaries = []
    for (method, variant, confidence), rows in sorted(groups.items()):
        n = len(rows)
        successes = sum(r["agreement"] for r in rows)
        lo, hi = wilson_interval(successes, n)
        summaries.append(
            dict(
                method=method,
                variant=variant,
                confidence=confidence,
                n=n,
                failures=n - successes,
                fidelity=successes / n,
                fidelity_ci_low=lo,
                fidelity_ci_high=hi,
                fidelity_minus_target=successes / n - confidence,
                retained_fraction=float(
                    np.mean([r["retained_fraction"] for r in rows])
                ),
                classifier_accuracy=float(np.mean([r["correct"] for r in rows])),
            )
        )
    return summaries


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summaries, timings, output, synthetic):
    # 使用非交互后端，服务器与 CI 不需要桌面或显示器。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    variants = sorted({r["variant"] for r in summaries})
    methods = sorted({r["method"] for r in summaries})
    label = "SYNTHETIC SOFTWARE CHECK" if synthetic else "Held-out test results"
    fig, axes = plt.subplots(
        2, len(variants), figsize=(5 * len(variants), 8), squeeze=False
    )
    for j, variant in enumerate(variants):
        confidences = sorted({r["confidence"] for r in summaries})
        axes[0, j].plot(confidences, confidences, "k--", label="Target")
        for method in methods:
            rows = sorted(
                [
                    r
                    for r in summaries
                    if r["method"] == method and r["variant"] == variant
                ],
                key=lambda r: r["confidence"],
            )
            x = [r["confidence"] for r in rows]
            y = np.array([r["fidelity"] for r in rows])
            error = np.maximum(
                0,
                [
                    y - [r["fidelity_ci_low"] for r in rows],
                    [r["fidelity_ci_high"] for r in rows] - y,
                ],
            )
            axes[0, j].errorbar(x, y, yerr=error, marker="o", capsize=3, label=method)
            axes[1, j].plot(
                x, [r["retained_fraction"] for r in rows], marker="o", label=method
            )
        axes[0, j].set_title(variant)
        for i in range(2):
            axes[i, j].set_ylim(-0.03, 1.03)
            axes[i, j].set_xlabel("Target confidence")
            axes[i, j].xaxis.set_major_formatter(PercentFormatter(1))
            axes[i, j].yaxis.set_major_formatter(PercentFormatter(1))
            axes[i, j].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Fidelity (pointwise 95% Wilson interval)")
    axes[1, 0].set_ylabel("Mean retained area")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)))
    fig.suptitle(label)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"confidence_curves.{extension}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    means = [r["mean_seconds"] * 1000 for r in timings]
    bars = ax.bar([r["method"] for r in timings], means)
    ax.bar_label(bars, labels=[f"{v:.3g}" for v in means], padding=3)
    ax.set_ylim(0, max(means) * 1.2 if max(means) > 0 else 1)
    ax.set_ylabel("Mean attribution time per test image (ms)")
    ax.set_title(
        label + "\nRecorded single-run timing; excludes training and mask search"
    )
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"attribution_timing.{extension}", dpi=180)
    plt.close(fig)


def generate_report(results, output=None):
    """保留原始实验文件，将可复核统计、图表及中文说明写到新目录。"""
    results = Path(results).resolve()
    output = Path(output).resolve() if output else results / "report"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Choose a fresh report output directory: {output}")
    metadata, groups = read_results(results)
    summaries = summarize(groups)
    failures = [r for key in sorted(groups) for r in groups[key] if not r["agreement"]]
    timings = []
    for method in sorted({key[0] for key in groups}):
        # 取任意一个完整配置即可覆盖每个测试样本，不能把重复行相加。
        rows = next(rows for key, rows in groups.items() if key[0] == method)
        seconds = [r["attribution_seconds"] for r in rows]
        timings.append(
            dict(
                method=method,
                n=len(rows),
                mean_seconds=float(np.mean(seconds)),
                median_seconds=float(np.median(seconds)),
                total_seconds=float(np.sum(seconds)),
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "metrics.csv", summaries, list(summaries[0]))
    fields = list(next(iter(groups.values()))[0])
    # 即使没有失败也输出表头，便于下游脚本按相同格式读取。
    write_csv(output / "failures.csv", failures, fields)
    write_csv(output / "timing.csv", timings, list(timings[0]))
    plot_results(summaries, timings, output, metadata["synthetic"])
    provenance = {
        "metadata_sha256": digest_file(results / "metadata.json"),
        "per_image_sha256": digest_file(results / "per_image.csv"),
        "experiment_metadata": metadata,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    below = sum(r["fidelity_minus_target"] < 0 for r in summaries)
    heading = (
        "合成数据软件检查（不代表 Imagenette 实验结论）"
        if metadata["synthetic"]
        else "独立测试集实验结果"
    )
    lines = [
        f"# {heading}",
        "",
        f"校准样本：{metadata['calibration_n']}；测试样本：{metadata['test_n']}；配置数：{len(summaries)}。",
        f"{below} 组配置的实测 fidelity 低于目标置信度；这是描述性比较，不是显著性检验或可靠性证明。",
        "各配置重复使用同一批测试图片，不能把配置数乘以样本量当成独立观测数。",
        "",
        "## 置信度与保留面积",
        "",
        "![Confidence curves](confidence_curves.png)",
        "",
        "误差棒为各配置的 95% Wilson 区间，不是同时置信带。Fidelity 比较掩码前后预测是否一致，"
        "并不表示分类正确率。保留面积越小表示使用的区域越少，需与 fidelity 一起解释。",
        "",
        "| 方法 | 校准方式 | 目标 | n | Fidelity [95% CI] | 保留面积 | 原图准确率 | 失败数 |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in summaries:
        lines.append(
            f"| {r['method']} | {r['variant']} | {r['confidence']:.0%} | {r['n']} | "
            f"{r['fidelity']:.1%} [{r['fidelity_ci_low']:.1%}, {r['fidelity_ci_high']:.1%}] | "
            f"{r['retained_fraction']:.1%} | {r['classifier_accuracy']:.1%} | {r['failures']} |"
        )
    lines += [
        "",
        "## 归因耗时",
        "",
        "![Attribution timing](attribution_timing.png)",
        "",
        "每种方法每张测试图仅统计一次；不叠加多个校准方式/置信度复用的计时。"
        "图中不含模型训练、校准搜索、掩码推理或 FastSHAP 训练成本。"
        "这是当前设备单次运行的记录，尚未进行预热和重复基准测量。",
        "",
        "## 失败案例与可追溯文件",
        "",
        f"[failures.csv](failures.csv) 包含全部 {len(failures)} 条预测改变记录，"
        "同一图片可能在多个配置出现。sample_id 可用于回查图片。"
        "已有 NPZ 示例仅保存测试集前若干张，未必包含所有失败图片。",
        "",
        "[metrics.csv](metrics.csv) 由逐图记录重新计算；[timing.csv](timing.csv) 保存去重计时。"
        "[provenance.json](provenance.json) 保存输入文件 SHA256 和实验元数据。"
        "图表同时输出 PNG 与矢量 PDF，便于报告引用。",
        "",
        "本报告没有重新校准、选择最佳方法或修改实验参数。测试结果用于最终分析，不能用于调参。",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return output / "report.md"
