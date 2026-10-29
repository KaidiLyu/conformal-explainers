import csv
import json

import pytest

from conformal_explainers.reporting import generate_report, read_results, summarize


@pytest.fixture
def saved_run(tmp_path):
    """两张图、两个置信度；一张原图预测错误但掩码前后保持一致。"""
    metadata = dict(
        status="complete",
        synthetic=True,
        calibration_n=8,
        test_n=2,
        config=dict(
            methods=["saliency"], variants=["pixelwise"], confidences=[0.9, 0.95]
        ),
    )
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    rows = []
    for confidence in (0.9, 0.95):
        for i in range(2):
            rows.append(
                dict(
                    method="saliency",
                    variant="pixelwise",
                    confidence=confidence,
                    sample_id=f"test/{i}.png",
                    label=0,
                    prediction=i,
                    masked_prediction=1,
                    agreement=i,
                    correct=1 - i,
                    retained_fraction=0.25 + i * 0.5,
                    attribution_seconds=0.01 + i * 0.02,
                )
            )
    save_rows(tmp_path, rows)
    return tmp_path, rows


def save_rows(folder, rows):
    with (folder / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_metrics_distinguish_fidelity_and_correctness(saved_run):
    folder, _ = saved_run
    _, groups = read_results(folder)
    for row in summarize(groups):
        assert row["n"] == 2
        assert row["failures"] == 1
        assert row["fidelity"] == 0.5
        assert row["retained_fraction"] == 0.5
        assert row["fidelity_ci_low"] == pytest.approx(0.0945312)
        assert row["fidelity_ci_high"] == pytest.approx(0.9054688)


def test_report_outputs_and_deduplicated_timing(saved_run):
    folder, _ = saved_run
    report = generate_report(folder)
    assert "合成数据软件检查" in report.read_text()
    assert "全部 2 条预测改变记录" in report.read_text()
    with (report.parent / "timing.csv").open() as f:
        row = next(csv.DictReader(f))
    assert int(row["n"]) == 2
    assert float(row["total_seconds"]) == pytest.approx(0.04)
    assert float(row["mean_seconds"]) == pytest.approx(0.02)
    for name in ("confidence_curves", "attribution_timing"):
        assert (report.parent / f"{name}.pdf").read_bytes().startswith(b"%PDF")
        assert (report.parent / f"{name}.png").read_bytes().startswith(b"\x89PNG")
    provenance = json.loads((report.parent / "provenance.json").read_text())
    assert len(provenance["per_image_sha256"]) == 64
    with pytest.raises(ValueError, match="fresh"):
        generate_report(folder)


@pytest.mark.parametrize(
    "problem",
    ["duplicate", "missing", "identity", "timing", "nan", "agreement", "sample"],
)
def test_reject_corrupt_results_before_writing(saved_run, problem):
    folder, rows = saved_run
    if problem == "duplicate":
        rows.append(rows[0].copy())
    elif problem == "missing":
        rows.pop()
    elif problem == "identity":
        rows[-1]["label"] = 1
        rows[-1]["correct"] = 1
    elif problem == "timing":
        rows[-1]["attribution_seconds"] = 0.2
    elif problem == "nan":
        rows[-1]["retained_fraction"] = "nan"
    elif problem == "agreement":
        rows[-1]["agreement"] = 0
    elif problem == "sample":
        rows[-1]["sample_id"] = "different.png"
    save_rows(folder, rows)
    with pytest.raises(ValueError):
        generate_report(folder)
    assert not (folder / "report").exists()


def test_reject_incomplete_run(saved_run):
    folder, _ = saved_run
    path = folder / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["status"] = "running"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="complete run"):
        generate_report(folder)


def test_empty_failures_has_header(saved_run):
    folder, rows = saved_run
    for row in rows:
        row["masked_prediction"] = row["prediction"]
        row["agreement"] = 1
    save_rows(folder, rows)
    report = generate_report(folder)
    with (report.parent / "failures.csv").open() as f:
        reader = csv.DictReader(f)
        assert "sample_id" in reader.fieldnames
        assert list(reader) == []
