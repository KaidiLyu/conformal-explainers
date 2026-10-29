from types import SimpleNamespace

import pytest
import torch

from conformal_explainers.pilot import run, summarize_measurements, write_json


def test_pilot_summary_excludes_warmup_and_separates_methods():
    def measurement(seconds, memory):
        return dict(
            seconds=seconds, peak_allocated_mib=memory, peak_reserved_mib=memory * 2
        )

    report = dict(
        train_steps=[
            dict(warmup=True, **measurement(100, 1000)),
            dict(warmup=False, **measurement(2, 20)),
            dict(warmup=False, **measurement(4, 30)),
        ],
        attribution=[
            dict(method="saliency", **measurement(0.1, 10)),
            dict(method="kernel_shap", **measurement(5, 80)),
        ],
        calibration_search=[
            dict(method="saliency", variant="scaled", **measurement(3, 40))
        ],
    )
    summary = summarize_measurements(report)
    assert summary["train"]["n"] == 2
    assert summary["train"]["mean_seconds"] == 3
    assert summary["train"]["max_peak_allocated_mib"] == 30
    assert summary["attribution/kernel_shap"]["mean_seconds"] == 5
    assert summary["search/saliency/scaled"]["n"] == 1


def test_json_rejects_nonfinite_without_damaging_previous_result(tmp_path):
    path = tmp_path / "pilot.json"
    write_json(path, {"status": "running"})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {"seconds": float("nan")})
    assert path.read_bytes() == original


def test_no_cuda_does_not_claim_gpu_results(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = SimpleNamespace(
        batch_size=2,
        steps=2,
        images=1,
        mask_batch_size=2,
        samples=64,
        threshold_steps=10,
        output=str(tmp_path / "gpu"),
    )
    with pytest.raises(RuntimeError, match="CUDA GPU is required"):
        run(args)
    assert not (tmp_path / "gpu").exists()
