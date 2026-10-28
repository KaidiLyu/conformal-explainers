"""训练分类器和 FastSHAP；训练/调参只使用 train 和 validation。"""

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fastshap import FastSHAP
from fastshap.image_imputers import BaselineImageImputer
from fastshap.utils import DatasetInputOnly

from .data import ImageDataset, digest_file
from .explainers import Probabilities
from .models import GridExplainer, build_classifier, load_classifier


def dataset(cfg, split, augment=False, limit=None):
    return ImageDataset(
        cfg["data_root"], cfg["manifest"], split, cfg["image_size"], augment, limit
    )


def train_classifier(cfg, device):
    """以验证集准确率选最佳模型；校准集和测试集不参与模型选择。"""
    train = dataset(cfg, "train", augment=True, limit=cfg.get("train_limit"))
    val = dataset(cfg, "validation", limit=cfg.get("validation_limit"))
    model = build_classifier(len(train.manifest["classes"]), cfg["pretrained"]).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    loaders = {
        "train": DataLoader(train, batch_size=cfg["batch_size"], shuffle=True),
        "validation": DataLoader(val, batch_size=cfg["batch_size"]),
    }
    best, history = -1.0, []
    output = Path(cfg["classifier_checkpoint"])
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg["epochs"]):
        record = {"epoch": epoch + 1}
        for split, loader in loaders.items():
            model.train(split == "train")
            total = correct = 0
            loss_sum = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.set_grad_enabled(split == "train"):
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(logits, y)
                    if split == "train":
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                total += len(y)
                correct += (logits.argmax(1) == y).sum().item()
                loss_sum += loss.item() * len(y)
            record[split + "_accuracy"] = correct / total
            record[split + "_loss"] = loss_sum / total
        history.append(record)
        print(json.dumps(record), flush=True)
        if record["validation_accuracy"] > best:
            best = record["validation_accuracy"]
            torch.save(
                dict(
                    state_dict=model.state_dict(),
                    classes=train.manifest["classes"],
                    image_size=cfg["image_size"],
                    manifest_sha256=digest_file(cfg["manifest"]),
                    config=cfg,
                    validation_accuracy=best,
                ),
                output,
            )
    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2))


def check_classifier_metadata(meta, cfg):
    if meta["image_size"] != cfg["image_size"]:
        raise ValueError("Classifier image size differs from experiment configuration")
    if meta["manifest_sha256"] != digest_file(cfg["manifest"]):
        raise ValueError("Classifier and experiment must share the same split manifest")


def make_fastshap(model, size, classes, grid, device):
    if size % grid:
        raise ValueError("image_size must be divisible by explainer grid")
    net = GridExplainer(classes, grid).to(device)
    imputer = BaselineImageImputer(
        Probabilities(model).eval(), 0.0, size, size, size // grid
    )
    return FastSHAP(net, imputer, normalization="additive")


def train_fastshap(cfg, device):
    """冻结分类器，以随机特征子集上的预测训练 FastSHAP 解释网络。"""
    model, meta = load_classifier(cfg["classifier_checkpoint"], device)
    check_classifier_metadata(meta, cfg)
    model.requires_grad_(False)
    fs = cfg["fastshap"]
    train = DatasetInputOnly(dataset(cfg, "train", limit=cfg.get("train_limit")))
    val = DatasetInputOnly(
        dataset(cfg, "validation", limit=cfg.get("validation_limit"))
    )
    if len(train) < fs["batch_size"]:
        raise ValueError("FastSHAP train set must contain at least one full batch")
    model_fs = make_fastshap(
        model, cfg["image_size"], len(meta["classes"]), cfg["grid"], device
    )
    start = time.perf_counter()
    model_fs.train(
        train,
        val,
        batch_size=fs["batch_size"],
        num_samples=fs["samples"],
        max_epochs=fs["epochs"],
        lr=fs["learning_rate"],
        lookback=fs["lookback"],
        validation_samples=fs["validation_samples"],
        training_seed=cfg["seed"],
        validation_seed=cfg["seed"] + 1,
        verbose=True,
    )
    output = Path(cfg["fastshap_checkpoint"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            state_dict=model_fs.explainer.state_dict(),
            null=model_fs.null.cpu(),
            classifier_sha256=digest_file(cfg["classifier_checkpoint"]),
            manifest_sha256=digest_file(cfg["manifest"]),
            grid=cfg["grid"],
            image_size=cfg["image_size"],
            training_seconds=time.perf_counter() - start,
            config=cfg,
        ),
        output,
    )


def load_fastshap(cfg, model, classes, device):
    """加载解释器，并验证其对应的分类器、数据划分和网格配置。"""
    c = torch.load(cfg["fastshap_checkpoint"], map_location="cpu", weights_only=True)
    if c["classifier_sha256"] != digest_file(cfg["classifier_checkpoint"]):
        raise ValueError("FastSHAP was trained for a different classifier checkpoint")
    if c["manifest_sha256"] != digest_file(cfg["manifest"]):
        raise ValueError("FastSHAP split manifest mismatch")
    if c["grid"] != cfg["grid"] or c["image_size"] != cfg["image_size"]:
        raise ValueError("FastSHAP grid/image size mismatch")
    f = make_fastshap(model, cfg["image_size"], classes, cfg["grid"], device)
    f.explainer.load_state_dict(c["state_dict"])
    f.explainer.eval()
    f.null = c["null"].to(device)
    return f
