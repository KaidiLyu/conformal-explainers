"""生成可复现的 Imagenette 四路划分：训练、验证、校准、测试。"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def digest_file(path):
    """记录文件指纹，防止模型、解释器、数据划分被不小心混用。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_splits(root, output, seed=42):
    """按类别分层抽样，仅保存相对路径，便于组员共享同一个划分清单。"""
    from sklearn.model_selection import train_test_split

    root = Path(root).resolve()
    classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    if len(classes) != 10:
        raise ValueError("Expected Imagenette train/ with 10 class folders")
    groups = {}
    for split in ("train", "val"):
        records = []
        for label, name in enumerate(classes):
            for path in sorted((root / split / name).rglob("*")):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    records.append(
                        {"path": str(path.relative_to(root)), "label": label}
                    )
        if not records:
            raise ValueError(f"No images found in {split}")
        groups[split] = records
    train = groups["train"]
    evaluation = groups["val"]
    # 标准 Imagenette 使用论文表 5 的数量，但随机样本身份不保证与作者一致。
    val_n = 947 if len(train) == 9469 else max(10, round(len(train) * 0.1))
    test_n = (
        1001 if len(evaluation) == 3925 else max(10, round(len(evaluation) * 0.255))
    )
    train, val = train_test_split(
        train, test_size=val_n, random_state=seed, stratify=[r["label"] for r in train]
    )
    cal, test = train_test_split(
        evaluation,
        test_size=test_n,
        random_state=seed,
        stratify=[r["label"] for r in evaluation],
    )
    splits = dict(train=train, validation=val, calibration=cal, test=test)
    paths = [r["path"] for rows in splits.values() for r in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Data leakage: overlapping paths")
    manifest = dict(seed=seed, classes=classes, splits=splits)
    output = Path(output)
    payload = json.dumps(manifest, indent=2)
    if output.exists() and output.read_text() != payload:
        raise ValueError(
            "Refusing to replace an existing split manifest; use a new path"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload)
    return {k: len(v) for k, v in splits.items()}


class ImageDataset(Dataset):
    def __init__(self, root, manifest, split, size=224, augment=False, limit=None):
        self.root = Path(root).resolve()
        self.manifest = json.loads(Path(manifest).read_text())
        self.records = self.manifest["splits"][split]
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            self.records = self.records[:limit]
        ops = [transforms.Resize((size, size))]
        if augment:
            ops.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(
            ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        path = (self.root / r["path"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Manifest path escapes data directory")
        with Image.open(path) as im:
            image = self.transform(im.convert("RGB"))
        return image, r["label"]


def to_rgb(image):
    """还原归一化图像以便显示；归一化空间的零对应 RGB 均值而非黑色。"""
    return np.clip(image.detach().cpu().permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
