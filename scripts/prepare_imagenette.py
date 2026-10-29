"""下载官方数据，固定划分，并逐文件检查可解码性与内容指纹。"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from torchvision.datasets import Imagenette
from torchvision.datasets.utils import check_integrity

from conformal_explainers.data import digest_file, write_splits


def prepare(root, manifest):
    root, manifest = Path(root), Path(manifest)
    Imagenette(root, size="320px", download=True)
    archive = root / "imagenette2-320.tgz"
    expected_md5 = "3df6f0d01a2c9592104656642f5e78a3"
    if not check_integrity(archive, expected_md5):
        raise ValueError("Official Imagenette archive checksum mismatch")
    dataset_root = root / "imagenette2-320"
    if not manifest.exists():
        write_splits(dataset_root, manifest, seed=42)
    saved = json.loads(manifest.read_text())
    expected = dict(train=8522, validation=947, calibration=2924, test=1001)
    if {k: len(v) for k, v in saved["splits"].items()} != expected:
        raise ValueError("Unexpected split sizes")
    if saved["classes"] != sorted(Imagenette._WNID_TO_CLASS):
        raise ValueError("Unexpected class mapping")
    paths, seen, content_hashes = [], set(), {}
    class_counts = {}
    for split, records in saved["splits"].items():
        class_counts[split] = {
            str(k): v for k, v in Counter(r["label"] for r in records).items()
        }
        for record in records:
            rel = record["path"]
            path = (dataset_root / rel).resolve()
            if not path.is_relative_to(dataset_root.resolve()) or rel in seen:
                raise ValueError("Invalid/overlapping image path")
            if Path(rel).parts[1] != saved["classes"][record["label"]]:
                raise ValueError("Image folder and label disagree")
            source = "train" if split in {"train", "validation"} else "val"
            if Path(rel).parts[0] != source:
                raise ValueError("Official training/evaluation boundaries crossed")
            seen.add(rel)
            with Image.open(path) as im:
                im.convert("RGB").load()
            sha = digest_file(path)
            content_hashes.setdefault(sha, []).append(rel)
            paths.append((rel, sha))
    actual = {
        str(p.relative_to(dataset_root))
        for p in dataset_root.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    if actual != seen:
        raise ValueError("Manifest does not cover exactly the dataset images")
    # 聚合指纹只依赖相对路径与文件内容，与本地/HPC 的目录位置无关。
    fingerprint = hashlib.sha256()
    for rel, sha in sorted(paths):
        fingerprint.update(f"{rel}\0{sha}\n".encode())
    audit = dict(
        counts=expected,
        class_counts=class_counts,
        images_decoded=len(paths),
        manifest_sha256=digest_file(manifest),
        dataset_sha256=fingerprint.hexdigest(),
        archive_md5=expected_md5,
        archive_sha256=digest_file(archive),
        duplicate_content_groups=[v for v in content_hashes.values() if len(v) > 1],
        source="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    )
    output = manifest.with_suffix(".audit.json")
    if output.exists() and json.loads(output.read_text()) != audit:
        raise ValueError(
            "Dataset/manifest differs from the frozen audit; refusing to replace it"
        )
    output.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    parser.add_argument("--manifest", default="manifests/imagenette-seed42.json")
    args = parser.parse_args()
    prepare(args.root, args.manifest)
