import json
from pathlib import Path

from PIL import Image

from conformal_explainers.data import write_splits


def test_deterministic_disjoint_stratified_splits(tmp_path):
    for split in ["train", "val"]:
        for c in range(10):
            folder = tmp_path / split / str(c)
            folder.mkdir(parents=True)
            for i in range(12):
                Image.new("RGB", (2, 2), (c, i, 0)).save(folder / f"{i}.png")
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for p in paths:
        write_splits(tmp_path, p, 42)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    m = json.loads(Path(paths[0]).read_text())
    sets = [set(r["path"] for r in rows) for rows in m["splits"].values()]
    assert sum(map(len, sets)) == len(set.union(*sets)) == 240
    assert all(len({r["label"] for r in rows}) == 10 for rows in m["splits"].values())
