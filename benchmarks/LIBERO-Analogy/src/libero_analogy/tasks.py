"""Portable frozen task inventory, without simulator or torch imports."""
import hashlib
import json
import os
from pathlib import Path

import yaml


def benchmark_root(root=None):
    candidate = Path(root or os.environ.get("LIBERO_ANALOGY_ROOT") or Path(__file__).resolve().parents[2]).resolve()
    if not (candidate / "TASK_INDEX.json").is_file():
        raise FileNotFoundError("Specify --benchmark-root /path/to/LIBERO-Analogy (or LIBERO_ANALOGY_ROOT)")
    return candidate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_index(root=None):
    root = benchmark_root(root)
    rows = json.loads((root / "TASK_INDEX.json").read_text())["tasks"]
    for row in rows:
        path = Path(row["bundle"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe bundle path: {path}")
    return rows


def validate(root=None):
    root = benchmark_root(root)
    rows = load_index(root)
    expected = {f"{category}_{i:03d}" for category in ("Decompose", "Adapt", "Compose") for i in range(1, 21)}
    if len(rows) != 60 or {row["task_id"] for row in rows} != expected:
        raise ValueError("The frozen release must contain exactly 20 tasks in each category")
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    for item in manifest["files"]:
        if sha256(root / item["target"]) != item["target_sha256"]:
            raise ValueError(f"Release hash mismatch: {item['target']}")
    for row in rows:
        bundle = root / row["bundle"]
        meta = yaml.safe_load((bundle / "task_meta.yaml").read_text())
        rules = yaml.safe_load((bundle / "eval_rules.yaml").read_text())
        if meta["task_id"] != row["task_id"] or rules["task_id"] != row["task_id"]:
            raise ValueError(f"Task identity mismatch: {row['task_id']}")
        if " ".join(meta["language"].split()) != row["instruction"]:
            raise ValueError(f"Instruction mismatch: {row['task_id']}")
    return {"tasks": len(rows), "source_backed_files_verified": len(manifest["files"]),
            "task_index_sha256": sha256(root / "TASK_INDEX.json")}

