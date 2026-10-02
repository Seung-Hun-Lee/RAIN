"""Expose final-stage packed features for the single-stage ablation (no recomputation)."""
import argparse
import json
import os
from pathlib import Path


def prepare(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    names = {"dino_index.json": "dino_index.json", "dino_packed.npy": "dino_packed_s3.npy", "text_features.npz": "text_features.npz"}
    for name in names.values():
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    meta = json.loads((source / "dino_meta.json").read_text())
    meta.update(multi_scale=False, num_scales=1, patch_only=True, scale_layers=[23])
    target.mkdir(parents=True)
    for name, original in names.items():
        (target / name).symlink_to(os.path.relpath(source / original, target))
    (target / "dino_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("target")
    args = parser.parse_args()
    prepare(args.source, args.target)


if __name__ == "__main__":
    main()
