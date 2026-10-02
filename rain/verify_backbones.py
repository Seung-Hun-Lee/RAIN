"""Verify the recorded local DINO source and pretrained weight identities (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(dino_repo, dino_weights):
    manifest = json.loads((Path(__file__).resolve().parents[1] / "configs/pretrained.json").read_text())
    identity = manifest["dinov2"]
    for relative, expected in identity["source_files"].items():
        path = Path(dino_repo) / relative
        if sha256(path) != expected:
            raise ValueError(f"DINO source mismatch: {relative}; do not claim the recorded source contract")
    if sha256(dino_weights) != identity["sha256"]:
        raise ValueError("DINO pretrained weights differ from the recorded checkpoint")
    return {"dinov2_model": identity["hub_model"], "verified_source_files": len(identity["source_files"]),
            "weight_sha256": identity["sha256"], "gpu_initialized": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-repo", required=True)
    parser.add_argument("--dino-weights", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.dino_repo, args.dino_weights), indent=2))


if __name__ == "__main__":
    main()
