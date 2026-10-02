"""Explicit, process-local LIBERO asset configuration; no implicit downloads."""
import json
import os
import tempfile
from pathlib import Path

from .tasks import benchmark_root, sha256

_CONFIG = None


def asset_root():
    path = Path(os.environ.get("LIBERO_ANALOGY_ASSETS", Path.home() / ".cache/libero/assets")).expanduser().resolve()
    if not (path / "articulated_objects/microwave.xml").is_file():
        raise FileNotFoundError("Set LIBERO_ANALOGY_ASSETS to the upstream asset directory; see README.md")
    return path


def verify_assets(root=None):
    manifest = json.loads((benchmark_root(root) / "ASSET_MANIFEST.json").read_text())
    assets = asset_root()
    for item in manifest["files"]:
        if sha256(assets / item["path"]) != item["sha256"]:
            raise ValueError(f"Upstream asset changed: {item['path']}")
    return {"asset_files_verified": len(manifest["files"]), "bytes": sum(item["bytes"] for item in manifest["files"])}


def configure():
    """Avoid upstream's interactive ~/.libero initialization without editing it."""
    global _CONFIG
    if _CONFIG is not None:
        return
    import yaml
    assets = asset_root()
    _CONFIG = tempfile.TemporaryDirectory(prefix="libero-analogy-config-")
    os.environ["LIBERO_CONFIG_PATH"] = _CONFIG.name
    config = {key: str(assets) for key in ("benchmark_root", "bddl_files", "init_states", "datasets", "assets")}
    Path(_CONFIG.name, "config.yaml").write_text(yaml.safe_dump(config))
    import libero.libero as libero
    # The pinned fork uses both get_libero_path and get_assets_path.
    libero.config_file = str(Path(_CONFIG.name, "config.yaml"))
    libero._assets_path_cache = str(assets)

