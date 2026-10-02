#!/usr/bin/env python3
"""Extract multi-scale DINOv2 features from LIBERO parquet images → packed memmap.

Extracts features from 3 intermediate ViT layers (stages 2, 3, 4 per ViT-Adapter convention).
For ViT-L (24 blocks): layers 11, 17, 23
For ViT-B (12 blocks): layers 5, 8, 11

Multi-GPU via subprocess (avoids mp.spawn issues).

Output (--output-dir):
  dino_packed_s{1,2,3}.npy   (N, num_patches, embed_dim) float16  [patch tokens only, no CLS]
  dino_index.json             {"EEEE_FFFF_VIEW": row_index, ...}
  dino_meta.json              metadata including num_scales=3, scale_layers

Usage:
  python scripts/extract_features/extract_dinov2_ms.py \
      --variant large --input-size 224 \
      --output-dir /data/LIBERO/libero/packed_dinov2_large_224_ms \
      --gpus 0,1,2,3,6,7 --batch-size 64
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision.transforms as T
from PIL import Image

# ---------------------------------------------------------------------------
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 14
VIEWS = [
    ("observation.images.image", "third"),
    ("observation.images.image2", "wrist"),
]

# DINOv2 variant → (torch_hub_name, embed_dim, scale_layers)
# scale_layers: block indices for stages 2, 3, 4 (ViT-Adapter convention)
VARIANTS = {
    "small":  ("dinov2_vits14_reg", 384,  [5, 8, 11]),
    "base":   ("dinov2_vitb14_reg", 768,  [5, 8, 11]),
    "large":  ("dinov2_vitl14_reg", 1024, [11, 17, 23]),
    "giant":  ("dinov2_vitg14_reg", 1536, [13, 26, 39]),
}


DEFAULT_PARQUET_ENVVAR = "LIBERO_PARQUET_DIR"


def resolve_parquet_dir(parquet_dir: str) -> Path:
    if parquet_dir:
        return Path(parquet_dir)
    env_dir = os.environ.get(DEFAULT_PARQUET_ENVVAR, "")
    if env_dir:
        return Path(env_dir)
    raise ValueError(
        "--parquet-dir is required unless LIBERO_PARQUET_DIR is set"
    )


def compute_num_patches(input_size: int) -> int:
    """Patch tokens only (no CLS)."""
    return (input_size // PATCH_SIZE) ** 2


def get_all_parquets(parquet_dir: str = ""):
    data_dir = resolve_parquet_dir(parquet_dir)
    paths = sorted(str(p) for p in data_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files below {data_dir}")
    for path in paths:
        columns = set(pq.read_schema(path).names)
        required = {"episode_index", "frame_index", *(column for column, _ in VIEWS)}
        if not required.issubset(columns):
            raise ValueError(f"{path}: missing original image columns {sorted(required-columns)}; replay PNGs are not an exact replacement")
    return paths


def build_index(parquet_files):
    entries = []
    for fpath in parquet_files:
        t = pq.read_table(fpath, columns=["episode_index", "frame_index"])
        for row_i, (eidx, fidx) in enumerate(
            zip(t["episode_index"].to_pylist(), t["frame_index"].to_pylist())
        ):
            for _, view_name in VIEWS:
                entries.append({
                    "file": fpath, "row": row_i,
                    "eidx": eidx, "fidx": fidx, "view": view_name,
                })
    str_index = {
        f"{e['eidx']:04d}_{e['fidx']:04d}_{e['view']}": i
        for i, e in enumerate(entries)
    }
    if len(str_index) != len(entries):
        raise ValueError("Duplicate episode/frame/view keys; select a single canonical parquet tree")
    return entries, str_index


def process_batch(model, device, tensors, indices, out_arrs, scale_layers):
    batch = torch.stack(tensors).to(device, non_blocking=True)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        # get_intermediate_layers returns patch tokens only (no CLS)
        outputs = model.get_intermediate_layers(batch, n=scale_layers, reshape=False, norm=True)
    for s_idx, feat in enumerate(outputs):
        tokens_np = feat.cpu().float().numpy().astype(np.float16)
        for i, gi in enumerate(indices):
            out_arrs[s_idx][int(gi)] = tokens_np[i]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def run_worker(args):
    gpu = args.gpu
    tag = f"[GPU{gpu}]"

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(gpu)

    parquet_files = get_all_parquets(args.parquet_dir)
    entries, _ = build_index(parquet_files)
    total_n = args.total_n
    num_patches = args.num_patches
    embed_dim = args.embed_dim
    scale_layers = json.loads(args.scale_layers)

    # Open 3 memmaps
    out_arrs = []
    for s in range(3):
        npy_path = str(Path(args.output_dir) / f"dino_packed_s{s+1}.npy")
        arr = np.memmap(npy_path, dtype="float16", mode="r+",
                        shape=(total_n, num_patches, embed_dim))
        out_arrs.append(arr)

    hub_name = VARIANTS[args.variant][0]
    print(f"{tag} Loading DINOv2 {args.variant} ({hub_name})...", flush=True)
    model = torch.hub.load("facebookresearch/dinov2", hub_name, verbose=False)
    model = model.to(device).eval()
    print(f"{tag} DINOv2 loaded. Scale layers: {scale_layers}", flush=True)

    transform = T.Compose([
        T.Resize((args.input_size, args.input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=DINO_MEAN, std=DINO_STD),
    ])

    # Shard by GPU list index
    all_files = sorted(set(e["file"] for e in entries))
    gpu_list = [int(g) for g in args.gpu_list.split(",")]
    my_index = gpu_list.index(gpu)
    num_workers = len(gpu_list)
    my_files = all_files[my_index::num_workers]

    file_groups = defaultdict(list)
    for gi, e in enumerate(entries):
        if e["file"] in set(my_files):
            file_groups[e["file"]].append((gi, e))

    n_done = 0
    my_total = sum(len(v) for v in file_groups.values())
    t0 = time.time()

    def decode_and_transform(raw_bytes):
        img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        return transform(img)

    pool = ThreadPoolExecutor(max_workers=8)

    for file_idx, fpath in enumerate(my_files):
        group = file_groups.get(fpath, [])
        if not group:
            continue

        table = pq.read_table(fpath, columns=[v[0] for v in VIEWS])

        raw_items = []
        for gi, entry in group:
            col = VIEWS[0][0] if entry["view"] == "third" else VIEWS[1][0]
            cell = table[col][entry["row"]].as_py()
            raw_items.append((gi, cell["bytes"]))

        for batch_start in range(0, len(raw_items), args.batch_size):
            batch_raw = raw_items[batch_start:batch_start + args.batch_size]
            batch_indices = [gi for gi, _ in batch_raw]
            futures = [pool.submit(decode_and_transform, raw) for _, raw in batch_raw]
            batch_tensors = [f.result() for f in futures]
            process_batch(model, device, batch_tensors, batch_indices, out_arrs, scale_layers)
            n_done += len(batch_tensors)

        if (file_idx + 1) % 5 == 0 or file_idx == 0:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 1
            eta = (my_total - n_done) / rate if rate > 0 else 0
            print(f"{tag} [{file_idx+1}/{len(my_files)}] {n_done}/{my_total} "
                  f"({rate:.1f} img/s, ETA {eta/60:.0f} min)", flush=True)

    pool.shutdown()
    for arr in out_arrs:
        arr.flush()
    elapsed = time.time() - t0
    rate = n_done / elapsed if elapsed > 0 else 0
    print(f"{tag} Done! {n_done} images in {elapsed/60:.1f} min ({rate:.1f} img/s)", flush=True)


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------
def run_launcher(args):
    variant = args.variant
    hub_name, embed_dim, scale_layers = VARIANTS[variant]
    input_size = args.input_size
    num_patches = compute_num_patches(input_size)

    gpu_list = [int(g) for g in args.gpus.split(",")]

    out_dir = Path(args.output_dir)
    if any(out_dir.glob("dino_*")):
        raise FileExistsError(f"Refusing to overwrite an existing feature extraction: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "dino_index.json"

    print(f"=== DINOv2 Multi-Scale {variant} ({hub_name}) ===", flush=True)
    print(f"Input: {input_size}x{input_size}, Patches: {num_patches}, Dim: {embed_dim}", flush=True)
    print(f"Scale layers: {scale_layers}", flush=True)
    print(f"GPUs: {gpu_list}, batch_size={args.batch_size}", flush=True)

    # Build index
    print("Scanning parquet files...", flush=True)
    t0 = time.time()
    parquet_files = get_all_parquets(args.parquet_dir)
    entries, str_index = build_index(parquet_files)
    total_n = len(entries)
    print(f"Total entries: {total_n} ({time.time()-t0:.1f}s)", flush=True)

    with open(index_path, "w") as f:
        json.dump(str_index, f)
    print(f"Index saved: {index_path}", flush=True)

    # Allocate 3 memmaps (float16, patch tokens only)
    bytes_per_elem = 2
    size_per_scale_gb = total_n * num_patches * embed_dim * bytes_per_elem / 1e9
    expected_per_scale = total_n * num_patches * embed_dim * bytes_per_elem

    for s in range(3):
        npy_path = out_dir / f"dino_packed_s{s+1}.npy"
        if npy_path.exists() and npy_path.stat().st_size == expected_per_scale:
            print(f"Memmap s{s+1} exists ({size_per_scale_gb:.2f} GB) -- reusing", flush=True)
        else:
            print(f"Allocating memmap s{s+1}: {size_per_scale_gb:.2f} GB ...", flush=True)
            arr = np.memmap(str(npy_path), dtype="float16", mode="w+",
                            shape=(total_n, num_patches, embed_dim))
            del arr
    print(f"Total storage: {size_per_scale_gb * 3:.2f} GB (3 scales)", flush=True)

    # Save metadata
    meta = {
        "dtype": "float16",
        "num_patches": num_patches,
        "num_tokens": num_patches,  # compat: no CLS in multi-scale
        "hidden_dim": embed_dim,
        "encoder": "dinov2",
        "variant": variant,
        "input_size": input_size,
        "patch_size": PATCH_SIZE,
        "hub_name": hub_name,
        "num_register_tokens": 0,
        "multi_scale": True,
        "num_scales": 3,
        "scale_layers": scale_layers,
        "source_images": "original_parquet_bytes",
        "preprocessing": "PIL RGB; torchvision PIL bicubic Resize224; ToTensor; ImageNet normalize",
        "completed": False,
    }
    with open(out_dir / "dino_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved dino_meta.json", flush=True)

    # Spawn workers
    script = os.path.abspath(__file__)
    procs = []
    for gpu in gpu_list:
        cmd = [
            sys.executable, "-u", script,
            "--worker",
            "--gpu", str(gpu),
            "--gpu-list", args.gpus,
            "--variant", variant,
            "--input-size", str(input_size),
            "--parquet-dir", args.parquet_dir,
            "--output-dir", args.output_dir,
            "--batch-size", str(args.batch_size),
            "--total-n", str(total_n),
            "--num-patches", str(num_patches),
            "--embed-dim", str(embed_dim),
            "--scale-layers", json.dumps(scale_layers),
        ]
        p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        procs.append(p)
        print(f"Spawned worker GPU{gpu} (PID {p.pid})", flush=True)

    for p in procs:
        p.wait()

    failed = [p for p in procs if p.returncode != 0]
    if failed:
        print(f"ERROR: {len(failed)} workers failed!", flush=True)
        sys.exit(1)

    # Verify each scale
    for s in range(3):
        npy_path = str(out_dir / f"dino_packed_s{s+1}.npy")
        out_arr = np.memmap(npy_path, dtype="float16", mode="r",
                            shape=(total_n, num_patches, embed_dim))
        n_nonzero = int((out_arr[:, 0, 0] != 0).sum())
        print(f"Scale {s+1} verification: {n_nonzero}/{total_n} non-zero "
              f"({100*n_nonzero/max(total_n,1):.1f}%)", flush=True)

    meta["completed"] = True
    with open(out_dir / "dino_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("All done!", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="large", choices=["large"],
                        help="Final recipe: DINOv2-L/14 with four register tokens")
    parser.add_argument("--input-size", type=int, default=224, choices=[224],
                        help="Input image size (must be divisible by 14)")
    parser.add_argument(
        "--parquet-dir",
        default="",
        help="Merged parquet dir. If omitted, LIBERO_PARQUET_DIR must be set.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--batch-size", type=int, default=64)
    # Worker-mode args
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-list", type=str, default="")
    parser.add_argument("--total-n", type=int, default=0)
    parser.add_argument("--num-patches", type=int, default=0)
    parser.add_argument("--embed-dim", type=int, default=0)
    parser.add_argument("--scale-layers", type=str, default="[]")
    args = parser.parse_args()

    if args.input_size % PATCH_SIZE != 0:
        print(f"ERROR: input-size {args.input_size} must be divisible by patch_size {PATCH_SIZE}")
        sys.exit(1)

    if args.worker:
        run_worker(args)
    else:
        run_launcher(args)


if __name__ == "__main__":
    main()
