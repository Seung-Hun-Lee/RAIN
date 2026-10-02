#!/usr/bin/env python3
"""Precompute one CLIP text cache before distributed RAIN training starts."""

import argparse
from pathlib import Path

from shared.clip_utils import DEFAULT_CLIP_TEXT_MODEL, get_text_cache_path
from shared.data.episode_io import load_episode_records
from shared.data.packed_features import TextFeatureCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-json", required=True)
    parser.add_argument("--packed-features-dir", required=True)
    parser.add_argument("--clip-model-name", default=DEFAULT_CLIP_TEXT_MODEL)
    args = parser.parse_args()

    episodes = load_episode_records(args.episodes_json)
    descriptions = sorted(
        {
            str(segment.get("description", "")).strip()
            for episode in episodes.values()
            for segment in episode.get("subtask_segments", [])
            if str(segment.get("description", "")).strip()
        }
    )
    output_path = get_text_cache_path(
        Path(args.packed_features_dir), args.clip_model_name
    )
    cache = TextFeatureCache(model_name=args.clip_model_name)
    cache.precompute(descriptions, save=True, cache_path=str(output_path))
    if not output_path.exists():
        raise RuntimeError(f"CLIP cache was not written: {output_path}")
    print(f"Saved {len(cache.features)} text features to {output_path}")


if __name__ == "__main__":
    main()
