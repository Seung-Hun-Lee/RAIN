"""Validate training configuration against the selected ablation recipe."""
from dataclasses import asdict
import json
from pathlib import Path


def config_differences(expected, actual, prefix=""):
    differences = []
    for key in sorted(set(expected) | set(actual)):
        name = f"{prefix}.{key}" if prefix else key
        if key not in expected or key not in actual:
            differences.append((name, expected.get(key), actual.get(key)))
        elif isinstance(expected[key], dict) and isinstance(actual[key], dict):
            differences.extend(config_differences(expected[key], actual[key], name))
        elif expected[key] != actual[key]:
            differences.append((name, expected[key], actual[key]))
    return differences


def check_training_config(config, expected_path):
    expected = json.loads(Path(expected_path).read_text())
    differences = config_differences(expected, asdict(config))
    if differences:
        raise RuntimeError(f"Ablation config mismatch (expected, actual): {differences}")
    print(f"ABLATION_CONFIG_CHECK_PASSED: {expected_path}", flush=True)
