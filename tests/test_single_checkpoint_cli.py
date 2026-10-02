"""Public CLI wiring checks without model, GPU, or simulator initialization."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_renderer_environment(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")


def load_cli(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "rain" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_libero_forwards_one_checkpoint_to_both_loader_slots(monkeypatch):
    cli = load_cli("eval_libero")
    forwarded = []
    monkeypatch.setattr(cli, "install_model", lambda: None)
    fake_module(monkeypatch, "eval", __path__=[])
    fake_module(monkeypatch, "eval.eval_libero", main=lambda: forwarded.extend(sys.argv))
    monkeypatch.setattr(sys, "argv", [
        "rain-eval-libero", "--checkpoint", "model/checkpoint.pt",
        "--episodes-json", "episodes.json", "--save-dir", "results",
    ])
    cli.main()
    for option in ("--checkpoint", "--progress-checkpoint"):
        assert forwarded.count(option) == 1
        assert forwarded[forwarded.index(option) + 1] == "model/checkpoint.pt"


def test_analogy_forwards_one_checkpoint_to_both_loader_slots(monkeypatch):
    cli = load_cli("eval_analogy")
    captured = []

    class WorkerIntercepted(Exception):
        pass

    def intercept_worker(*args, **kwargs):
        captured.append((args, kwargs))
        raise WorkerIntercepted

    fake_module(monkeypatch, "libero_analogy", __path__=[])
    fake_module(monkeypatch, "libero_analogy.tasks", validate=lambda root: None,
                load_index=lambda root: [])
    fake_module(monkeypatch, "final_libero_ex_eval", __path__=[])
    fake_module(monkeypatch, "final_libero_ex_eval.impl",
                runtime=SimpleNamespace(GPUInferenceWorker=intercept_worker),
                rollout=SimpleNamespace())
    monkeypatch.setattr(sys, "argv", [
        "rain-eval-analogy", "--benchmark-root", "benchmark",
        "--checkpoint", "model/checkpoint.pt",
    ])
    with pytest.raises(WorkerIntercepted):
        cli.main()
    assert captured == [(
        ("model/checkpoint.pt", "model/checkpoint.pt", "rainv2"),
        {"gpu_id": 0, "dino_input_size": 224},
    )]


@pytest.mark.parametrize("name,arguments", [
    ("eval_libero", ["--checkpoint", "model.pt", "--episodes-json", "episodes.json",
                     "--save-dir", "results"]),
    ("eval_analogy", ["--benchmark-root", "benchmark", "--checkpoint", "model.pt"]),
])
def test_separate_progress_checkpoint_option_is_rejected(name, arguments, monkeypatch, capsys):
    cli = load_cli(name)
    monkeypatch.setattr(sys, "argv", [name, *arguments, "--progress-checkpoint", "other.pt"])
    with pytest.raises(SystemExit) as raised:
        cli.main()
    assert raised.value.code == 2
    assert "unrecognized arguments: --progress-checkpoint other.pt" in capsys.readouterr().err
