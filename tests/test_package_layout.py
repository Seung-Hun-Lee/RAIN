"""The public model, configuration, and trainers share one package."""

import importlib
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_single_model_package():
    packages = sorted(path.name for path in ROOT.glob("rain*") if (path / "__init__.py").is_file())
    assert packages == ["rain"]
    assert not (ROOT / "rainv2").exists()


def test_public_exports_use_package_implementation():
    import rain
    from rain.model import PoolingModel
    from rain.models.model import RAINModel
    from rain.models.vision_encoder import TargetAdaptiveCrossViewEncoder, TarLN
    from rain.transition_head import PoolingTCHead
    from shared.plan_dit import PlanDiT

    assert rain.__all__ == ["RAIN", "TCE", "TarLN", "PlanDiT", "TransitionHead"]
    assert rain.RAIN is PoolingModel
    assert issubclass(PoolingModel, RAINModel)
    assert rain.TCE is TargetAdaptiveCrossViewEncoder
    assert rain.TarLN is TarLN
    assert rain.PlanDiT is PlanDiT
    assert rain.TransitionHead is PoolingTCHead


@pytest.mark.parametrize("module", [
    "rain.configs.config", "rain.models.model", "rain.models.vision_encoder",
    "rain.models.progress_heads", "rain.training.action", "rain.training.transition",
])
def test_implementation_imports_from_package(module):
    imported = importlib.import_module(module)
    assert Path(imported.__file__).resolve().is_relative_to(ROOT / "rain")


@pytest.mark.parametrize("module", [
    "rain.train", "rain.training.action", "rain.training.transition",
    "rain.train_transition", "rain.eval_libero", "rain.eval_analogy",
])
def test_entrypoint_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
