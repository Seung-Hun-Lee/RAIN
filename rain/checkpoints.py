"""Strict RAIN checkpoint restoration with checkpoint-compatible tensor keys."""
import torch


def _load_subset(model, path, *, transition, variant=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if transition and checkpoint.get("progress_architecture") != variant:
        raise ValueError(f"Transition architecture {checkpoint.get('progress_architecture')!r} != {variant!r}")
    state = checkpoint.get("model_state_dict", checkpoint)
    current = model.state_dict()
    select = lambda k: k.startswith("fusion_branch.") == transition
    expected = {k for k in current if select(k)}
    actual = {k for k in state if select(k)}
    if not expected or expected != actual:
        raise ValueError(f"Checkpoint key mismatch: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
    bad = [k for k in expected if current[k].shape != state[k].shape or current[k].dtype != state[k].dtype]
    if bad:
        raise ValueError(f"Checkpoint tensor shape/dtype mismatch: {bad}")
    model.load_state_dict({k: state[k] for k in expected}, strict=False)
    loaded = model.state_dict()
    if not all(torch.equal(loaded[k].detach().cpu(), state[k]) for k in expected):
        raise RuntimeError("Tensor values changed during checkpoint restoration")
    return checkpoint


def load_exact_action(model, path):
    return _load_subset(model, path, transition=False)


def load_exact_transition(model, path, variant="region_gated"):
    return _load_subset(model, path, transition=True, variant=variant)
