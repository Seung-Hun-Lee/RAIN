"""RAIN evaluation with GT and simulator masks on LIBERO."""
import argparse
import os
import sys


def install_model():
    import rain.models.model as implementation
    from .model import PoolingModel
    implementation.RAINModel = PoolingModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes-json", required=True)
    parser.add_argument("--benchmark", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"], default="libero_10")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--episode-ids")
    parser.add_argument("--tasks")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--feas-threshold", type=float, default=0.7)
    parser.add_argument("--consecutive-stop", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    install_model()
    command = [sys.argv[0], "--model-type", "rain", "--use-sim-seg", "--num-parallel", "1", "--dino-input-size", "224", "--replan-steps", "8", "--num-inference-steps", "4"]
    # One checkpoint contains both action and Transition Head tensors.
    command.extend(["--progress-checkpoint", args.checkpoint])
    for key, value in vars(args).items():
        if value is None or value is False:
            continue
        command.append("--" + key.replace("_", "-"))
        if value is not True:
            command.append(str(value))
    sys.argv = command
    from eval.eval_libero import main as evaluate
    evaluate()


# Spawn children must install the same model class before creating CUDA workers.
if __name__ in ("__main__", "__mp_main__"):
    install_model()
    if __name__ == "__main__":
        main()
