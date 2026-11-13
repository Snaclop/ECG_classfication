"""Train the waveform-only CNN, or resume an existing run at its last complete epoch."""
import argparse
from pathlib import Path

if __package__:
    from .baseline_data import ROOT
    from .baseline_checkpoint import load_checkpoint
    from .baseline_training import make_config, run_training
else:
    from baseline_data import ROOT
    from baseline_checkpoint import load_checkpoint
    from baseline_training import make_config, run_training


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, help="Path to last.pt; saved settings are reused")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Default cpu; cuda also selects AMD ROCm")
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--data-root", type=Path, help="Root for waveform paths (default project/)")
    parser.add_argument("--output", type=Path, help="New run directory; existing directories are never overwritten")
    parser.add_argument("--epochs", type=int, help="Total epochs, default 30; must match on resume")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-threads", type=int, help="CPU compute threads, default min(4, CPU count)")
    parser.add_argument("--no-progress", action="store_true", default=None)
    parser.add_argument("--smoke", action="store_true", default=None,
                        help="Functional check only: defaults to 1 epoch, first 32 train / 64 validation rows")
    parser.add_argument("--train-limit", type=int, help="Requires --smoke")
    parser.add_argument("--validation-limit", type=int, help="Requires --smoke")
    args = vars(parser.parse_args(argv))
    resume = args.pop("resume")
    no_progress = args.pop("no_progress")
    config = load_checkpoint(resume)["config"].copy() if resume else {}
    if args["smoke"] and not resume:
        config.update(smoke=True, epochs=1, train_limit=32, validation_limit=64,
                      output=ROOT / "outputs/baseline_cnn_smoke")
    config.update({key: value for key, value in args.items() if value is not None})
    if no_progress:
        config["progress"] = False
    return make_config(**config), resume


def main(argv=None):
    config, resume = parse_config(argv)
    print("FUNCTIONAL SMOKE CHECK; not a formal experiment" if config["smoke"] else "Waveform CNN training", flush=True)
    print(f"Device: {config['device']}; output: {config['output']}", flush=True)
    try:
        output = run_training(config, resume)
    except KeyboardInterrupt:
        last = Path(config["output"]) / "last.pt"
        print(f"Interrupted. Resume from {last}" if last.exists() else
              "Interrupted before the first complete epoch. Start again with a new output directory.", flush=True)
        raise SystemExit(130)
    print(f"Run complete: {output}. Final test evaluation is a separate command.", flush=True)


if __name__ == "__main__":
    main()
