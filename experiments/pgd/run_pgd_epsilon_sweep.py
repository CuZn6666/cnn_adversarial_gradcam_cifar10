"""Run a fixed PGD-Linf epsilon sweep using the existing single-run runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.fgsm.run_fgsm_experiment import parse_epsilon_values
from experiments.pgd.run_pgd_experiment import (
    PGDExperimentConfig,
    run_pgd_experiment,
)


DEFAULT_EPSILONS = "0,1/255,2/255,4/255,8/255,12/255,16/255"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple isolated PGD-Linf experiments by reusing the "
            "production single-epsilon PGD runner."
        )
    )
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="cupy")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoints/portfolio_baseline_best.npz",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epsilons", default=DEFAULT_EPSILONS)
    parser.add_argument("--alpha", default="2/255")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--random-start",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="results/runs")
    parser.add_argument(
        "--manifest-output",
        default="results/pgd_epsilon_sweep_manifest.json",
    )

    args = parser.parse_args()

    epsilons = parse_epsilon_values(args.epsilons)
    alpha_values = parse_epsilon_values(args.alpha)

    if len(alpha_values) != 1:
        raise ValueError("--alpha must contain exactly one value.")

    alpha = float(alpha_values[0])

    runs = []

    for epsilon in epsilons:
        print(f"\n=== PGD epsilon={epsilon:.12g} ===", flush=True)

        config = PGDExperimentConfig(
            backend=args.backend,
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            split=args.split,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            epsilon=float(epsilon),
            alpha=alpha,
            steps=args.steps,
            random_start=args.random_start,
            seed=args.seed,
            output_root=args.output_root,
        )

        run_result = run_pgd_experiment(config)
        metrics = run_result["pgd_result"]

        runs.append(
            {
                "epsilon": float(epsilon),
                "run_id": config.run_id,
                "run_dir": str(run_result["run_dir"]),
                "status": "COMPLETED",
                "clean_accuracy": metrics["clean_accuracy"],
                "adversarial_accuracy": metrics["adversarial_accuracy"],
                "accuracy_drop": metrics["accuracy_drop"],
                "attack_success_rate": metrics["attack_success_rate"],
            }
        )

        print(
            f"completed: {config.run_id} | "
            f"adv_acc={metrics['adversarial_accuracy']:.6f} | "
            f"asr={metrics['attack_success_rate']:.6f}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "experiment_type": "pgd_linf_epsilon_sweep",
        "backend": args.backend,
        "split": args.split,
        "sample_count": args.max_samples,
        "batch_size": args.batch_size,
        "epsilons": [float(value) for value in epsilons],
        "alpha": alpha,
        "steps": args.steps,
        "random_start": args.random_start,
        "seed": args.seed,
        "runs": runs,
    }

    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\nmanifest: {manifest_path}")


if __name__ == "__main__":
    main()
