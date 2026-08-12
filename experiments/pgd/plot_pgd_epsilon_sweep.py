"""Plot a curated PGD-Linf epsilon sweep from a sweep manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


EPSILON_LABELS = {
    0.0: "0",
    1.0 / 255.0: "1/255",
    2.0 / 255.0: "2/255",
    4.0 / 255.0: "4/255",
    8.0 / 255.0: "8/255",
    12.0 / 255.0: "12/255",
    16.0 / 255.0: "16/255",
}


def epsilon_label(value: float) -> str:
    for expected, label in EPSILON_LABELS.items():
        if abs(value - expected) < 1e-12:
            return label
    return f"{value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate final PGD-Linf epsilon sweep evidence."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/curated/portfolio"),
    )
    parser.add_argument(
        "--plot-filename",
        default="final_pgd_epsilon_sweep.png",
    )
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text(encoding="utf-8"))

    if data.get("experiment_type") != "pgd_linf_epsilon_sweep":
        raise ValueError("Unexpected experiment_type.")

    if data.get("sample_count") != 10000:
        raise ValueError("Expected sample_count=10000.")

    if data.get("steps") != 10:
        raise ValueError("Expected steps=10.")

    runs = data.get("runs", [])
    if len(runs) != 7:
        raise ValueError("Expected exactly 7 epsilon runs.")

    for run in runs:
        if run.get("status") != "COMPLETED":
            raise ValueError("All sweep runs must be COMPLETED.")

    epsilons = [float(run["epsilon"]) for run in runs]
    clean = [100.0 * float(run["clean_accuracy"]) for run in runs]
    adversarial = [100.0 * float(run["adversarial_accuracy"]) for run in runs]
    asr = [100.0 * float(run["attack_success_rate"]) for run in runs]
    labels = [epsilon_label(value) for value in epsilons]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = args.output_dir / args.plot_filename
    csv_path = args.output_dir / "pgd_epsilon_sweep_summary.csv"
    json_path = args.output_dir / "pgd_epsilon_sweep_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epsilon",
                "epsilon_label",
                "clean_accuracy",
                "adversarial_accuracy",
                "accuracy_drop",
                "attack_success_rate",
                "run_id",
            ]
        )
        for run, label in zip(runs, labels):
            writer.writerow(
                [
                    run["epsilon"],
                    label,
                    run["clean_accuracy"],
                    run["adversarial_accuracy"],
                    run["accuracy_drop"],
                    run["attack_success_rate"],
                    run["run_id"],
                ]
            )

    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_type": "pgd_linf_epsilon_sweep",
                "sample_count": data["sample_count"],
                "backend": data["backend"],
                "batch_size": data["batch_size"],
                "alpha": data["alpha"],
                "steps": data["steps"],
                "random_start": data["random_start"],
                "seed": data["seed"],
                "epsilons": epsilons,
                "epsilon_labels": labels,
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(labels, clean, marker="o", linewidth=2.5, label="Clean accuracy baseline")
    ax.plot(labels, adversarial, marker="o", linewidth=2.5, label="PGD adversarial accuracy")
    ax.plot(labels, asr, marker="o", linewidth=2.5, label="Attack success rate")

    ax.set_title(
        "Full CIFAR-10 Test-Set PGD-Linf Robustness | "
        "10,000 samples | alpha=2/255 | 10 steps"
    )
    ax.set_xlabel("PGD epsilon")
    ax.set_ylabel("Percent")
    ax.set_ylim(-2, 103)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"summary_csv: {csv_path}")
    print(f"summary_json: {json_path}")
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
