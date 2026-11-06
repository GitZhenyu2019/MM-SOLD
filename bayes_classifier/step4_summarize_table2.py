"""
step4_summarize_table2.py
==============================
Collect the four result files produced by step2/step3 and print + save the
final digit classification results table
"""
import argparse
import json
import os

import config as cfg  # noqa: E402


ROWS = [
    ("MLP", "mlp_baseline"),
    ("MLP (MM-SOLD Aug.)", "mlp_augmented"),
    ("MM Target Min. ECM C. (plug-in)", "ecm_plugin"),
    ("MM Target Min. ECM C. (dual)", "ecm_dual"),
]


def _load(results_dir, subdir):
    path = os.path.join(results_dir, subdir, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _per_class_and_overall(summary, subdir):
    if summary is None:
        return None, None
    if subdir.startswith("mlp"):
        return summary["per_class_acc_test"], summary["test_acc"]
    else:  # ecm_*
        best = summary["best"]
        return best["per_class_acc_test"], best["test_acc"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=cfg.RESULTS_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(args.results_dir, "table2.json")

    table = {}
    print("=" * 100)
    print("digit classification results table: handwritten digit classification in the top-40 NRAE latent subspace")
    print("=" * 100)
    header = f"{'Method':<34}" + "".join(f"{d:>6}" for d in range(10)) + f"{'Overall':>9}"
    print(header)
    print("-" * len(header))

    missing = []
    for label, subdir in ROWS:
        summary = _load(args.results_dir, subdir)
        per_class, overall = _per_class_and_overall(summary, subdir)
        if per_class is None:
            missing.append(subdir)
            print(f"{label:<34}" + "  [missing -- run the corresponding step]")
            continue
        row_str = "".join(f"{100*a:>6.1f}" for a in per_class) + f"{100*overall:>9.2f}"
        print(f"{label:<34}{row_str}")
        table[label] = {"per_class_acc": per_class, "overall_acc": overall,
                        "source": subdir}

    print("-" * len(header))
    if missing:
        print(f"\n[NOTE] missing results for: {missing} -- table above is incomplete.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2, default=float)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
