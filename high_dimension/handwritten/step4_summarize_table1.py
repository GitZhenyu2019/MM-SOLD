"""
step4_summarize_table1.py
==============================
Assemble digit-8 generation results table (digit-8 generation quality + computational cost)
"""
import argparse
import json
import os

import config as cfg


METHODS = [
    ("mmsold",      "MM-SOLD",       "results_mmsold.json",      "timing_mmsold.json"),
    ("cfdm",        "sigma-CFDM",    "results_cfdm.json",        "timing_cfdm.json"),
    ("knn_pflow",   "KNN-PFlow",     "results_knnpflow.json",    "timing_knnpflow.json"),
    ("swf",         "SWF",           "results_swf.json",         "timing_swf.json"),
    ("plugin_baoab","Plug-in BAOAB", "results_pluginbaoab.json", "timing_pluginbaoab.json"),
    ("sg_baoab",    "SG-BAOAB",      "results_sgbaoab.json",     "timing_sgbaoab.json"),
]
SUBDIRS = {
    "mmsold": "mmsold", "cfdm": "cfdm", "knn_pflow": "knnpflow", "swf": "swf",
    "plugin_baoab": "pluginbaoab", "sg_baoab": "sgbaoab",
}


def _best_over_grid(grid: dict):
    cells = list(grid.values())
    best_kid = min(cells, key=lambda c: c["kid_mean"])
    best_recall = max(cells, key=lambda c: c["recall_mean"])
    best_dup = min(cells, key=lambda c: c["duprate_mean"])
    return best_kid, best_recall, best_dup


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=cfg.RESULTS_DIR)
    a = ap.parse_args()

    table = {}

    for key, label, results_fname, timing_fname in METHODS:
        subdir = os.path.join(a.results_dir, SUBDIRS[key])
        rpath = os.path.join(subdir, results_fname)
        tpath = os.path.join(subdir, timing_fname)
        if not os.path.exists(rpath):
            print(f"[MISSING] {label}: {rpath}")
            continue
        res = json.load(open(rpath))
        best_kid, best_recall, best_dup = _best_over_grid(res["grid"])
        timing = json.load(open(tpath)) if os.path.exists(tpath) else {}
        table[key] = {
            "label": label,
            "kid": best_kid["kid_mean"], "kid_std": best_kid["kid_std"],
            "kid_cell": {k: v for k, v in best_kid.items()
                        if k not in ("kid_vals", "recall_vals", "duprate_vals")},
            "recall": best_recall["recall_mean"], "recall_std": best_recall["recall_std"],
            "recall_cell": {k: v for k, v in best_recall.items()
                           if k not in ("kid_vals", "recall_vals", "duprate_vals")},
            "duprate": best_dup["duprate_mean"], "duprate_std": best_dup["duprate_std"],
            "duprate_cell": {k: v for k, v in best_dup.items()
                             if k not in ("kid_vals", "recall_vals", "duprate_vals")},
            "timing": timing,
        }

    ddpm_path = os.path.join(a.results_dir, "ddpm", "results_ddpm.json")
    if os.path.exists(ddpm_path):
        ddpm_res = json.load(open(ddpm_path))
        ddpm_timing = json.load(open(os.path.join(a.results_dir, "ddpm", "timing_ddpm.json")))
        table["latent_ddpm"] = {
            "label": "Latent DDPM",
            "kid": ddpm_res["kid_mean"], "kid_std": ddpm_res["kid_std"],
            "recall": ddpm_res["recall_mean"], "recall_std": ddpm_res["recall_std"],
            "duprate": ddpm_res["duprate_mean"], "duprate_std": ddpm_res["duprate_std"],
            "timing": ddpm_timing,
        }
    else:
        print(f"[MISSING] Latent DDPM: {ddpm_path}")

    # ── Print digit-8 generation results table ────────────────────────────────────────────────────
    order = ["mmsold", "cfdm", "knn_pflow", "swf", "plugin_baoab", "sg_baoab", "latent_ddpm"]
    print("\n" + "=" * 100)
    print("digit-8 generation results table: Generation quality and computational cost on handwritten digit-8")
    print("=" * 100)
    header = f"{'Metric':<12}" + "".join(f"{table[k]['label']:>16}" for k in order if k in table)
    print(header)

    def _fmt_row(name, getter):
        row = f"{name:<12}"
        for k in order:
            if k not in table:
                continue
            v = getter(table[k])
            row += f"{v:>16}"
        print(row)

    _fmt_row("KID", lambda t: f"{t['kid']:.4f}±{t['kid_std']:.4f}")
    _fmt_row("Recall", lambda t: f"{t['recall']:.3f}±{t['recall_std']:.3f}")
    _fmt_row("DupRate", lambda t: f"{t['duprate']:.3f}±{t['duprate_std']:.3f}")

    def _train_opt(t):
        ti = t.get("timing", {})
        if "train_opt_time_h" in ti:
            return f"{ti['train_opt_time_h']:.2f}h"
        if "training_time_h" in ti:
            return f"{ti['training_time_h']:.2f}h"
        if "opt_time_s" in ti and ti["opt_time_s"] is not None:
            return f"{ti['opt_time_s']/3600.0:.3f}h"
        return "n/a"

    def _sample_time(t):
        ti = t.get("timing", {})
        v = ti.get("sample_time_s")
        return f"{v:.3f}s" if v is not None else "n/a"

    def _device(t):
        ti = t.get("timing", {})
        return ti.get("device") or ti.get("sample_device") or "n/a"

    _fmt_row("Train/opt", _train_opt)
    _fmt_row("Sample time", _sample_time)
    _fmt_row("Device", _device)

    out = {"table": table}
    out_path = os.path.join(a.results_dir, "table1.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
