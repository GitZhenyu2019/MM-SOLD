"""
step0_prepare_subset.py
=======================
Prepare the CelebA-HQ-256 data subset for the experiment.
"""
import argparse
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_DIR, VALID_DIR, TRAIN_USE_DIR, VALID_USE_DIR,
    N_TRAIN, N_TEST,
)


def _collect_pngs(folder: str) -> list:
    """Return numerically sorted list of PNG paths in folder."""
    entries = sorted(
        (f for f in os.listdir(folder) if f.lower().endswith(".png")),
        key=lambda f: int(re.search(r'\d+', f).group()),
    )
    return [os.path.join(folder, f) for f in entries]


def _copy_or_link(src_paths: list, dst_dir: str, use_symlink: bool):
    os.makedirs(dst_dir, exist_ok=True)
    for src in src_paths:
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst) or os.path.islink(dst):
            continue
        if use_symlink:
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)


def main(args):
    print("=" * 60)
    print("Step 0: Prepare CelebA-HQ-256 subset")
    print("=" * 60)

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"\nScanning training PNGs in: {args.train_dir}")
    train_all = _collect_pngs(args.train_dir)
    print(f"  Found {len(train_all)} training PNGs")
    if len(train_all) < args.n_train:
        raise ValueError(
            f"Only {len(train_all)} training PNGs available, need {args.n_train}")
    train_sel = train_all[:args.n_train]

    print(f"  Copying first {args.n_train} → {args.train_use}")
    _copy_or_link(train_sel, args.train_use, args.symlink)
    print(f"  Done: {args.train_use}/  ({args.n_train} files)")

    # ── Valid ────────────────────────────────────────────────────────────────
    print(f"\nScanning validation PNGs in: {args.valid_dir}")
    valid_all = _collect_pngs(args.valid_dir)
    print(f"  Found {len(valid_all)} validation PNGs")
    if len(valid_all) < args.n_test:
        raise ValueError(
            f"Only {len(valid_all)} validation PNGs available, need {args.n_test}")
    valid_sel = valid_all[:args.n_test]

    print(f"  Copying first {args.n_test} → {args.valid_use}")
    _copy_or_link(valid_sel, args.valid_use, args.symlink)
    print(f"  Done: {args.valid_use}/  ({args.n_test} files)")

    print("\nStep 0 complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 0: Prepare CelebA-HQ-256 data subset.")
    parser.add_argument("--train_dir",  type=str, default=TRAIN_DIR)
    parser.add_argument("--valid_dir",  type=str, default=VALID_DIR)
    parser.add_argument("--train_use",  type=str, default=TRAIN_USE_DIR)
    parser.add_argument("--valid_use",  type=str, default=VALID_USE_DIR)
    parser.add_argument("--n_train",    type=int, default=N_TRAIN)
    parser.add_argument("--n_test",     type=int, default=N_TEST)
    parser.add_argument("--symlink",    action="store_true",
                        help="Create symlinks instead of copying (saves disk space)")
    args = parser.parse_args()
    main(args)
