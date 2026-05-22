"""Voxel-wise Dice comparison between two label NIfTI files."""
import argparse
import sys

import nibabel as nib
import numpy as np


def compare(ref_path: str, new_path: str) -> int:
    ref = nib.load(ref_path).get_fdata().astype(np.uint8)
    new = nib.load(new_path).get_fdata().astype(np.uint8)
    if ref.shape != new.shape:
        print(f"shape mismatch: {ref.shape} vs {new.shape}", file=sys.stderr)
        return 2

    labels = sorted(set(np.unique(ref).tolist()) | set(np.unique(new).tolist()))
    print(f'  {"lbl":>4} {"ref":>10} {"new":>10}  Dice')
    worst = 1.0
    for lab in labels:
        r = ref == lab
        n = new == lab
        s = int(r.sum()) + int(n.sum())
        dice = 2 * int((r & n).sum()) / s if s > 0 else 1.0
        worst = min(worst, dice)
        print(f"  {lab:>4} {int(r.sum()):>10} {int(n.sum()):>10}  {dice:.4f}")

    agree = int((ref == new).sum())
    total = ref.size
    print(f"\nvoxel agreement: {agree}/{total} = {agree/total*100:.4f}%")
    print(f"worst per-class Dice: {worst:.4f}")
    return 0 if worst >= 0.99 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()
    sys.exit(compare(args.ref, args.new))


if __name__ == "__main__":
    main()
