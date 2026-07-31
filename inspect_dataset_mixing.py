"""One-off: build a contact-sheet image per dataset/<Name>/ folder, grouping
crops by their idN_/plain sub-batch (the tracker person_id they were
originally collected under, before folders were renamed to real names), so
mixed-identity folders can be spotted and split by eye before retraining.

Usage:
    python inspect_dataset_mixing.py --data dataset --out contact_sheets
"""

import argparse
import os
import re
from collections import defaultdict

import cv2
import numpy as np

THUMB = 96
COLS = 10


def group_files(person_dir):
    files = sorted(f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    groups = defaultdict(list)
    for f in files:
        m = re.match(r"id(\d+)_", f)
        key = f"id{m.group(1)}" if m else "plain"
        groups[key].append(f)
    return groups


def make_sheet(person_dir, groups, out_path):
    rows = []
    for key in sorted(groups, key=lambda k: (k == "plain", k)):
        sample = groups[key][:: max(1, len(groups[key]) // COLS)][:COLS]
        thumbs = []
        for fname in sample:
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue
            img = cv2.resize(img, (THUMB, THUMB))
            thumbs.append(img)
        while len(thumbs) < COLS:
            thumbs.append(np.zeros((THUMB, THUMB, 3), dtype=np.uint8))
        row = np.hstack(thumbs)
        label = np.zeros((20, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(label, f"{key} (n={len(groups[key])})", (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        rows.append(np.vstack([label, row]))
    sheet = np.vstack(rows)
    cv2.imwrite(out_path, sheet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--out", default="contact_sheets")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for person in sorted(os.listdir(args.data)):
        person_dir = os.path.join(args.data, person)
        if not os.path.isdir(person_dir):
            continue
        groups = group_files(person_dir)
        out_path = os.path.join(args.out, f"{person}.jpg")
        make_sheet(person_dir, groups, out_path)
        print(f"[INFO] {person}: {len(groups)} sub-batches -> {out_path}")


if __name__ == "__main__":
    main()
