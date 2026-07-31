"""Calibrate REID_* similarity thresholds in config.py/config.yaml against the
actual embedding space of the currently-configured REID_MODEL_NAME.

Swapping in a fine-tuned ReID model shifts the whole cosine-similarity
distribution, so thresholds picked for the previous model are not valid for
the new one. This script embeds every crop under DATASET_PATH (the same
per-person folders dataset_collector.py produces and train_reid.py trains
on) using the exact same encoder/normalization identity.py uses live, then
reports the same-person vs different-person similarity distributions and
suggests threshold values from where they separate.

Usage:
    python calibrate_reid_thresholds.py --data dataset
"""

import argparse
import os
import random
from collections import defaultdict

import cv2
import numpy as np

from identity import load_reid_encoder


def embed_dataset(encoder, root, max_per_person=60, seed=0):
    rng = random.Random(seed)
    embeddings_by_person = defaultdict(list)

    for person_id in sorted(os.listdir(root)):
        person_dir = os.path.join(root, person_id)
        if not os.path.isdir(person_dir):
            continue
        files = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        rng.shuffle(files)
        files = files[:max_per_person]

        for fname in files:
            img = cv2.imread(os.path.join(person_dir, fname))
            if img is None:
                continue
            h, w = img.shape[:2]
            xywh = np.array([w / 2, h / 2, w, h], dtype=np.float32)
            emb = encoder(img, xywh[None, :])[0]
            emb = emb / np.linalg.norm(emb)
            embeddings_by_person[person_id].append(emb)

    return embeddings_by_person


def pairwise_similarities(embeddings_by_person):
    same_sims, diff_sims = [], []
    people = list(embeddings_by_person.items())

    for person_id, embs in people:
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                same_sims.append(float(np.dot(embs[i], embs[j])))

    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            for e1 in people[i][1]:
                for e2 in people[j][1]:
                    diff_sims.append(float(np.dot(e1, e2)))

    return np.array(same_sims), np.array(diff_sims)


def summarize(name, arr):
    if arr.size == 0:
        print(f"[{name}] no pairs")
        return
    pct = np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print(
        f"[{name}] n={arr.size} mean={arr.mean():.3f} std={arr.std():.3f} "
        f"p1={pct[0]:.3f} p5={pct[1]:.3f} p10={pct[2]:.3f} p25={pct[3]:.3f} "
        f"p50={pct[4]:.3f} p75={pct[5]:.3f} p90={pct[6]:.3f} p95={pct[7]:.3f} p99={pct[8]:.3f}"
    )


def suggest_thresholds(same_sims, diff_sims):
    """Pick thresholds from the separation between the two distributions:
    REID_MATCH_THRESHOLD at the 5th percentile of same-person similarity
    (accept the vast majority of true matches), REID_HIGH_CONF_THRESHOLD at
    its 50th percentile (typical true-match confidence), and
    REID_BORDERLINE_THRESHOLD/REID_CORROBORATION_THRESHOLD stepped below the
    match threshold toward the different-person distribution's upper tail.

    Ordering REID_CORROBORATION < REID_BORDERLINE < REID_MATCH < REID_HIGH_CONF
    is enforced by construction: each threshold is derived from the one above
    it and stepped down by a fixed margin, rather than independently computed
    from diff_p95 and then clamped after the fact (the previous approach could
    leave corroboration above borderline/match when the same/diff distributions
    overlap heavily, since diff_p95 can exceed match_thr in that case)."""
    match_thr = float(np.percentile(same_sims, 5))
    high_conf_thr = float(np.percentile(same_sims, 50))
    diff_p95 = float(np.percentile(diff_sims, 95)) if diff_sims.size else match_thr - 0.1

    if diff_p95 >= match_thr:
        print(
            f"[WARN] Different-person p95 similarity ({diff_p95:.3f}) is >= "
            f"same-person match threshold ({match_thr:.3f}): the embedding space "
            "does not separate identities well. More/better training data or "
            "further fine-tuning is likely needed - thresholds alone cannot "
            "fully fix this."
        )

    margin = 0.01
    borderline_thr = min(diff_p95, match_thr - margin)
    corroboration_thr = min(diff_p95 - margin, borderline_thr - margin)
    high_conf_thr = max(high_conf_thr, match_thr + margin)

    return {
        "REID_MATCH_THRESHOLD": round(match_thr, 3),
        "REID_BORDERLINE_THRESHOLD": round(borderline_thr, 3),
        "REID_CORROBORATION_THRESHOLD": round(corroboration_thr, 3),
        "REID_HIGH_CONF_THRESHOLD": round(high_conf_thr, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset", help="Root dataset dir (default: dataset)")
    parser.add_argument("--max-per-person", type=int, default=60, help="Cap crops sampled per identity")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.isdir(args.data):
        raise SystemExit(f"Dataset dir not found: {args.data}")

    print("[INFO] Loading REID encoder (uses REID_MODEL_NAME from config)...")
    encoder = load_reid_encoder()

    print(f"[INFO] Embedding crops under {args.data} ...")
    embeddings_by_person = embed_dataset(encoder, args.data, args.max_per_person, args.seed)
    people_with_data = {k: v for k, v in embeddings_by_person.items() if len(v) >= 2}
    print(f"[INFO] {len(people_with_data)} identities with >= 2 samples")
    if len(people_with_data) < 2:
        raise SystemExit("Need at least 2 identities with >= 2 samples each to calibrate.")

    same_sims, diff_sims = pairwise_similarities(people_with_data)
    summarize("SAME-PERSON", same_sims)
    summarize("DIFF-PERSON", diff_sims)

    suggested = suggest_thresholds(same_sims, diff_sims)
    print("\n[SUGGESTED] Paste into config.yaml (values are cosine similarity, 0-1):")
    for key, value in suggested.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
