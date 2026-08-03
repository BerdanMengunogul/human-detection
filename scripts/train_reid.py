"""Offline fine-tuning of a person-ReID embedder on crops collected by
dataset_collector.py, exported to ONNX as a drop-in replacement for
REID_MODEL_NAME.

Usage:
    python train_reid.py --data dataset --out reid_finetuned.onnx

Starts from torchreid's ImageNet+ReID-pretrained OSNet and fine-tunes it with
a combined softmax + batch-hard triplet loss (the standard ReID metric-learning
recipe) over the collected person_id folders: cross-entropy pulls embeddings
toward their identity's classifier direction, while the triplet term directly
pulls same-identity embeddings together and pushes different identities apart
in embedding space. Batches are built via PKSampler so every batch contains
multiple samples per identity, as batch-hard mining requires. Validation uses
1-NN cosine-similarity accuracy against a per-identity mean-embedding gallery
built from the train split, mirroring identity.py's live matching approach.
The classifier head is only used during training - at export time the model
is switched to eval mode, which yields the plain embedding tensor, so the
ONNX export plugs into the same (img, dets) -> embedding interface
identity.py already expects.
"""

import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

import torchreid

MIN_PERSONS = 4
MIN_SAMPLES_PER_PERSON = 20
# Ultralytics' ReID wrapper (reid.py) only supports square ONNX input shapes -
# it auto-detects a single imgsz from the model's height dim and always builds
# a square crop tensor, so a rectangular export (e.g. the conventional 256x128
# person-ReID shape) causes a shape-mismatch at inference. Must stay square.
INPUT_HEIGHT, INPUT_WIDTH = 256, 256


class PersonCropDataset(Dataset):
    def __init__(self, samples, label_map, transform):
        self.samples = samples
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, person_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.label_map[person_id]


class PKSampler(Sampler):
    """Yields batches of P identities x K samples each, as required by
    batch-hard triplet loss (every batch needs multiple samples per identity
    so each anchor has both a valid positive and negative)."""

    def __init__(self, samples, num_instances, batch_size, seed=0):
        self.num_instances = num_instances
        self.rng = random.Random(seed)

        self.index_by_pid = defaultdict(list)
        for idx, (_, person_id) in enumerate(samples):
            self.index_by_pid[person_id].append(idx)

        self.pids = list(self.index_by_pid.keys())
        # Clamp to the number of identities actually available - with
        # batch_size // num_instances > len(pids) (e.g. the common case of
        # having fewer than 8 identities with the 32/4 defaults), the batch
        # would never fill and DataLoader would silently yield zero batches
        # every epoch (train_loss/train_acc stuck at 0, no weight updates).
        self.num_pids_per_batch = max(1, min(batch_size // num_instances, len(self.pids)))
        self.num_batches = len(self.pids) // self.num_pids_per_batch
        self.length = self.num_batches * self.num_pids_per_batch * self.num_instances

    def __len__(self):
        return self.length

    def __iter__(self):
        pid_pool = defaultdict(list)
        for pid, indices in self.index_by_pid.items():
            idxs = indices[:]
            self.rng.shuffle(idxs)
            pid_pool[pid] = idxs

        avail_pids = [pid for pid in self.pids if len(pid_pool[pid]) > 0]
        self.rng.shuffle(avail_pids)

        batches = []
        while len(batches) < self.num_batches and len(avail_pids) >= self.num_pids_per_batch:
            selected = avail_pids[: self.num_pids_per_batch]
            batch = []
            for pid in selected:
                pool = pid_pool[pid]
                take = pool[: self.num_instances]
                if len(take) < self.num_instances:
                    take = take + self.rng.choices(take, k=self.num_instances - len(take))
                del pid_pool[pid][: self.num_instances]
                batch.extend(take)
            batches.append(batch)
            avail_pids = [pid for pid in avail_pids[self.num_pids_per_batch:] if len(pid_pool[pid]) > 0] + [
                pid for pid in selected if len(pid_pool[pid]) > 0
            ]
            self.rng.shuffle(avail_pids)

        for batch in batches:
            yield from batch


def scan_dataset(root):
    """Return {person_id: [file_paths]} for subdirectories with image files."""
    people = {}
    for name in sorted(os.listdir(root)):
        person_dir = os.path.join(root, name)
        if not os.path.isdir(person_dir):
            continue
        files = [
            os.path.join(person_dir, f)
            for f in sorted(os.listdir(person_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if files:
            people[name] = files
    return people


def split_train_val(people, val_fraction=0.2, seed=0):
    rng = random.Random(seed)
    train_samples, val_samples = [], []
    for person_id, files in people.items():
        files = files[:]
        rng.shuffle(files)
        n_val = max(1, int(len(files) * val_fraction))
        val_samples += [(f, person_id) for f in files[:n_val]]
        train_samples += [(f, person_id) for f in files[n_val:]]
    return train_samples, val_samples


def build_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, eval_tf


def train(args):
    people = scan_dataset(args.data)
    too_few = {pid: len(files) for pid, files in people.items() if len(files) < MIN_SAMPLES_PER_PERSON}
    usable = {pid: files for pid, files in people.items() if len(files) >= MIN_SAMPLES_PER_PERSON}

    if too_few:
        print(f"[WARN] Skipping {len(too_few)} person(s) with < {MIN_SAMPLES_PER_PERSON} samples: {too_few}")

    if len(usable) < MIN_PERSONS:
        print(
            f"[ERROR] Only {len(usable)} person(s) have >= {MIN_SAMPLES_PER_PERSON} samples; "
            f"need at least {MIN_PERSONS} to fine-tune. Collect more data first."
        )
        sys.exit(1)

    label_map = {pid: i for i, pid in enumerate(sorted(usable.keys()))}
    print(f"[INFO] Training on {len(usable)} identities: {dict((k, len(v)) for k, v in usable.items())}")

    train_samples, val_samples = split_train_val(usable, val_fraction=args.val_fraction, seed=args.seed)
    train_tf, eval_tf = build_transforms()
    train_ds = PersonCropDataset(train_samples, label_map, train_tf)
    val_ds = PersonCropDataset(val_samples, label_map, eval_tf)

    train_sampler = PKSampler(train_samples, args.num_instances, args.batch_size, seed=args.seed)
    actual_batch_size = train_sampler.num_pids_per_batch * args.num_instances
    if actual_batch_size != args.batch_size:
        print(
            f"[WARN] Only {len(train_sampler.pids)} identities available; using "
            f"batch_size={actual_batch_size} ({train_sampler.num_pids_per_batch} identities x "
            f"{args.num_instances} samples) instead of --batch-size {args.batch_size}."
        )
    # batch_size must match the PxK grouping PKSampler.__iter__ actually yields -
    # DataLoader's default batching just chunks the sampler's flat index stream
    # by batch_size, so a mismatched size would split/merge PxK groups and break
    # the "every batch has K samples per identity" guarantee batch-hard mining needs.
    train_loader = DataLoader(train_ds, batch_size=actual_batch_size, sampler=train_sampler, drop_last=True)
    # Separate sequential pass over the same train samples (with eval-time
    # transforms) used only to build the per-identity gallery for validation.
    train_eval_ds = PersonCropDataset(train_samples, label_map, eval_tf)
    train_loader_eval = DataLoader(train_eval_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchreid.models.build_model(
        args.backbone, num_classes=len(label_map), loss="triplet", pretrained=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    triplet_loss = torchreid.losses.TripletLoss(margin=args.triplet_margin)

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, embeddings = model(imgs)
            loss = ce_loss(logits, labels) + args.triplet_weight * triplet_loss(embeddings, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += imgs.size(0)
        scheduler.step()

        train_loss = total_loss / max(total_n, 1)
        train_acc = total_correct / max(total_n, 1)

        val_acc = evaluate_embedding_accuracy(model, train_loader_eval, val_loader, device)

        print(
            f"[EPOCH {epoch}/{args.epochs}] train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"[INFO] Best val_acc={best_val_acc:.3f}")
    model.load_state_dict(best_state)
    export_onnx(model, args.out, device)


def evaluate_embedding_accuracy(model, gallery_loader, query_loader, device):
    """1-NN cosine-similarity accuracy of query embeddings against a
    per-identity mean-embedding gallery built from the train split. Mirrors
    how identity.py's IdentityGallery matches live embeddings."""
    model.eval()
    gallery_sums, gallery_counts = {}, {}
    with torch.no_grad():
        for imgs, labels in gallery_loader:
            embeddings = torch.nn.functional.normalize(model(imgs.to(device)), dim=1)
            for emb, label in zip(embeddings, labels):
                label = label.item()
                gallery_sums[label] = gallery_sums.get(label, 0) + emb
                gallery_counts[label] = gallery_counts.get(label, 0) + 1

        gallery_labels = sorted(gallery_sums.keys())
        gallery = torch.stack(
            [torch.nn.functional.normalize(gallery_sums[l] / gallery_counts[l], dim=0) for l in gallery_labels]
        )

        correct, total = 0, 0
        for imgs, labels in query_loader:
            embeddings = torch.nn.functional.normalize(model(imgs.to(device)), dim=1)
            sims = embeddings @ gallery.T
            preds = [gallery_labels[i] for i in sims.argmax(dim=1).tolist()]
            correct += sum(p == l.item() for p, l in zip(preds, labels))
            total += len(labels)

    return correct / max(total, 1)


def export_onnx(model, out_path, device):
    # torchreid triplet-loss models return the pooled 512-d feature vector
    # directly in eval() mode (the classifier head is only used in training).
    model.eval()
    dummy = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH, device=device)
    with torch.no_grad():
        sample_out = model(dummy)
    print(f"[INFO] Exported embedding dim: {sample_out.shape[-1]}")

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"[INFO] Saved ONNX embedder to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset", help="Root dataset dir (default: dataset)")
    parser.add_argument("--out", default="reid_finetuned.onnx", help="Output ONNX path")
    parser.add_argument("--backbone", default="osnet_x1_0", help="torchreid backbone name")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-instances", type=int, default=4,
        help="K: samples per identity per batch, required by batch-hard triplet mining (default: 4)",
    )
    parser.add_argument(
        "--triplet-margin", type=float, default=0.3,
        help="Margin for the batch-hard triplet loss (default: 0.3)",
    )
    parser.add_argument(
        "--triplet-weight", type=float, default=1.0,
        help="Weight applied to the triplet loss term in the combined softmax+triplet loss (default: 1.0)",
    )
    args = parser.parse_args()

    if args.batch_size % args.num_instances != 0:
        raise ValueError("--batch-size must be divisible by --num-instances")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
