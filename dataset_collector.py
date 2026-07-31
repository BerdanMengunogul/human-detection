"""Passive collection of identity-labeled person crops for ReID training.

Saves crops already produced by the live identification/top-up pipeline to
<DATASET_PATH>/<person_id>/<timestamp>.jpg. Purely additive: never influences
matching, thresholds, or track state.
"""

import os
import time


class DatasetCollector:
    def __init__(self, cfg):
        self.enabled = cfg.DATASET_COLLECTION_ENABLED
        self.root = cfg.DATASET_PATH
        self.max_samples_per_person = cfg.DATASET_MAX_SAMPLES_PER_PERSON
        self.min_save_interval = cfg.DATASET_MIN_SAVE_INTERVAL_SECONDS
        self._last_save_time = {}
        self._sample_count = {}

        if self.enabled:
            os.makedirs(self.root, exist_ok=True)
            for name in os.listdir(self.root):
                person_dir = os.path.join(self.root, name)
                if os.path.isdir(person_dir):
                    self._sample_count[name] = len(os.listdir(person_dir))

    def save(self, person_id, crop):
        """Save a BGR crop for person_id, subject to rate limit and per-person cap."""
        if not self.enabled or person_id is None or crop is None or crop.size == 0:
            return

        now = time.monotonic()
        last = self._last_save_time.get(person_id, 0.0)
        if now - last < self.min_save_interval:
            return

        key = str(person_id)
        if self._sample_count.get(key, 0) >= self.max_samples_per_person:
            return

        person_dir = os.path.join(self.root, key)
        os.makedirs(person_dir, exist_ok=True)

        import cv2

        filename = f"{time.time():.6f}.jpg"
        path = os.path.join(person_dir, filename)
        if cv2.imwrite(path, crop):
            self._last_save_time[person_id] = now
            self._sample_count[key] = self._sample_count.get(key, 0) + 1
