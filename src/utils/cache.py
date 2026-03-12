"""Disk-based cache for predictions so re-runs skip already-computed markets."""

import json
from pathlib import Path


class PredictionCache:
    """Simple file-backed cache keyed by (model_name, market_id)."""

    def __init__(self, cache_dir: str = "data/predictions"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, model_name: str) -> Path:
        safe_name = model_name.replace("/", "_")
        return self.cache_dir / f"{safe_name}.jsonl"

    def load_existing(self, model_name: str) -> dict[str, dict]:
        """Load existing predictions as {market_id: prediction_dict}."""
        path = self._path(model_name)
        if not path.exists():
            return {}
        existing = {}
        for line in path.read_text().strip().split("\n"):
            if line:
                pred = json.loads(line)
                existing[pred["market_id"]] = pred
        return existing

    def append(self, model_name: str, prediction: dict) -> None:
        """Append a single prediction to the cache file."""
        path = self._path(model_name)
        with open(path, "a") as f:
            f.write(json.dumps(prediction) + "\n")

    def save_all(self, model_name: str, predictions: list[dict]) -> None:
        """Overwrite cache with full prediction list."""
        path = self._path(model_name)
        with open(path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
