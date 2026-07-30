from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.baselines import make_baseline_model
from diffgbm import DiffGBM


@dataclass(frozen=True)
class Variant:
    name: str
    params: dict[str, Any]
    model: str = "diffgbm"

    def make_model(self, seed: int) -> DiffGBM:
        if self.model != "diffgbm":
            return make_baseline_model(model_type=self.model, params=self.params, seed=seed)
        params = dict(self.params)
        params["seed"] = seed
        return DiffGBM(**params)


def make_variants(config: list[dict[str, Any]]) -> list[Variant]:
    variants = []
    for item in config:
        if not item.get("enabled", True):
            continue
        name = item["name"]
        model = item.get("model", "diffgbm")
        params = item.get("params", {})
        variants.append(Variant(name=name, params=params, model=model))
    if not variants:
        raise ValueError("Benchmark config must enable at least one variant.")
    return variants
