from __future__ import annotations

from ..data.generator import iter_dataset
from ._config import load_config


def main() -> None:
    cfg = load_config()
    counts = {s: sum(1 for _ in iter_dataset(cfg, s)) for s in ("train", "val", "test")}
    print(f"[gen_dataset] {cfg['name']}: {counts}")


if __name__ == "__main__":
    main()
