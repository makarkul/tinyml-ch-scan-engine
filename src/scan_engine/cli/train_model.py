from __future__ import annotations

from ._config import load_config


def main() -> None:
    cfg = load_config()
    print(f"[train_model] stub for config={cfg['name']} model={cfg['model']['name']}; "
          "PyTorch wiring lands in M3.")


if __name__ == "__main__":
    main()
