from __future__ import annotations

from ._config import load_config


def main() -> None:
    cfg = load_config()
    print(f"[eval_all] stub for {cfg['name']}; full benchmark harness lands in M3/M5.")


if __name__ == "__main__":
    main()
