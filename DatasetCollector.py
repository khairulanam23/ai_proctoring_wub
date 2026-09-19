#!/usr/bin/env python3
"""DatasetCollector.exe — Portable Standalone Image Dataset Collection Application.

Windows / Linux Double-Clickable Launcher for AI Proctoring Dataset Collection.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path


def setup_application_logging() -> Path:
    """Configures application-level file and console logging."""
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).resolve().parent
    else:
        base_dir = Path(__file__).resolve().parent

    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "collector.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(str(log_file), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("=" * 60)
    logging.info("DatasetCollector application starting...")
    logging.info(f"Python Platform: {sys.platform}")
    logging.info(f"Base Directory : {base_dir}")
    logging.info("=" * 60)
    return log_file


def main() -> int:
    log_file = setup_application_logging()

    try:
        from tools.dataset_collector.collector_app import main as app_main

        return app_main()
    except Exception as e:
        logging.error(f"Fatal error running DatasetCollector: {e}\n{traceback.format_exc()}")
        print("\n" + "!" * 60)
        print("  APPLICATION ERROR")
        print("!" * 60)
        print(f"An unexpected error occurred: {e}")
        print(f"\nA technical log has been recorded to:\n  {log_file}")
        print("\nPlease contact your dataset administrator with this log file.")
        print("!" * 60)
        if sys.platform.startswith("win"):
            # On Windows, keep window open so user can read message
            input("\nPress Enter to exit...")
        return 1


if __name__ == "__main__":
    sys.exit(main())
