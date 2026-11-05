"""Assemble local assets from the single canonical graph explorer implementation."""
import argparse
import shutil
from pathlib import Path

from schain.config import REPO_ROOT
from schain.viewer import write_viewer


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO_ROOT / "visualization/index.html", output / "index.html")
    write_viewer({"nodes": [], "edges": [], "documents": []}, output / "explorer.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
