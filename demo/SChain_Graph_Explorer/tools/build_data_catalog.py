import json
from pathlib import Path


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    filenames = sorted(
        path.name
        for path in DATA_DIRECTORY.glob("*.json")
        if path.name != "catalog.json"
    )
    catalog_path = DATA_DIRECTORY / "catalog.json"
    catalog_path.write_text(
        json.dumps({"files": filenames}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Indexed {len(filenames)} JSON file(s)")


if __name__ == "__main__":
    main()
