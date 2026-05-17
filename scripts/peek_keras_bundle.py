"""Print the embedded config.json + metadata.json from a *.keras bundle."""
from __future__ import annotations

import json
import sys
import zipfile


def main(path: str) -> int:
    with zipfile.ZipFile(path, "r") as zf:
        for name in ("metadata.json", "config.json"):
            try:
                with zf.open(name) as fp:
                    data = json.loads(fp.read().decode("utf-8"))
                print(f"\n=== {path}::{name} ===")
                print(json.dumps(data, indent=2)[:4000])
            except KeyError:
                print(f"\n=== {path}::{name} (missing) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
