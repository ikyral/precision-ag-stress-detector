"""Reproduce both supplied examples; works from any current directory."""
from pathlib import Path
import sys

MODULE = Path(__file__).resolve().parents[1]
ROOT = MODULE.parents[1]
sys.path.insert(0, str(MODULE))
from vra import generate_prescription


def main():
    for name, folder in (("sample_zones.tif", "sample"), ("sample_zones_m2.tif", "module2")):
        paths = generate_prescription(ROOT / "data" / "samples" / name,
                                      ROOT / "data" / "module3_demo" / folder, "2026_08_07")
        print(name)
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
