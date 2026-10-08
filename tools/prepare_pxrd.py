#!/usr/bin/env python3
"""Prepare experimental two-column PXRD text for classification (SI-3)."""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.pxrd import preprocess_experimental


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Whitespace-separated 2theta/intensity text; # comments allowed.")
    parser.add_argument("--output", required=True, type=Path, help="Output .npy profile.")
    args = parser.parse_args()
    data = np.loadtxt(args.input, ndmin=2)
    if data.shape[1] != 2:
        raise ValueError("input must have exactly two columns: 2theta and intensity")
    result = preprocess_experimental(data[:, 0], data[:, 1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result, allow_pickle=False)


if __name__ == "__main__":
    main()
