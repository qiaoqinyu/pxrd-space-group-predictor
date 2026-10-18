#!/usr/bin/env python3
"""Deduplicate relaxed CIFs within space groups and export DTW input arrays."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime.crystal_candidates import read_candidates, write_candidates, sha256
from src.runtime.crystal_processing import deduplicate, simulate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    rows = read_candidates(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reps, mapping = deduplicate(rows)
    write_candidates(args.output_dir / "representatives.csv", reps)
    with (args.output_dir / "deduplication.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    profiles = np.stack([simulate(Path(r["cif_path"])) for r in reps])
    np.save(args.output_dir / "candidates_raw.npy", profiles, allow_pickle=False)
    np.save(args.output_dir / "energies.npy", np.array([float(r["energy_per_molecule"]) for r in reps]), allow_pickle=False)
    np.save(args.output_dir / "ids.npy", np.array([r["candidate_id"] for r in reps]), allow_pickle=False)
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "input_manifest_sha256": sha256(args.manifest), "input_count": len(rows),
        "representative_count": len(reps), "energy_unit": "kJ/mol per molecule",
        "step_limit_count": sum(r["status"] == "step_limit" for r in reps),
        "protocol": "docs/downstream-workflow.md"}, indent=2) + "\n")
    print(f"Retained {len(reps)} of {len(rows)} candidates; wrote DTW input arrays.")


if __name__ == "__main__":
    main()
