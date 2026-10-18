#!/usr/bin/env python3
"""Optional licensed CCDC validation; results never enter candidate ranking."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime.crystal_candidates import read_candidates, sha256


def compare(reference: Path, candidate: Path) -> dict:
    from ccdc.crystal import PackingSimilarity
    from ccdc.io import CrystalReader
    engine = PackingSimilarity()
    settings = engine.settings
    settings.packing_shell_size = 15
    settings.distance_tolerance = 0.2
    settings.angle_tolerance = 20
    settings.allow_molecular_differences = False
    settings.allow_artificial_inversion = True
    settings.ignore_hydrogen_positions = True
    settings.ignore_hydrogen_counts = True
    settings.ignore_bond_counts = True
    ref = CrystalReader(str(reference))[0]
    crystal = CrystalReader(str(candidate))[0]
    result = engine.compare(crystal, ref)
    nmatched = int(result.nmatched_molecules) if result is not None else 0
    return {"status": "matched" if nmatched >= 15 else "no_match", "nmatched": nmatched,
            "rmsd": float(result.rmsd) if nmatched else None, "hit": nmatched >= 15}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--reference", type=Path, required=True, help="Unoptimized experimental reference CIF.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=30, help="Seconds per candidate.")
    p.add_argument("--_candidate", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args._candidate:
        try:
            result = compare(args.reference, args._candidate)
        except Exception as exc:
            result = {"status": f"error:{type(exc).__name__}", "nmatched": None, "rmsd": None, "hit": None}
        args.output_dir.write_text(json.dumps(result))
        return
    if not args.manifest or not args.reference.is_file() or args.timeout <= 0:
        p.error("valid manifest, reference CIF and positive timeout are required")
    # Fail before creating results when the API is absent. License/runtime errors remain explicit.
    from ccdc.crystal import PackingSimilarity
    PackingSimilarity()
    rows = read_candidates(args.manifest)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for index, row in enumerate(rows):
        result_path = output / f"result_{index:08d}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--reference", str(args.reference.resolve()),
                   "--_candidate", row["cif_path"], "--output-dir", str(result_path)]
        with (output / f"candidate_{index:08d}.log").open("x") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                result = json.loads(result_path.read_text()) if process.returncode == 0 and result_path.is_file() else {
                    "status": "process_error", "nmatched": None, "rmsd": None, "hit": None}
            except subprocess.TimeoutExpired:
                result = {"status": "timeout", "nmatched": None, "rmsd": None, "hit": None}
        results.append(dict(candidate_id=row["candidate_id"], **result))
    with (output / "packing.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output / "provenance.json").write_text(json.dumps({
        "manifest_sha256": sha256(args.manifest), "reference_sha256": sha256(args.reference),
        "purpose": "post-ranking validation only", "shell_molecules": 15,
        "distance_tolerance": 0.2, "angle_tolerance": 20, "allow_artificial_inversion": True}, indent=2) + "\n")
    failures = sum(r["status"] not in {"matched", "no_match"} for r in results)
    print(f"Validated {len(results)} candidates; {failures} unresolved errors/timeouts (not negative matches).")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
