#!/usr/bin/env python3
"""Run the two-stage paper MACE protocol using an isolated pinned BOMLIP snapshot."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.runtime.crystal_candidates import read_candidates, write_candidates, sha256
from src.runtime.downstream_execution import snapshot, pin_mace_model, run_logged, record, BOMLIP_COMMIT


def molecular_atom_count(manifest: Path, metadata: Path, sample: str, asserted: int | None = None) -> int:
    generation = json.loads(metadata.read_text())
    count = generation["atoms_per_molecule"]
    if type(count) is not int or count < 1 or generation["sample_id"] != sample or generation["manifest_sha256"] != sha256(manifest):
        raise ValueError("generation metadata does not match manifest or molecular atom count")
    if asserted is not None and asserted != count:
        raise ValueError("atoms-per-molecule differs from generation metadata")
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--atoms-per-molecule", type=int, help="Optional assertion against generation metadata; includes hydrogens.")
    p.add_argument("--generation-metadata", type=Path, help="Defaults to generation.json next to the input manifest.")
    p.add_argument("--bomlip-root", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="Trusted local paper MACE-OFF23-small model.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--workers", type=int, default=8, help="Workers on this one GPU.")
    p.add_argument("--timeout", type=float, default=86400)
    args = p.parse_args()
    if args.workers < 1 or args.gpu < 0 or args.timeout <= 0:
        p.error("invalid atom count, worker count, GPU index or timeout")
    rows = read_candidates(args.manifest, require_energy=False)
    metadata_path = args.generation_metadata or args.manifest.resolve().parent / "generation.json"
    args.atoms_per_molecule = molecular_atom_count(args.manifest, metadata_path, rows[0]["sample_id"], args.atoms_per_molecule)
    from ase.io import read
    for row in rows:
        atoms = read(row["cif_path"])
        if len(atoms) % args.atoms_per_molecule:
            raise ValueError("cell atom count is not divisible by atoms-per-molecule")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    upstream = snapshot(args.bomlip_root, output / "upstream")
    model = args.model.resolve()
    provenance = pin_mace_model(upstream, model)
    inputs = output / "input_cifs"
    inputs.mkdir()
    names = {}
    for index, row in enumerate(rows):
        name = f"candidate_{index:08d}.cif"
        shutil.copy2(row["cif_path"], inputs / name)
        names[name] = row
    command = [sys.executable, str(upstream / "mace-bench/scripts/opt_batch.py"),
        "--target_folder", str(inputs), "--output_path", str(output),
        "--num_workers", str(args.workers), "--n_gpus", "1", "--gpu_offset", str(args.gpu),
        "--batch_size", "0", "--max_steps", "3000", "--filter1", "UnitCellFilter",
        "--filter2", "UnitCellFilter", "--optimizer1", "BFGSFusedLS", "--optimizer2", "BFGS",
        "--scalar_pressure", "0.0006", "--num_threads", "2", "--cueq", "True",
        "--molecule_single", str(args.atoms_per_molecule), "--use_ordered_files", "True", "--model", "mace"]
    provenance.update(bomlip_commit=BOMLIP_COMMIT, input_manifest_sha256=sha256(args.manifest),
        generation_metadata_sha256=sha256(metadata_path), command=command, input_mapping=names, atoms_per_molecule=args.atoms_per_molecule,
        energy_unit="kJ/mol per molecule", max_steps_per_stage=3000, fmax=0.01)
    record(output / "relaxation.json", provenance)
    env = os.environ.copy()
    # Compilers and Ninja must come from the same environment as Python, even
    # when this CLI is invoked by absolute path without shell activation.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PXRD_MACE_MODEL"] = str(model)
    env["PYTHONPATH"] = str(upstream / "mace-bench/src") + os.pathsep + env.get("PYTHONPATH", "")
    env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "2"
    run_logged(command, cwd=upstream, env=env, log=output / "relaxation.log", timeout=args.timeout)
    relaxed = []
    for name, original in names.items():
        cif = output / "cif_result_final" / name
        final = json.loads((output / "json_result_final" / Path(name).with_suffix(".json")).read_text())
        first = json.loads((output / "json_result_press" / Path(name).with_suffix(".json")).read_text())
        energy = float(final["energy"])
        if not math.isfinite(energy) or len(read(cif)) != len(read(original["cif_path"])):
            raise ValueError("invalid relaxation output")
        limit = any(int(stage["steps"]) >= 2999 for stage in (first, final))
        row = dict(original, cif_path=str(cif), cif_sha256=sha256(cif),
            energy_per_molecule=energy, energy_unit="kJ/mol", status="step_limit" if limit else "completed")
        relaxed.append(row)
    write_candidates(output / "relaxed.csv", relaxed)
    print(f"Wrote {len(relaxed)} two-stage outputs; completed is not a separate force-convergence certification.")


if __name__ == "__main__":
    main()
