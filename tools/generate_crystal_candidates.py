#!/usr/bin/env python3
"""Prepare paper-protocol conformers and generate Z-prime-one BOMLIP-CSP candidates."""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import re
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "pyproject.toml").is_file():
    sys.path.insert(0, str(ROOT))
else:
    sys.path.append(str(ROOT))
from src.runtime.crystal_candidates import candidate_row, write_candidates, sha256
from src.runtime.downstream_execution import BOMLIP_COMMIT, snapshot, run_logged, record


def prepare_conformer(smiles: str, output: Path) -> dict:
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("a valid single-component molecule is required")
    molecule = Chem.AddHs(molecule)
    if not AllChem.UFFHasAllMoleculeParams(molecule):
        raise ValueError("UFF parameters are unavailable for this molecule")
    params = AllChem.ETKDGv3()
    params.numThreads = 0
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = 10000
    else:
        params.maxIterations = 10000
    params.randomSeed = 240430
    params.pruneRmsThresh = 0.1
    ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=20, params=params))
    if not ids:
        raise ValueError("conformer embedding failed")
    energies = AllChem.UFFOptimizeMoleculeConfs(molecule, numThreads=0)
    if len(ids) != len(energies) or not all(math.isfinite(e) for _, e in energies):
        raise ValueError("invalid UFF output")
    ranked = sorted((int(flag), float(energy), cid) for cid, (flag, energy) in zip(ids, energies))
    cid = ranked[0][2]
    conformer = molecule.GetConformer(cid)
    lines = [str(molecule.GetNumAtoms()), f"conformer {cid}; UFF energy {ranked[0][1]:.8f}"]
    for atom in molecule.GetAtoms():
        pos = conformer.GetAtomPosition(atom.GetIdx())
        lines.append(f"{atom.GetSymbol():<6} {pos.x:16.8f} {pos.y:16.8f} {pos.z:16.8f}")
    output.write_text("\n".join(lines) + "\n")
    return {"smiles": Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True),
        "rdkit_version": rdBase.rdkitVersion, "atoms_per_molecule": molecule.GetNumAtoms(),
        "requested_conformers": 20, "embedded_conformers": len(ids), "seed": 240430,
        "rms_pruning_angstrom": 0.1, "selected_conformer_id": cid,
        "ranked_uff_status_energy_id": ranked, "xyz_sha256": sha256(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smiles", required=True)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--space-groups", type=int, nargs="+", required=True)
    p.add_argument("--bomlip-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=5000, help="Successful candidates per group; smaller counts are smoke tests.")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=float, default=86400, help="Maximum seconds per group, including unsuccessful generation attempts.")
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.sample_id):
        p.error("sample-id must use letters, digits, underscores or hyphens")
    if not args.space_groups or any(not 1 <= g <= 230 for g in args.space_groups) or len(set(args.space_groups)) != len(args.space_groups):
        p.error("supply unique concrete space groups in 1..230")
    if args.count < 1 or args.workers < 1 or args.timeout <= 0:
        p.error("count, workers and timeout must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    upstream = snapshot(args.bomlip_root, output / "upstream")
    metadata = prepare_conformer(args.smiles, output / "selected.xyz")
    metadata.update(sample_id=args.sample_id, bomlip_commit=BOMLIP_COMMIT,
                    candidates_per_group=args.count, space_groups=args.space_groups,
                    protocol_run=args.count == 5000, commands=[])
    record(output / "generation.json", metadata)
    rows = []
    for group in args.space_groups:
        run = output / f"sg{group}"
        conformers = run / "molecule_1/conformers"
        conformers.mkdir(parents=True)
        shutil.copy2(output / "selected.xyz", conformers / "conformer_0.xyz")
        command = [sys.executable, str(upstream / "main.py"), "--path", str(run),
            "--generate_conformers", "0", "--use_conformers", "1", "--mode", "structure_only",
            "--molecule_num_in_cell", "1", "--space_group_list", str(group),
            "--num_generation", str(args.count), "--add_name", args.sample_id,
            "--max_workers", str(args.workers)]
        metadata["commands"].append(command)
        record(output / "generation.json", metadata)
        run_logged(command, cwd=upstream, env=os.environ.copy(), log=run / "generation.log", timeout=args.timeout)
        cifs = sorted((run / "structures").glob("*.cif"))
        if len(cifs) != args.count:
            raise RuntimeError(f"expected {args.count} CIFs for SG{group}, got {len(cifs)}")
        rows.extend(candidate_row(args.sample_id, group, cif) for cif in cifs)
    write_candidates(output / "candidates.csv", rows)
    metadata["manifest_sha256"] = sha256(output / "candidates.csv")
    record(output / "generation.json", metadata)
    print(f"Generated {len(rows)} candidates. Manifest: {output / 'candidates.csv'}")


if __name__ == "__main__":
    main()
