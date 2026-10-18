"""Candidate identity, energy and provenance contracts for downstream tools."""
from __future__ import annotations
import csv
import hashlib
import math
from pathlib import Path

FIELDS = ["sample_id", "space_group", "candidate_id", "cif_path", "cif_sha256",
          "energy_per_molecule", "energy_unit", "status"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_candidates(path: Path, *, require_energy: bool = True) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"manifest requires columns: {FIELDS}")
        rows = list(reader)
    if not rows or len({r["sample_id"] for r in rows}) != 1:
        raise ValueError("manifest must contain candidates for exactly one sample")
    if any(not r["candidate_id"] or not r["sample_id"] for r in rows):
        raise ValueError("sample and candidate IDs must be non-empty")
    if len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("candidate IDs must be unique across selected space groups")
    for row in rows:
        group = int(row["space_group"])
        if not 1 <= group <= 230:
            raise ValueError("space_group must be an integer from 1 to 230; Others is not a group")
        cif = (path.resolve().parent / row["cif_path"]).resolve()
        if sha256(cif) != row["cif_sha256"]:
            raise ValueError(f"CIF hash mismatch: {row['candidate_id']}")
        row["cif_path"] = str(cif)
        if require_energy:
            if row["status"] not in {"completed", "step_limit"}:
                raise ValueError("ranking requires completed two-stage relaxation outputs")
            if row["energy_unit"] != "kJ/mol" or not math.isfinite(float(row["energy_per_molecule"])):
                raise ValueError("energies must be finite kJ/mol per molecule")
    return rows


def write_candidates(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no candidates produced")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in FIELDS} for row in rows)


def candidate_row(sample: str, group: int, cif: Path, *, energy="", status="generated") -> dict:
    return dict(zip(FIELDS, [sample, group, f"sg{group}:{cif.stem}", str(cif.resolve()),
                            sha256(cif), energy, "kJ/mol" if energy != "" else "", status]))
