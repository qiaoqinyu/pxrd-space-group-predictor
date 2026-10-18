"""Paper-protocol StructureMatcher compression and pseudo-Voigt PXRD."""
from __future__ import annotations
import math
import numpy as np

THETA = 2.0 + np.arange(3400, dtype=np.float64) * 0.02


def deduplicate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher
    from pymatgen.io.cif import CifParser
    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5,
        primitive_cell=False, scale=True, attempt_supercell=False,
        comparator=ElementComparator(), ignored_species=["H"])
    representatives, mapping = [], []
    for group in sorted({int(r["space_group"]) for r in rows}):
        pool = sorted((r for r in rows if int(r["space_group"]) == group),
                      key=lambda r: (float(r["energy_per_molecule"]), r["candidate_id"]))
        retained = []
        for row in pool:
            structure = CifParser(row["cif_path"], frac_tolerance=0).parse_structures(primitive=False)[0]
            representative = next((rep for rep, parsed in retained if matcher.fit(parsed, structure)), None)
            if representative is None:
                representative = row
                retained.append((row, structure))
                representatives.append(row)
            mapping.append({"candidate_id": row["candidate_id"],
                            "representative_candidate_id": representative["candidate_id"]})
    return representatives, mapping


def simulate(cif_path) -> np.ndarray:
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from pymatgen.io.cif import CifParser
    structure = CifParser.from_str(cif_path.read_text(encoding="utf-8")).parse_structures(primitive=False)[0]
    pattern = XRDCalculator(wavelength=1.54184).get_pattern(structure, scaled=True, two_theta_range=(2, 70))
    profile = np.zeros(3400, dtype=np.float64)
    for center, intensity in zip(pattern.x, pattern.y):
        if not math.isfinite(center) or not math.isfinite(intensity) or intensity <= 0:
            continue
        distance = (THETA - center) / 0.15
        gaussian = np.exp(-4 * math.log(2) * distance ** 2)
        lorentz = 1 / (1 + 4 * distance ** 2)
        profile += intensity * (0.5 * gaussian + 0.5 * lorentz)
    profile = np.clip(profile, 0, None)
    maximum = float(profile.max())
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError(f"no usable simulated peaks: {cif_path.name}")
    return (profile * (100 / maximum)).astype(np.float32)
