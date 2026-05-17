#!/usr/bin/env python3
"""Identify the Pycaso coin specimen dataset for 3D reconstruction."""

import json
from pathlib import Path

PYCASO = Path("examples/pycaso_data/Exemple/Images_example")
if not PYCASO.exists():
    PYCASO = Path("/home/jeff/StereoComplex/examples/pycaso_data/Exemple/Images_example")

OUT = Path("docs/assets/pycaso_real_data")
OUT.mkdir(parents=True, exist_ok=True)

# ── Inventory ──
candidate_pairs = []
all_left = sorted(d for d in PYCASO.iterdir() if d.is_dir() and d.name.startswith("left_"))
all_right = sorted(d for d in PYCASO.iterdir() if d.is_dir() and d.name.startswith("right_"))

for ld in all_left:
    for rd in all_right:
        l_files = sorted(f.name for f in ld.iterdir() if f.suffix in (".tif", ".tiff", ".png"))
        r_files = sorted(f.name for f in rd.iterdir() if f.suffix in (".tif", ".tiff", ".png"))
        n_left = len(l_files)
        n_right = len(r_files)
        reason = ""

        if n_left == 0 or n_right == 0:
            reason = "empty directory"
        elif set(l_files) != set(r_files):
            reason = "different filenames"
        elif n_left == 1:
            reason = "single specimen image (coin)"
        elif n_left <= 11:
            reason = "calibration stack (ChArUco)"
        else:
            reason = "calibration sweep"

        candidate_pairs.append({
            "left_dir": ld.name,
            "right_dir": rd.name,
            "n_left": n_left,
            "n_right": n_right,
            "left_files": l_files[:5],
            "right_files": r_files[:5],
            "reason": reason,
        })

# ── Select the specimen pair ──
# Coin specimen: left_identification/coin.tif, right_identification2/coin_1.tif
selected = {
    "left_dir": "left_identification",
    "right_dir": "right_identification2",
    "left_file": "coin.tif",
    "right_file": "coin_1.tif",
    "note": "Speckled coin specimen for 3D reconstruction. Single stereo pair, no ChArUco pattern."
}

inventory = {
    "pycaso_root": str(PYCASO.resolve()),
    "total_candidate_pairs": len(candidate_pairs),
    "candidate_pairs": candidate_pairs,
    "selected_specimen_pair": selected,
}

with open(OUT / "specimen_dataset_inventory.json", "w") as f:
    json.dump(inventory, f, indent=2)

print(f"Inventory saved: {OUT / 'specimen_dataset_inventory.json'}")
print(f"  {len(candidate_pairs)} candidate pairs scanned")
print(f"  Selected: left={selected['left_dir']}/{selected['left_file']}")
print(f"           right={selected['right_dir']}/{selected['right_file']}")
