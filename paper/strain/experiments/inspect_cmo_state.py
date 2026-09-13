"""Print the content structure of the saved Pycaso CMO intermediate state.

This is a provenance/engineering helper for the Strain reframing. It lets us
reuse the *committed* CMO calibration state rather than re-entering parameters
from the manuscript by hand.
"""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
state_path = ROOT / "docs/assets/pycaso_real_data/intermediate_state.npz"
print(f"STATE {state_path}")
with np.load(state_path, allow_pickle=True) as z:
    print("KEYS", z.files)
    for key in z.files:
        a = z[key]
        print(f"KEY {key!r}: shape={a.shape}, dtype={a.dtype}, size={a.size}")
        if a.size <= 200:
            try:
                print("VALUE", np.asarray(a).tolist())
            except Exception as exc:
                print("VALUE_ERROR", exc)

for name in (
    "aligned_cmo_fit.json",
    "arm_alignment_diagnostic.json",
    "autopsy_20p.json",
    "autopsy_26p.json",
    "corner_ba_refinement.json",
    "model_comparison.json",
):
    p = ROOT / "docs/assets/pycaso_real_data" / name
    if not p.exists():
        continue
    print(f"\nJSON {name}")
    obj = json.loads(p.read_text())
    if name == "corner_ba_refinement.json":
        print("TOP_KEYS", sorted(obj.keys()))
        for k, v in obj.items():
            if any(token in k.lower() for token in ("param", "model", "arm", "rot", "trans", "initial", "final")):
                text = json.dumps(v)
                print(k, text[:12000])
    else:
        print(json.dumps(obj, indent=2)[:20000])

print("\nTEXT OCCURRENCES OF 26P IMPLEMENTATION TOKENS")
needles = ("x_26p", "aligned_26p", "arm_L", "rv_Lx", "full SE(3)")
for p in ROOT.rglob("*"):
    if p.suffix.lower() not in {".py", ".md", ".tex", ".txt"}:
        continue
    try:
        lines = p.read_text(errors="ignore").splitlines()
    except Exception:
        continue
    hits = []
    for i, line in enumerate(lines, start=1):
        if any(n in line for n in needles):
            hits.append((i, line.strip()))
    if hits:
        print(f"FILE {p.relative_to(ROOT)}")
        for i, line in hits[:80]:
            print(f"  {i}: {line}")
