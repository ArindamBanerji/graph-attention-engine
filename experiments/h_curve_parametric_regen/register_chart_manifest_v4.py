from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "raw" / "two_arm_v4" / "manifest.json"
FIG = ROOT / "figures"

data = json.loads(MANIFEST.read_text(encoding="utf-8"))
files = [f for f in data["files"] if f["path"] not in {
    "figures/hcurve_gamma_by_epsilon.png",
    "figures/hcurve_gamma_by_epsilon.pdf",
    "figures/chart_metadata.json",
    "../../figures/hcurve_gamma_by_epsilon.png",
    "../../figures/hcurve_gamma_by_epsilon.pdf",
    "../../figures/chart_metadata.json",
    "../../figures/hcurve_gamma_by_epsilon.caption.txt",
}]
for name, role in [
    ("hcurve_gamma_by_epsilon.png", "publication_verification_chart_png"),
    ("hcurve_gamma_by_epsilon.pdf", "publication_verification_chart_pdf"),
    ("chart_metadata.json", "publication_verification_chart_metadata"),
    ("hcurve_gamma_by_epsilon.caption.txt", "publication_verification_chart_caption"),
]:
    path = FIG / name
    raw = path.read_bytes()
    files.append({
        "path": f"../../figures/{name}",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "role": role,
    })
data["files"] = files
data["verification_chart"] = {
    "png": "../../figures/hcurve_gamma_by_epsilon.png",
    "pdf": "../../figures/hcurve_gamma_by_epsilon.pdf",
    "caption": "../../figures/hcurve_gamma_by_epsilon.caption.txt",
    "source": "persisted fit_gate_recompute.json, Arm C summary.json, and trajectories",
}
fd, temp = tempfile.mkstemp(prefix="manifest.", suffix=".json", dir=MANIFEST.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    os.replace(temp, MANIFEST)
finally:
    if os.path.exists(temp):
        os.unlink(temp)

check = json.loads(MANIFEST.read_text(encoding="utf-8"))
valid = True
for item in check["files"]:
    path = MANIFEST.parent / item["path"]
    valid = valid and path.exists() and path.stat().st_size == item["bytes"]
    if path.exists():
        valid = valid and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
print(json.dumps({"files": len(check["files"]), "hashes_valid": valid}, indent=2))
if not valid:
    raise SystemExit(1)
