#!/usr/bin/env python3
"""Create the ablation config (ComplianceChecker disabled).

Copies harness_mu/config/my_experiment.yaml and sets
scenarios.access_control.modules.compliance_checker to false.

Usage:
    python repro/make_ablation_config.py
"""

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[ERROR] pyyaml not installed. Run: pip install pyyaml")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "harness_mu" / "config" / "my_experiment.yaml"
DST = REPO_ROOT / "harness_mu" / "config" / "ablation_no_cc.yaml"


def main():
    if not SRC.exists():
        print(f"[ERROR] Source config not found: {SRC}")
        sys.exit(1)

    with open(SRC) as f:
        cfg = yaml.safe_load(f)

    # Enable AC and disable ComplianceChecker
    ac = cfg.setdefault("scenarios", {}).setdefault("access_control", {})
    ac["enabled"] = True
    ac.setdefault("modules", {})["compliance_checker"] = False

    with open(DST, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"[OK] Ablation config written to: {DST}")
    print("     Change: scenarios.access_control.modules.compliance_checker = false")
    print("     Use with: python -m harness_mu --scenario access_control --config", DST)


if __name__ == "__main__":
    main()
