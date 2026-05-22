"""Output directory and file utilities.

Adapted from muses_bench to remove external dependency.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def setup_output_structure(
    task_name: str,
    model: str,
    data_file: str,
    max_turns: int,
    debug: bool,
    output_file: Optional[str] = None,
    results_dir: str = "results",
    subdir: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Create standardized output directory structure.

    Format: ``{results_dir}/{task_name}[/{subdir}]/{data_stem}/{safe_model}/``
    Contains: detail.jsonl, eva.json

    Args:
        task_name: Scenario name (e.g. "instruction_following").
        model: Model name (slashes replaced with hyphens).
        data_file: Path to the input JSONL data file.
        max_turns: Max turns (used in legacy directory naming).
        debug: If True, append "_debug" to the model directory name.
        output_file: If provided, use this exact path for detail.jsonl.
        results_dir: Base directory for results (default "results").
        subdir: Optional segment inserted between ``task_name`` and
            ``data_stem`` (e.g. "replay" or "live"). ``None`` (default)
            preserves the legacy layout.

    Returns:
        Tuple of (log_file: Path, eva_file: Path).
    """
    if output_file:
        output_path = Path(output_file)
        os.makedirs(output_path.parent, exist_ok=True)
        eva_file = output_path.parent / "eva.json"
        print(f"[INFO] Using provided output file: {output_path}")
        print(f"[INFO] Evaluation Summary will be saved to: {eva_file}")
        return output_path, eva_file

    data_stem = Path(data_file).stem if data_file else "unknown_data"
    safe_model = model.replace("/", "-")
    if debug:
        safe_model += "_debug"

    base = Path(results_dir) / task_name
    if subdir:
        base = base / subdir
    output_dir = base / data_stem / safe_model
    os.makedirs(output_dir, exist_ok=True)

    log_file = output_dir / "detail.jsonl"
    eva_file = output_dir / "eva.json"

    print(f"[INFO] Output Directory: {output_dir}")
    print(f"[INFO] Detailed Log: {log_file}")
    print(f"[INFO] Evaluation: {eva_file}")

    return log_file, eva_file


def load_existing_results(
    log_file: str,
    required_keys: Optional[List[str]] = None,
) -> Tuple[List[Dict], Set[str]]:
    """Load existing results from a JSONL log for resume capability.

    Args:
        log_file: Path to detail.jsonl.
        required_keys: If provided, skip entries missing any of these keys.

    Returns:
        Tuple of (results: List[Dict], processed_ids: Set[str]).
    """
    results: List[Dict] = []
    processed_ids: Set[str] = set()

    if os.path.exists(log_file):
        print(f"Found existing log file: {log_file}. Resuming...")
        try:
            with open(log_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        res = json.loads(line)
                        if required_keys:
                            missing = [k for k in required_keys if k not in res]
                            if missing:
                                sid = res.get("scenario_id", "?")
                                print(
                                    f"[RESUME SKIP] Ignoring old result for {sid}: "
                                    f"missing keys {missing}"
                                )
                                continue
                        results.append(res)
                        sid = res.get("scenario_id")
                        if sid:
                            processed_ids.add(sid)
                    except json.JSONDecodeError:
                        print(f"Skipping malformed line in {log_file}")
            print(f"Loaded {len(results)} valid existing results.")
        except Exception as e:
            print(f"Error reading existing log file: {e}")

    return results, processed_ids
