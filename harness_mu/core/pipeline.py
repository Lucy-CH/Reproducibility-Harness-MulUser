"""Pipeline orchestrator that assembles modules and drives batch execution."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness_mu.utils.file_utils import setup_output_structure, load_existing_results

from harness_mu.config.schema import HarnessMUConfig, ScenarioConfig
from harness_mu.core.types import ScenarioRecord
from harness_mu.llm.factory import LLMFactory
from harness_mu.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Assembles modules per scenario config and drives batch execution.

    Reads a JSONL data file, routes each line through the appropriate
    scenario adapter, and writes results incrementally.
    """

    def __init__(
        self,
        config: HarnessMUConfig,
        llm_factory: LLMFactory,
        metrics_collector: MetricsCollector,
    ):
        self._config = config
        self._llm_factory = llm_factory
        self._metrics_collector = metrics_collector
        self._adapters: Dict[str, object] = {}

    def run_batch(
        self,
        scenario_name: str,
        data_file: Optional[str] = None,
        output_file: Optional[str] = None,
        debug: bool = False,
    ) -> List[ScenarioRecord]:
        """Run a full batch of scenarios.

        Args:
            scenario_name: Key in config.scenarios (e.g. "instruction_following").
            data_file: Override path to JSONL data file.
            output_file: Override path for output.
            debug: If True, limit to 3 scenarios with single-worker execution.

        Returns:
            List of ScenarioRecord, one per JSONL line.
        """
        scenario_cfg = self._config.scenarios.get(scenario_name)
        if scenario_cfg is None:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(self._config.scenarios.keys())}"
            )

        data_path = data_file or scenario_cfg.data_file
        if not data_path:
            raise ValueError(f"No data_file configured for scenario '{scenario_name}'")

        # Resolve data path relative to Muses-Bench root
        data_path_obj = self._resolve_data_path(data_path)
        if not data_path_obj.exists():
            raise FileNotFoundError(f"Data file not found: {data_path_obj}")

        logger.info("Loading data from: %s", data_path_obj)

        with open(data_path_obj, "r") as f:
            lines = f.readlines()

        if debug:
            lines = lines[:3]
            logger.info("[debug] limited to first %d scenarios", len(lines))

        max_workers = 1 if debug else self._config.execution.max_workers
        logger.info("Executing with max_workers=%d", max_workers)

        # Setup output files
        log_file, eva_file = setup_output_structure(
            scenario_name,
            self._config.default_model.model,
            str(data_path_obj),
            max_turns=1,
            debug=debug,
            output_file=output_file,
            results_dir=self._config.output.results_dir,
            subdir=self._config.output.subdir,
        )

        # Resume capability
        existing_results, processed_ids = (
            load_existing_results(str(log_file)) if log_file else ([], set())
        )

        # Seed results with resumed scenarios so eva.json aggregates over the
        # full set (resumed + fresh) instead of only the fresh ones.
        results: List[ScenarioRecord] = [
            ScenarioRecord(
                scenario_id=r.get("scenario_id", ""),
                metrics=r.get("metrics", {}) or {},
            )
            for r in existing_results
        ]

        # Import adapter lazily to avoid circular imports
        adapter = self._get_adapter(scenario_name)
        io_lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for line in lines:
                # Skip already-processed scenarios
                try:
                    data = json.loads(line)
                    sid = data.get("id", "")
                except json.JSONDecodeError:
                    continue
                if sid in processed_ids:
                    continue
                futures[executor.submit(adapter.run, line)] = sid

            for future in as_completed(futures):
                sid = futures.get(future, "?")
                try:
                    record = future.result()
                except Exception as e:
                    logger.warning("[%s] scenario crashed: %s", sid, e)
                    continue
                if record is None:
                    continue

                with io_lock:
                    results.append(record)

                    # Write detail line
                    if log_file:
                        with open(log_file, "a") as f_out:
                            f_out.write(json.dumps({
                                "scenario_id": record.scenario_id,
                                "metrics": record.metrics,
                                "instruction_results": record.instruction_results,
                                "module_results": [
                                    {
                                        "module_name": mr.module_name,
                                        "success": mr.success,
                                        "data": mr.data,
                                        "latency_ms": mr.latency_ms,
                                        "token_usage": mr.token_usage,
                                        "error": mr.error,
                                    }
                                    for mr in record.module_results
                                ],
                                "raw_outputs": record.raw_outputs,
                            }, ensure_ascii=False) + "\n")

                    # Update evaluation summary
                    if eva_file:
                        self._write_eva_summary(eva_file, results)

        # Write module metrics summary
        if eva_file and self._config.output.save_metrics:
            module_metrics_path = Path(eva_file).parent / "module_metrics.json"
            module_summaries = self._metrics_collector.get_module_summaries()
            with open(module_metrics_path, "w") as f:
                json.dump({
                    name: {
                        "total_calls": s.total_calls,
                        "total_prompt_tokens": s.total_prompt_tokens,
                        "total_completion_tokens": s.total_completion_tokens,
                        "total_latency_ms": s.total_latency_ms,
                        "avg_latency_ms": s.avg_latency_ms,
                        "model": s.model,
                    }
                    for name, s in module_summaries.items()
                }, f, indent=2)
            logger.info("Module metrics written to: %s", module_metrics_path)

        # Final summary
        self._print_summary(results)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_adapter(self, scenario_name: str):
        """Get or lazily create the scenario adapter."""
        if scenario_name not in self._adapters:
            if scenario_name == "instruction_following":
                from harness_mu.adapters.instruction_following import (
                    InstructionFollowingAdapter,
                )
                self._adapters[scenario_name] = InstructionFollowingAdapter(
                    config=self._config,
                    llm_factory=self._llm_factory,
                    metrics_collector=self._metrics_collector,
                )
            elif scenario_name == "access_control":
                from harness_mu.adapters.access_control import AccessControlAdapter
                self._adapters[scenario_name] = AccessControlAdapter(
                    config=self._config,
                    llm_factory=self._llm_factory,
                    metrics_collector=self._metrics_collector,
                )
            else:
                raise ValueError(f"No adapter for scenario: {scenario_name}")
        return self._adapters[scenario_name]

    @staticmethod
    def _resolve_data_path(data_path: str) -> Path:
        """Resolve data path relative to the harness_mu package root."""
        p = Path(data_path)
        if p.is_absolute():
            return p
        # Try relative to harness_mu package directory
        pkg_root = Path(__file__).resolve().parents[1]
        return pkg_root / data_path

    @staticmethod
    def _write_eva_summary(eva_path: Path, results: List[ScenarioRecord]):
        """Write the incremental evaluation summary JSON."""
        summary: Dict[str, Any] = {"total_scenarios": len(results)}

        def _avg(key: str) -> Optional[float]:
            vals = [r.metrics.get(key) for r in results if isinstance(r.metrics.get(key), (int, float))]
            return sum(vals) / len(vals) if vals else None

        for k in (
            "accuracy", "weighted_accuracy",
            "privacy_score", "utility_score", "overall_score",
            # raw/final dual-output: raw = pre-Boundary-Layer score.
            "accuracy_raw", "weighted_accuracy_raw",
            "privacy_score_raw", "utility_score_raw", "overall_score_raw",
            "boundary_delta_accuracy", "boundary_delta_weighted_accuracy",
            "boundary_delta_privacy", "boundary_delta_utility", "boundary_delta_overall",
            # Boundary-Layer activity counters (IF revision rounds / AC fail-closed rewrites).
            "revision_count", "num_boundary_intercepts",
        ):
            v = _avg(k)
            if v is not None:
                summary[f"avg_{k}"] = v

        # Harness-MU IF custom metrics
        authority_scores = []
        authority_scores_raw = []
        per_user_rates = []
        for r in results:
            hmu = r.metrics.get("_harness_mu", {}) or {}
            acs = hmu.get("authority_compliance_score")
            if acs is not None:
                authority_scores.append(acs)
            acs_raw = hmu.get("authority_compliance_score_raw")
            if acs_raw is not None:
                authority_scores_raw.append(acs_raw)
            for pu in hmu.get("per_user_satisfaction", []) or []:
                per_user_rates.append(pu.get("score", 0.0))
        if authority_scores:
            summary["avg_authority_compliance"] = sum(authority_scores) / len(authority_scores)
        if authority_scores_raw:
            summary["avg_authority_compliance_raw"] = (
                sum(authority_scores_raw) / len(authority_scores_raw)
            )
        if per_user_rates:
            summary["avg_per_user_accuracy"] = sum(per_user_rates) / len(per_user_rates)

        with open(eva_path, "w") as f:
            json.dump(summary, f, indent=2)

    @staticmethod
    def _print_summary(results: List[ScenarioRecord]):
        """Print final summary to stdout."""
        if not results:
            logger.info("No results to summarize.")
            return
        print(f"\n{'='*50}")
        print(f"  Scenarios completed: {len(results)}")
        for k in ("accuracy", "privacy_score", "utility_score", "overall_score"):
            vals = [r.metrics.get(k) for r in results if isinstance(r.metrics.get(k), (int, float))]
            if vals:
                print(f"  Average {k:<14} {sum(vals)/len(vals):.2%}")
        print(f"{'='*50}")
