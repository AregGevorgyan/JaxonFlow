"""Regression tracking for JaxonFlow benchmarks.

Tracks benchmark results across git commits, detects regressions, and
generates comparison reports.  Results are stored as JSON files in a
configurable results directory (default: benchmarks/results/).

Usage::

    tracker = RegressionTracker()

    # Record a new result
    result = runner.run_suite(suite)
    tracker.record(result)

    # Compare against the previous run
    comparison = tracker.compare_latest()
    print(comparison.summary())

    # Check for regressions
    regressions = tracker.check_regressions(threshold=0.05)
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .runner import BenchmarkResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskComparison:
    """Comparison of a single task between two benchmark runs.

    Attributes:
        task_name: Name of the task.
        old_speedup: Speedup from the baseline run.
        new_speedup: Speedup from the current run.
        delta: Absolute change in speedup (new - old).
        delta_pct: Percentage change ((new - old) / old * 100).
        regressed: Whether the task regressed beyond the threshold.
        old_correct: Whether the task was correct in the baseline run.
        new_correct: Whether the task is correct in the current run.
    """

    task_name: str
    old_speedup: float = 0.0
    new_speedup: float = 0.0
    delta: float = 0.0
    delta_pct: float = 0.0
    regressed: bool = False
    old_correct: bool = False
    new_correct: bool = False


@dataclass
class ComparisonReport:
    """Report comparing two benchmark runs.

    Attributes:
        baseline_commit: Git commit hash of the baseline run.
        current_commit: Git commit hash of the current run.
        suite_name: Name of the benchmark suite.
        task_comparisons: Per-task comparisons.
        baseline_summary: Summary of the baseline run.
        current_summary: Summary of the current run.
    """

    baseline_commit: str
    current_commit: str
    suite_name: str
    task_comparisons: list[TaskComparison] = field(default_factory=list)
    baseline_summary: dict[str, Any] = field(default_factory=dict)
    current_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def regressions(self) -> list[TaskComparison]:
        """Tasks that regressed."""
        return [c for c in self.task_comparisons if c.regressed]

    @property
    def improvements(self) -> list[TaskComparison]:
        """Tasks that improved."""
        return [c for c in self.task_comparisons if c.delta > 0 and not c.regressed]

    @property
    def correctness_changes(self) -> list[TaskComparison]:
        """Tasks where correctness status changed."""
        return [c for c in self.task_comparisons if c.old_correct != c.new_correct]

    def summary(self) -> dict[str, Any]:
        return {
            "baseline_commit": self.baseline_commit,
            "current_commit": self.current_commit,
            "suite_name": self.suite_name,
            "total_tasks": len(self.task_comparisons),
            "regressions": len(self.regressions),
            "improvements": len(self.improvements),
            "correctness_changes": len(self.correctness_changes),
            "baseline_correctness_rate": self.baseline_summary.get("correctness_rate", 0),
            "current_correctness_rate": self.current_summary.get("correctness_rate", 0),
            "baseline_geomean_speedup": self.baseline_summary.get("geometric_mean_speedup", 0),
            "current_geomean_speedup": self.current_summary.get("geometric_mean_speedup", 0),
        }

    def to_markdown(self) -> str:
        """Render the comparison as a Markdown table."""
        lines = [
            f"## Benchmark Comparison: {self.suite_name}",
            "",
            f"| Metric | Baseline (`{self.baseline_commit[:8]}`) | Current (`{self.current_commit[:8]}`) |",
            "|--------|----------|---------|",
            f"| Correctness | {self.baseline_summary.get('correctness_rate', 0):.1%} | {self.current_summary.get('correctness_rate', 0):.1%} |",
            f"| Geomean speedup | {self.baseline_summary.get('geometric_mean_speedup', 0):.3f}x | {self.current_summary.get('geometric_mean_speedup', 0):.3f}x |",
            f"| Total tokens | {self.baseline_summary.get('total_tokens', 0):,} | {self.current_summary.get('total_tokens', 0):,} |",
            "",
        ]

        if self.regressions:
            lines.append("### Regressions")
            lines.append("")
            lines.append("| Task | Old | New | Delta |")
            lines.append("|------|-----|-----|-------|")
            for r in self.regressions:
                lines.append(
                    f"| {r.task_name} | {r.old_speedup:.3f}x | {r.new_speedup:.3f}x | {r.delta_pct:+.1f}% |"
                )
            lines.append("")

        if self.improvements:
            lines.append("### Improvements")
            lines.append("")
            lines.append("| Task | Old | New | Delta |")
            lines.append("|------|-----|-----|-------|")
            for r in self.improvements:
                lines.append(
                    f"| {r.task_name} | {r.old_speedup:.3f}x | {r.new_speedup:.3f}x | {r.delta_pct:+.1f}% |"
                )
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result entry — wraps a BenchmarkResult with git metadata
# ---------------------------------------------------------------------------

@dataclass
class ResultEntry:
    """A benchmark result tagged with git metadata.

    Attributes:
        commit: Git commit hash.
        branch: Git branch name.
        timestamp: ISO-8601 timestamp.
        result: The benchmark result.
    """

    commit: str
    branch: str
    timestamp: str
    result: BenchmarkResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "commit": self.commit,
            "branch": self.branch,
            "timestamp": self.timestamp,
            "result": json.loads(self.result.to_json()),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResultEntry:
        result_data = data["result"]
        result = BenchmarkResult(
            suite_name=result_data["summary"]["suite_name"],
            hardware=result_data["summary"]["hardware"],
            wall_clock_time_s=result_data["summary"].get("wall_clock_time_s", 0.0),
        )
        # We only need summary-level data for regression tracking
        return cls(
            commit=data["commit"],
            branch=data["branch"],
            timestamp=data["timestamp"],
            result=result,
        )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class RegressionTracker:
    """Tracks benchmark results across git commits and detects regressions.

    Args:
        results_dir: Directory for storing result files.
        regression_threshold: Minimum fractional slowdown to flag as regression
            (e.g. 0.05 = 5% slower).
    """

    def __init__(
        self,
        results_dir: str | Path | None = None,
        regression_threshold: float = 0.05,
    ) -> None:
        self.results_dir = Path(results_dir) if results_dir else Path("benchmarks/results")
        self.regression_threshold = regression_threshold
        self._index_path = self.results_dir / "index.json"

    # ----- recording ----------------------------------------------------------

    def record(self, result: BenchmarkResult) -> Path:
        """Record a benchmark result with git metadata.

        Args:
            result: The benchmark result to record.

        Returns:
            Path to the saved result file.
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)

        commit = _get_git_commit()
        branch = _get_git_branch()
        timestamp = datetime.now().isoformat()

        # Save the full result (microseconds in filename to avoid collisions)
        filename = f"{result.suite_name}_{commit[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        result_path = self.results_dir / filename
        result.save(result_path)

        # Update the index
        index = self._load_index()
        index.append({
            "commit": commit,
            "branch": branch,
            "timestamp": timestamp,
            "suite_name": result.suite_name,
            "file": filename,
            "summary": result.summary(),
        })
        self._save_index(index)

        logger.info(f"Recorded benchmark result: {filename} (commit {commit[:8]})")
        return result_path

    # ----- comparison ---------------------------------------------------------

    def compare_latest(self, suite_name: str | None = None) -> ComparisonReport | None:
        """Compare the two most recent runs of a suite.

        Args:
            suite_name: Suite to compare. If None, uses the most recent suite.

        Returns:
            ComparisonReport or None if fewer than 2 runs exist.
        """
        index = self._load_index()

        if suite_name:
            entries = [e for e in index if e["suite_name"] == suite_name]
        else:
            entries = index

        if len(entries) < 2:
            logger.warning("Need at least 2 runs to compare")
            return None

        # Sort by timestamp descending
        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        current = entries[0]
        baseline = entries[1]

        return self._compare_entries(baseline, current)

    def compare_commits(
        self, baseline_commit: str, current_commit: str, suite_name: str | None = None
    ) -> ComparisonReport | None:
        """Compare results from two specific commits.

        Args:
            baseline_commit: Git commit hash prefix for the baseline.
            current_commit: Git commit hash prefix for the current.
            suite_name: Optional suite name filter.

        Returns:
            ComparisonReport or None if results not found.
        """
        index = self._load_index()

        baseline_entry = None
        current_entry = None

        for entry in index:
            if suite_name and entry["suite_name"] != suite_name:
                continue
            if entry["commit"].startswith(baseline_commit):
                baseline_entry = entry
            if entry["commit"].startswith(current_commit):
                current_entry = entry

        if baseline_entry is None or current_entry is None:
            logger.warning("Could not find results for both commits")
            return None

        return self._compare_entries(baseline_entry, current_entry)

    def check_regressions(
        self,
        threshold: float | None = None,
        suite_name: str | None = None,
    ) -> list[TaskComparison]:
        """Check for regressions in the most recent run.

        Args:
            threshold: Override the regression threshold.
            suite_name: Suite to check.

        Returns:
            List of regressed task comparisons.
        """
        report = self.compare_latest(suite_name)
        if report is None:
            return []

        thr = threshold if threshold is not None else self.regression_threshold
        regressions = []
        for comp in report.task_comparisons:
            if comp.old_speedup > 0 and comp.delta_pct < -(thr * 100):
                comp.regressed = True
                regressions.append(comp)

        return regressions

    # ----- history ------------------------------------------------------------

    def list_runs(self, suite_name: str | None = None) -> list[dict[str, Any]]:
        """List all recorded runs.

        Args:
            suite_name: Optional filter by suite name.

        Returns:
            List of index entries (commit, branch, timestamp, summary).
        """
        index = self._load_index()
        if suite_name:
            index = [e for e in index if e["suite_name"] == suite_name]
        return index

    def get_history(self, task_name: str, suite_name: str | None = None) -> list[dict[str, Any]]:
        """Get performance history for a specific task across commits.

        Args:
            task_name: Task to track.
            suite_name: Optional suite filter.

        Returns:
            List of {commit, timestamp, speedup, correct} entries.
        """
        index = self._load_index()
        history = []

        for entry in index:
            if suite_name and entry["suite_name"] != suite_name:
                continue

            # Load the full result to get per-task data
            result_path = self.results_dir / entry["file"]
            if not result_path.exists():
                continue

            try:
                result = BenchmarkResult.load(result_path)
                for tr in result.task_results:
                    if tr.task_name == task_name:
                        history.append({
                            "commit": entry["commit"],
                            "timestamp": entry["timestamp"],
                            "speedup": tr.speedup_vs_reference,
                            "correct": tr.correctness,
                            "execution_time_us": tr.execution_time_us,
                        })
                        break
            except Exception as e:
                logger.debug(f"Could not load result {entry['file']}: {e}")

        return history

    # ----- internal -----------------------------------------------------------

    def _compare_entries(
        self, baseline: dict[str, Any], current: dict[str, Any]
    ) -> ComparisonReport:
        """Compare two index entries."""
        # Load full results
        baseline_result = self._load_result(baseline["file"])
        current_result = self._load_result(current["file"])

        report = ComparisonReport(
            baseline_commit=baseline["commit"],
            current_commit=current["commit"],
            suite_name=current.get("suite_name", "unknown"),
            baseline_summary=baseline.get("summary", {}),
            current_summary=current.get("summary", {}),
        )

        if baseline_result is None or current_result is None:
            return report

        # Build task lookup
        baseline_tasks = {r.task_name: r for r in baseline_result.task_results}
        current_tasks = {r.task_name: r for r in current_result.task_results}

        all_task_names = set(baseline_tasks.keys()) | set(current_tasks.keys())

        for name in sorted(all_task_names):
            old = baseline_tasks.get(name)
            new = current_tasks.get(name)

            old_speedup = old.speedup_vs_reference if old else 0.0
            new_speedup = new.speedup_vs_reference if new else 0.0
            delta = new_speedup - old_speedup
            delta_pct = (delta / old_speedup * 100) if old_speedup > 0 else 0.0

            comp = TaskComparison(
                task_name=name,
                old_speedup=old_speedup,
                new_speedup=new_speedup,
                delta=delta,
                delta_pct=delta_pct,
                regressed=delta_pct < -(self.regression_threshold * 100),
                old_correct=old.correctness if old else False,
                new_correct=new.correctness if new else False,
            )
            report.task_comparisons.append(comp)

        return report

    def _load_result(self, filename: str) -> BenchmarkResult | None:
        """Load a BenchmarkResult from the results directory."""
        path = self.results_dir / filename
        if not path.exists():
            logger.warning(f"Result file not found: {path}")
            return None
        try:
            return BenchmarkResult.load(path)
        except Exception as e:
            logger.warning(f"Failed to load result {filename}: {e}")
            return None

    def _load_index(self) -> list[dict[str, Any]]:
        """Load the index file."""
        if not self._index_path.exists():
            return []
        try:
            return json.loads(self._index_path.read_text())
        except Exception:
            return []

    def _save_index(self, index: list[dict[str, Any]]) -> None:
        """Save the index file."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps(index, indent=2))
