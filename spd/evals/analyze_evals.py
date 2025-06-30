"""Analyze completed SPD evaluation runs and generate reports.

This script fetches W&B runs by evals_id, performs various analyses,
and updates the evaluation report with results.

Usage:
    uv run spd/evals/analyze_evals.py <evals_id> <report_url>
"""

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, override

import wandb
import wandb_workspaces.reports.v2 as wr
from dotenv import load_dotenv

from spd.log import logger
from spd.registry import EXPERIMENT_REGISTRY


@dataclass
class EvalRunData:
    """Data structure for individual evaluation run results."""

    run_id: str
    experiment_name: str
    experiment_type: str
    state: str
    final_faithfulness_loss: float | None
    run_name: str
    tags: list[str]
    url: str
    config: dict[str, Any]


@dataclass
class AnalysisResult:
    """Structured result from an analysis."""

    analyzer_name: str
    success: bool
    summary: str
    details: str
    warnings: list[str]
    markdown_report: str


class BaseAnalyzer(ABC):
    """Abstract base class for evaluation analyzers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this analyzer."""
        pass

    @abstractmethod
    def analyze(self, runs: list[EvalRunData]) -> AnalysisResult:
        """Perform analysis on the provided runs."""
        pass


class FaithfulnessAnalyzer(BaseAnalyzer):
    """Analyzer for faithfulness loss thresholds."""

    def __init__(self):
        self.threshold = 1e-3  # Hardcoded threshold

    @property
    @override
    def name(self) -> str:
        return "Faithfulness Loss Analysis"

    @override
    def analyze(self, runs: list[EvalRunData]) -> AnalysisResult:
        """Analyze faithfulness loss across runs."""
        warnings = []

        # Filter runs with faithfulness data
        completed_runs = [run for run in runs if run.state == "finished"]
        runs_with_data = [run for run in completed_runs if run.final_faithfulness_loss is not None]

        # Track runs without data
        runs_without_data = [run for run in completed_runs if run.final_faithfulness_loss is None]
        failed_runs = [run for run in runs if run.state != "finished"]

        if failed_runs:
            warnings.append(f"{len(failed_runs)} runs failed or did not complete")

        if runs_without_data:
            warnings.append(f"{len(runs_without_data)} completed runs missing faithfulness data")

        if not runs_with_data:
            return AnalysisResult(
                analyzer_name=self.name,
                success=False,
                summary="No runs with faithfulness data available",
                details="Cannot perform faithfulness analysis without valid data",
                warnings=warnings,
                markdown_report="## ❌ Faithfulness Analysis Failed\n\nNo runs with valid faithfulness data found.",
            )

        # Perform analysis
        passing_runs = []
        failing_runs = []

        for run in runs_with_data:
            assert run.final_faithfulness_loss is not None
            if run.final_faithfulness_loss <= self.threshold:
                passing_runs.append(run)
            else:
                failing_runs.append(run)

        # Generate summary
        total_analyzed = len(runs_with_data)
        pass_rate = len(passing_runs) / total_analyzed if total_analyzed > 0 else 0

        all_passed = len(failing_runs) == 0

        summary = f"{len(passing_runs)}/{total_analyzed} runs passed faithfulness threshold ({self.threshold:.0e})"

        # Generate detailed markdown report
        markdown_report = self._generate_markdown_report(
            passing_runs, failing_runs, warnings, pass_rate
        )

        return AnalysisResult(
            analyzer_name=self.name,
            success=all_passed,
            summary=summary,
            details=f"Pass rate: {pass_rate:.1%}",
            warnings=warnings,
            markdown_report=markdown_report,
        )

    def _generate_markdown_report(
        self,
        passing_runs: list[EvalRunData],
        failing_runs: list[EvalRunData],
        warnings: list[str],
        pass_rate: float,
    ) -> str:
        """Generate detailed markdown report."""
        status_emoji = "✅" if len(failing_runs) == 0 else "❌"

        report = f"## {status_emoji} Faithfulness Loss Analysis\n\n"

        if warnings:
            report += "### ⚠️ Warnings\n"
            for warning in warnings:
                report += f"- {warning}\n"
            report += "\n"

        report += f"**Threshold:** {self.threshold:.0e}  \n"
        report += f"**Pass Rate:** {pass_rate:.1%} ({len(passing_runs)}/{len(passing_runs) + len(failing_runs)})  \n\n"

        if passing_runs:
            report += "### ✅ Passing Runs\n"
            report += "| Experiment | Final Faithfulness Loss |\n"
            report += "|------------|------------------------|\n"
            for run in passing_runs:
                report += (
                    f"| [{run.experiment_name}]({run.url}) | {run.final_faithfulness_loss:.2e} |\n"
                )
            report += "\n"

        if failing_runs:
            report += "### ❌ Failing Runs\n"
            report += "| Experiment | Final Faithfulness Loss |\n"
            report += "|------------|------------------------|\n"
            for run in failing_runs:
                report += (
                    f"| [{run.experiment_name}]({run.url}) | {run.final_faithfulness_loss:.2e} |\n"
                )
            report += "\n"

        return report


class EvalAnalyzer:
    """Main analyzer class that coordinates multiple analysis types."""

    def __init__(self, evals_id: str, report_url: str):
        self.evals_id = evals_id
        self.report_url = report_url
        self.analyzers: list[BaseAnalyzer] = []

        # Load W&B configuration
        load_dotenv(override=True)
        self.wandb_api = wandb.Api()

    def add_analyzer(self, analyzer: BaseAnalyzer) -> None:
        """Add an analyzer to the analysis pipeline."""
        self.analyzers.append(analyzer)

    def _find_experiment_name(self, tags: list[str]) -> str | None:
        """Find experiment name from run tags.

        First tries exact match with experiment names, then falls back to
        matching by experiment type.

        Args:
            tags: List of tags from the run

        Returns:
            Experiment name if found, None otherwise
        """
        # Try exact match first
        for tag in tags:
            if tag in EXPERIMENT_REGISTRY:
                return tag

        # Fall back to matching by experiment type (e.g. resid_mlp, tms)
        experiment_type_map = {}
        for exp_name, exp_config in EXPERIMENT_REGISTRY.items():
            exp_type = exp_config.experiment_type
            if exp_type not in experiment_type_map:
                experiment_type_map[exp_type] = []
            experiment_type_map[exp_type].append(exp_name)

        for tag in tags:
            if tag in experiment_type_map:
                # Return the first experiment of this type
                return experiment_type_map[tag][0]

        return None

    def fetch_runs(self) -> list[EvalRunData]:
        """Fetch all W&B runs with the specified evals_id tag."""
        logger.info(f"Fetching runs for evals_id: {self.evals_id}")

        # Search for runs with the evals_id tag
        runs = self.wandb_api.runs(
            path="spd",  # W&B project name
            filters={"tags": {"$in": [f"evals_id-{self.evals_id}"]}},
        )

        logger.info(f"Found {len(runs)} runs, running analysis ...")
        eval_runs = []
        for run in runs:
            # Determine experiment name from tags
            experiment_name = self._find_experiment_name(run.tags)
            if not experiment_name:
                logger.warning(
                    f"Could not determine experiment name for run {run.id} with tags {run.tags}"
                )
                continue

            # Extract final faithfulness loss
            final_faithfulness_loss = None
            try:
                # Try the most efficient method first - get from summary
                final_faithfulness_loss = run.summary.get("loss/faithfulness")

                # If not in summary, fall back to scanning history
                if final_faithfulness_loss is None:
                    history = run.scan_history(keys=["loss/faithfulness"])
                    for row in history:
                        if "loss/faithfulness" in row:
                            final_faithfulness_loss = row["loss/faithfulness"]
            except Exception as e:
                logger.warning(f"Could not extract faithfulness loss for run {run.id}: {e}")

            eval_run = EvalRunData(
                run_id=run.id,
                experiment_name=experiment_name,
                experiment_type=EXPERIMENT_REGISTRY[experiment_name].experiment_type,
                state=run.state,
                final_faithfulness_loss=final_faithfulness_loss,
                run_name=run.name,
                tags=run.tags,
                url=run.url,
                config=run.config,
            )
            eval_runs.append(eval_run)

        logger.info(f"Found {len(eval_runs)} runs for analysis")
        return eval_runs

    def run_analysis(self) -> list[AnalysisResult]:
        """Run all configured analyzers on the fetched runs."""
        runs = self.fetch_runs()

        if not runs:
            logger.warning("No runs found for analysis")
            return []

        results = []
        for analyzer in self.analyzers:
            logger.info(f"Running {analyzer.name}")
            result = analyzer.analyze(runs)
            results.append(result)

            # Log result summary
            status = "✅" if result.success else "❌"
            logger.info(f"{status} {analyzer.name}: {result.summary}")

            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"{analyzer.name}: {warning}")

        return results

    def generate_combined_report(self, results: list[AnalysisResult]) -> str:
        """Generate a combined markdown report from all analysis results."""
        if not results:
            return "# Analysis Report\n\nNo analysis results available.\n"

        report = "# SPD Evaluation Analysis Report\n\n"
        report += f"**Evaluation ID:** `{self.evals_id}`  \n"
        report += f"**Generated:** {datetime.now().strftime('%a %b %d %H:%M:%S %Z %Y')}  \n\n"

        # Overall summary
        all_passed = all(result.success for result in results)
        overall_emoji = "✅" if all_passed else "❌"
        report += f"## {overall_emoji} Overall Status\n\n"

        for result in results:
            status_emoji = "✅" if result.success else "❌"
            report += f"- {status_emoji} **{result.analyzer_name}:** {result.summary}\n"

        report += "\n---\n\n"

        # Detailed results
        for result in results:
            report += result.markdown_report + "\n"

        return report

    def save_report(self, report: str) -> Path:
        """Save the analysis report to a file."""
        # TODO: Update to use the base directory for the repo, and not a temporary dir (REPO_ROOT
        # will point to a temporary dir if run in evals)
        output_file = Path.home() / f"spd/spd/evals/out/eval_analysis_{self.evals_id}.md"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report)
        logger.info(f"Analysis report saved to: {output_file}")
        return output_file

    def update_wandb_report(self, report: str) -> None:
        """Update the W&B report with analysis results."""
        # Clean the URL to remove query parameters that cause parsing issues
        clean_url = self.report_url.split("?")[0]  # Remove query params
        if clean_url.endswith("/edit"):
            clean_url = clean_url[:-5]  # Remove /edit suffix

        logger.info(f"Updating W&B report: {clean_url}")

        existing_report = wr.Report.from_url(clean_url)

        # Fix filter expressions if they have unquoted evals_id (bug in original creation)
        try:
            for block in existing_report.blocks:  # pyright: ignore[reportAttributeAccessIssue]
                if hasattr(block, "runsets"):
                    for runset in block.runsets:  # pyright: ignore[reportAttributeAccessIssue]
                        assert runset.filters is not None
                        if (
                            f"evals_id-{self.evals_id}" in runset.filters
                            and f'"evals_id-{self.evals_id}"' not in runset.filters
                        ):
                            # Fix unquoted evals_id in filters
                            logger.info(f"Fixing malformed filter in runset: {runset.name}")
                            runset.filters = runset.filters.replace(
                                f"evals_id-{self.evals_id}", f'"evals_id-{self.evals_id}"'
                            )
        except Exception as e:
            logger.warning(f"Could not fix filters: {e}. Continuing anyway.")

        # Create analysis block and add at the beginning
        analysis_block = wr.MarkdownBlock(text=report)

        existing_report.blocks.insert(0, analysis_block)  # pyright: ignore[reportAttributeAccessIssue]

        # Save the updated report
        existing_report.save()  # pyright: ignore[reportAttributeAccessIssue]
        logger.info("W&B report updated successfully with analysis results")


def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(description="Analyze SPD evaluation runs")
    parser.add_argument("evals_id", help="The evaluation ID to analyze")
    parser.add_argument("report_url", help="The W&B report URL to update")

    args = parser.parse_args()

    analyzer = EvalAnalyzer(args.evals_id, args.report_url)

    analyzer.add_analyzer(FaithfulnessAnalyzer())

    results = analyzer.run_analysis()

    report = analyzer.generate_combined_report(results)

    output_file = analyzer.save_report(report)

    analyzer.update_wandb_report(report)

    print(f"\n{'=' * 50}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 50}")
    print(f"Evaluation ID: {args.evals_id}")
    print(f"Report saved to: {output_file}")
    print(f"W&B report updated: {args.report_url}")


if __name__ == "__main__":
    main()
