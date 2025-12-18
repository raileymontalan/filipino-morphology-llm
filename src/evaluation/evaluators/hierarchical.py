"""
Analysis tools for hierarchical task evaluation results.

Provides methods to:
1. Generate capability profiles showing performance at each level
2. Identify failure cascades (where Level N failure leads to Level N+1 failure)
3. Compare models to diagnose specific weaknesses
4. Visualize hierarchical performance
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd


class HierarchicalAnalyzer:
    """Analyzer for hierarchical task evaluation results."""

    def __init__(self, results_file: str):
        """
        Initialize analyzer with results.

        Args:
            results_file: Path to JSONL file with prediction results
                         Each line should have: level, category, subcategory,
                         correct (bool), predicted_answer, gold_answer
        """
        self.results = []
        with open(results_file) as f:
            for line in f:
                self.results.append(json.loads(line))

    def compute_capability_profile(self) -> pd.DataFrame:
        """
        Compute performance at each level.

        Returns:
            DataFrame with columns: level, accuracy, n_total, n_correct
        """
        level_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for result in self.results:
            level = result["level"]
            correct = result["correct"]

            level_stats[level]["total"] += 1
            if correct:
                level_stats[level]["correct"] += 1

        profile = []
        for level in sorted(level_stats.keys()):
            stats = level_stats[level]
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            profile.append(
                {
                    "level": level,
                    "accuracy": accuracy,
                    "n_correct": stats["correct"],
                    "n_total": stats["total"],
                }
            )

        return pd.DataFrame(profile)

    def compute_subcategory_profile(self) -> pd.DataFrame:
        """
        Compute performance for each subcategory within each level.

        Returns:
            DataFrame with columns: level, subcategory, accuracy, n_total
        """
        subcat_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for result in self.results:
            level = result["level"]
            subcategory = result["subcategory"]
            key = (level, subcategory)

            subcat_stats[key]["total"] += 1
            if result["correct"]:
                subcat_stats[key]["correct"] += 1

        profile = []
        for (level, subcategory), stats in subcat_stats.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            profile.append(
                {
                    "level": level,
                    "subcategory": subcategory,
                    "accuracy": accuracy,
                    "n_correct": stats["correct"],
                    "n_total": stats["total"],
                }
            )

        return pd.DataFrame(profile).sort_values(["level", "subcategory"])

    def identify_failure_cascades(self, threshold: float = 0.6) -> List[Tuple[int, int, float, float]]:
        """
        Identify level transitions where performance drops significantly.

        A cascade is detected when:
        - Level N performance < threshold
        - Level N+1 performance <= Level N performance

        Args:
            threshold: Accuracy threshold below which we consider failure

        Returns:
            List of (level_n, level_n+1, acc_n, acc_n+1) tuples
        """
        profile = self.compute_capability_profile()
        cascades = []

        for i in range(len(profile) - 1):
            level_n = profile.iloc[i]
            level_n_plus_1 = profile.iloc[i + 1]

            if level_n["accuracy"] < threshold and level_n_plus_1["accuracy"] <= level_n["accuracy"]:
                cascades.append(
                    (
                        int(level_n["level"]),
                        int(level_n_plus_1["level"]),
                        level_n["accuracy"],
                        level_n_plus_1["accuracy"],
                    )
                )

        return cascades

    def get_failure_examples(self, level: int, subcategory: str = None, n: int = 10) -> List[Dict]:
        """
        Get examples of failures at a specific level/subcategory.

        Args:
            level: Level to retrieve examples from
            subcategory: Optional subcategory filter
            n: Number of examples to return

        Returns:
            List of failure examples with full context
        """
        failures = []
        for result in self.results:
            if result["level"] == level and not result["correct"]:
                if subcategory is None or result["subcategory"] == subcategory:
                    failures.append(result)

        return failures[:n]

    def compute_level_dependencies(self) -> Dict[int, Dict[str, float]]:
        """
        Analyze which lower levels correlate with higher level performance.

        For each level, compute correlation between success at lower levels
        and success at this level (using per-example analysis).

        Returns:
            Dictionary: {level: {prerequisite_level: correlation}}
        """
        # Group results by example identifier if available
        # This is a placeholder - proper implementation needs per-example tracking
        dependencies = {}

        # For now, return simple accuracy ratios
        profile = self.compute_capability_profile()

        for i in range(1, len(profile)):
            level = int(profile.iloc[i]["level"])
            dependencies[level] = {}

            for j in range(i):
                prereq_level = int(profile.iloc[j]["level"])
                # Ratio of accuracies as proxy for dependency
                ratio = profile.iloc[i]["accuracy"] / (profile.iloc[j]["accuracy"] + 1e-6)
                dependencies[level][prereq_level] = ratio

        return dependencies

    def compare_models(self, other_analyzer: "HierarchicalAnalyzer") -> pd.DataFrame:
        """
        Compare this model's performance with another model.

        Args:
            other_analyzer: Another HierarchicalAnalyzer instance

        Returns:
            DataFrame with columns: level, model1_acc, model2_acc, diff, significant
        """
        profile1 = self.compute_capability_profile()
        profile2 = other_analyzer.compute_capability_profile()

        comparison = []
        for _, row1 in profile1.iterrows():
            level = row1["level"]
            row2 = profile2[profile2["level"] == level]

            if len(row2) == 0:
                continue

            row2 = row2.iloc[0]

            diff = row1["accuracy"] - row2["accuracy"]

            # Simple significance test: difference > 5% and enough samples
            significant = abs(diff) > 0.05 and min(row1["n_total"], row2["n_total"]) > 20

            comparison.append(
                {
                    "level": level,
                    "model1_acc": row1["accuracy"],
                    "model2_acc": row2["accuracy"],
                    "difference": diff,
                    "significant": significant,
                    "model1_n": row1["n_total"],
                    "model2_n": row2["n_total"],
                }
            )

        return pd.DataFrame(comparison)

    def generate_diagnostic_report(self) -> str:
        """
        Generate a human-readable diagnostic report.

        Returns:
            Formatted string with analysis
        """
        profile = self.compute_capability_profile()
        subcat_profile = self.compute_subcategory_profile()
        cascades = self.identify_failure_cascades()

        report = []
        report.append("=" * 60)
        report.append("HIERARCHICAL CAPABILITY PROFILE")
        report.append("=" * 60)
        report.append("")

        # Overall profile
        report.append("Performance by Level:")
        report.append("-" * 60)
        for _, row in profile.iterrows():
            bar = "█" * int(row["accuracy"] * 20)
            report.append(
                f"Level {int(row['level'])}: {row['accuracy']:.1%} {bar} ({row['n_correct']}/{row['n_total']})"
            )
        report.append("")

        # Identify strongest and weakest levels
        best_level = profile.loc[profile["accuracy"].idxmax()]
        worst_level = profile.loc[profile["accuracy"].idxmin()]

        report.append("Summary:")
        report.append(f"  ✓ Strongest: Level {int(best_level['level'])} ({best_level['accuracy']:.1%})")
        report.append(f"  ✗ Weakest: Level {int(worst_level['level'])} ({worst_level['accuracy']:.1%})")
        report.append("")

        # Failure cascades
        if cascades:
            report.append("Failure Cascades Detected:")
            report.append("-" * 60)
            for level_n, level_n_plus_1, acc_n, acc_n_plus_1 in cascades:
                report.append(f"  Level {level_n} ({acc_n:.1%}) → Level {level_n_plus_1} ({acc_n_plus_1:.1%})")
            report.append("")
            report.append("Interpretation: Performance drops indicate capability bottlenecks.")
            report.append("")

        # Subcategory breakdown for weakest level
        report.append(f"Detailed Breakdown - Level {int(worst_level['level'])}:")
        report.append("-" * 60)
        worst_level_subcats = subcat_profile[subcat_profile["level"] == worst_level["level"]]
        for _, row in worst_level_subcats.iterrows():
            report.append(f"  {row['subcategory']}: {row['accuracy']:.1%} ({row['n_correct']}/{row['n_total']})")
        report.append("")

        # Recommendations
        report.append("Recommendations:")
        report.append("-" * 60)
        if worst_level["level"] == 0:
            report.append("  → Model struggles with basic character recognition")
            report.append("  → Tokenization may be too coarse-grained")
        elif worst_level["level"] == 1:
            report.append("  → Model can recognize but not manipulate characters")
            report.append("  → May need better positional understanding")
        elif worst_level["level"] == 2:
            report.append("  → Model struggles with morphological decomposition")
            report.append("  → Tokenization does not align with morpheme boundaries")
            report.append("  → Consider affix-aware tokenization (Patok)")
        elif worst_level["level"] >= 3:
            report.append("  → Model understands morphology but cannot manipulate it")
            report.append("  → May need more morphologically diverse training data")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)

    def export_for_visualization(self, output_file: str):
        """
        Export results in format suitable for visualization.

        Creates a JSON file with:
        - Overall profile
        - Subcategory breakdown
        - Example failures
        """
        profile = self.compute_capability_profile()
        subcat_profile = self.compute_subcategory_profile()

        export_data = {
            "overall_profile": profile.to_dict(orient="records"),
            "subcategory_profile": subcat_profile.to_dict(orient="records"),
            "failure_examples": {},
        }

        # Add failure examples for each level
        for level in sorted(set(r["level"] for r in self.results)):
            export_data["failure_examples"][level] = self.get_failure_examples(level, n=5)

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported visualization data to {output_file}")


def compare_multiple_models(analyzers: Dict[str, HierarchicalAnalyzer]) -> pd.DataFrame:
    """
    Compare multiple models on hierarchical tasks.

    Args:
        analyzers: Dictionary of {model_name: analyzer}

    Returns:
        DataFrame with columns: level, model1_acc, model2_acc, ..., best_model
    """
    all_profiles = {}
    for model_name, analyzer in analyzers.items():
        profile = analyzer.compute_capability_profile()
        all_profiles[model_name] = profile.set_index("level")["accuracy"]

    comparison_df = pd.DataFrame(all_profiles)
    comparison_df["best_model"] = comparison_df.idxmax(axis=1)
    comparison_df["worst_model"] = comparison_df.idxmin(axis=1)
    comparison_df["range"] = comparison_df.max(axis=1) - comparison_df.min(axis=1)

    return comparison_df.reset_index()


def visualize_capability_profile(profile_df: pd.DataFrame, title: str = "Capability Profile"):
    """
    Create a simple text-based visualization of capability profile.

    Args:
        profile_df: DataFrame from compute_capability_profile()
        title: Plot title
    """
    print(f"\n{title}")
    print("=" * 60)
    print("Level  Accuracy")
    print("-" * 60)

    for _, row in profile_df.iterrows():
        level = int(row["level"])
        accuracy = row["accuracy"]
        bar_length = int(accuracy * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        print(f"  {level}    {bar} {accuracy:.1%}")

    print("=" * 60)
