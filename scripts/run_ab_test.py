"""Run A/B test comparing different embedding models.

Usage:
    # Run all variants defined in config/ab_variants/
    uv run scripts/run_ab_test.py

    # Run specific variants by passing their YAML config files
    uv run scripts/run_ab_test.py config/ab_variants/mpnet_baseline.yaml config/ab_variants/bge_large.yaml

    # Pass a directory to run all variants in it
    uv run scripts/run_ab_test.py config/ab_variants/
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ab_testing import (
    ABExperiment,
    ExperimentConfig,
)
from src.ab_testing.variant import VariantConfig
from src.evaluation import ABTestReportGenerator
from src.observability.tracing import init_tracing

init_tracing(enable_console=False)

DEFAULT_VARIANTS_DIR = Path(__file__).parent.parent / "config" / "ab_variants"


def load_variant_configs(paths: list[str]) -> tuple[list[VariantConfig], list[Path]]:
    """Load VariantConfig objects from a list of file/directory paths.

    Args:
        paths: List of YAML file paths or directories containing YAML files

    Returns:
        Tuple of (configs, yaml_paths)
    """
    yaml_paths: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            yaml_paths.extend(sorted(p.glob("*.yaml")))
        elif p.is_file():
            yaml_paths.append(p)
        else:
            print(f"Warning: '{p}' is not a file or directory, skipping")

    if not yaml_paths:
        print(f"No variant YAML files found. Put configs in {DEFAULT_VARIANTS_DIR}/")
        sys.exit(1)

    configs: list[VariantConfig] = []
    for path in yaml_paths:
        print(f"  Loading variant: {path.name}")
        configs.append(VariantConfig.from_yaml(path))
    return configs, yaml_paths


def _archive_results(
    variant_yaml_paths: list[Path],
    variants: list[VariantConfig],
    output_dir: Path,
    experiment_name: str,
) -> Path:
    """Zip the result files and variant configs into a timestamped archive.

    Args:
        variant_yaml_paths: YAML config files that were used in the run
        variants: Loaded variant configs (for naming)
        output_dir: Directory containing the result files
        experiment_name: Experiment name used as the base for result filenames

    Returns:
        Path to the created zip file
    """
    archive_dir = output_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant_names = "_".join(v.name for v in variants)
    zip_name = f"{variant_names}_{timestamp}.zip"
    zip_path = archive_dir / zip_name

    result_files = [
        output_dir / f"{experiment_name}_results.json",
        output_dir / f"{experiment_name}_report.json",
        output_dir / f"{experiment_name}_report.md",
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for result_file in result_files:
            if result_file.exists():
                zf.write(result_file, result_file.name)

        for yaml_path in variant_yaml_paths:
            zf.write(yaml_path, f"configs/{yaml_path.name}")

    return zip_path


def main() -> None:
    """Run A/B test experiment."""
    parser = argparse.ArgumentParser(
        description="Run A/B test comparing embedding models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "variants",
        nargs="*",
        help=(
            "Variant YAML config files or directories. "
            f"Defaults to all files in {DEFAULT_VARIANTS_DIR}/"
        ),
    )
    args = parser.parse_args()

    # Default: load all configs from config/ab_variants/
    variant_paths = args.variants if args.variants else [str(DEFAULT_VARIANTS_DIR)]

    print("Loading variant configs...")
    variants, yaml_paths = load_variant_configs(variant_paths)
    print(f"Loaded {len(variants)} variant(s): {[v.name for v in variants]}")
    print()

    # Create experiment configuration
    config = ExperimentConfig(
        name="embedding_model_comparison_2026",
        description="Comparison of embedding models for metadata retrieval quality",
        variants=variants,
        gold_dataset_path="tests/fixtures/gold_dataset_review.json",
        vector_store_path="data/ab_testing/vector_stores",
        metadata={
            "objective": "Determine which embedding model gives best retrieval quality",
            "hypothesis": "BGE Large will improve P@5 by at least 10% over baseline",
        },
    )

    # Check gold dataset
    gold_path = Path(config.gold_dataset_path)
    if not gold_path.exists():
        print(f"Error: Gold dataset not found at {gold_path}")
        print("Run scripts/build_gold_dataset.py first to create it.")
        sys.exit(1)

    print(f"Running experiment: {config.name}")
    print(f"Description: {config.description}")
    print(f"Variants: {len(config.variants)}")
    print()

    # Create and run experiment
    experiment = ABExperiment(config)

    print("Setting up experiment (may download models and re-index data)...")
    experiment.setup()
    print()

    print("Running evaluation queries...")
    results = experiment.run()

    print()
    print("=== Experiment Results ===")
    print()

    for variant_name, result in results.items():
        print(f"Variant: {variant_name}")
        print(f"  P@5: {result.metrics.precision_at_k[5]:.4f}")
        print(f"  R@5: {result.metrics.recall_at_k[5]:.4f}")
        print(f"  MRR: {result.metrics.mrr:.4f}")
        print(f"  Mean Latency: {result.latency_stats['mean_ms']:.2f}ms")
        print()

    # Save results
    output_dir = Path("data/ab_testing/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{config.name}_results.json"
    experiment.save_results(results, results_path)

    # Generate report
    experiment_data = json.loads(results_path.read_text())
    ABTestReportGenerator.generate_experiment_report(
        experiment_results=experiment_data,
        output_dir=output_dir,
        format="both",
    )

    # Archive results + configs
    zip_path = _archive_results(yaml_paths, variants, output_dir, config.name)
    print(f"Results saved to: {results_path}")
    print(f"Report generated in: {output_dir}")
    print(f"Archived to: {zip_path}")

    # Cleanup variant collections (skips existing/baseline indexes)
    print()
    print("Cleaning up experiment resources...")
    experiment.cleanup()
    print("Done!")


if __name__ == "__main__":
    main()
