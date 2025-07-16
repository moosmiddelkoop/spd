#!/usr/bin/env python3
"""Generate all plots for the Computation in Superposition paper from a trained model."""

import argparse
import sys
from pathlib import Path

from spd.experiments.cis.models import CISModel
from spd.experiments.cis.plotting import (
    create_raw_weights_heatmap,
    create_stacked_weight_plot,
)
from spd.log import logger


def main():
    parser = argparse.ArgumentParser(description="Generate CIS plots from trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (WandB format or local path)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./spd/experiments/cis/cis_plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["stacked", "heatmap"],
        choices=["stacked", "heatmap"],
        help="Which plots to generate",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        sys.exit(1)

    logger.info(f"Loading CIS model from: {args.model_path}")

    # Load the trained model
    try:
        model, config_dict = CISModel.from_pretrained(args.model_path)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model from {args.model_path}: {e}")
        sys.exit(1)

    required_keys = ["cis_model_config", "feature_sparsity", "importance_decay"]
    missing_keys = [key for key in required_keys if key not in config_dict]
    if missing_keys:
        logger.error(f"Missing required config keys: {missing_keys}")
        sys.exit(1)

    logger.info(f"Model config: {config_dict['cis_model_config']}")
    logger.info(
        f"Training config: sparsity={config_dict['feature_sparsity']}, decay={config_dict['importance_decay']}"
    )

    # Generate plots
    if "stacked" in args.plots:
        logger.info("Generating stacked weight plot...")
        try:
            create_stacked_weight_plot(
                model=model,
                filepath=output_dir / "stacked_weights.png",
                sort_by_importance=True,
            )
            logger.info(f"Saved: {output_dir / 'stacked_weights.png'}")
        except Exception as e:
            logger.error(f"Failed to generate stacked weight plot: {e}")

    if "heatmap" in args.plots:
        logger.info("Generating raw weights heatmaps...")
        try:
            create_raw_weights_heatmap(
                model=model,
                filepath=output_dir / "raw_weights_W1.png",
                layer="W1",
            )
            logger.info(f"Saved: {output_dir / 'raw_weights_W1.png'}")
        except Exception as e:
            logger.error(f"Failed to generate W1 heatmap: {e}")

        try:
            create_raw_weights_heatmap(
                model=model,
                filepath=output_dir / "raw_weights_W2.png",
                layer="W2",
            )
            logger.info(f"Saved: {output_dir / 'raw_weights_W2.png'}")
        except Exception as e:
            logger.error(f"Failed to generate W2 heatmap: {e}")

    logger.info(f"Plot generation complete. Results saved to: {output_dir}")
    logger.info("Plot descriptions:")
    logger.info(
        "- stacked_weights.png: Main stacked weight plot showing feature representation in neurons"
    )
    logger.info("- raw_weights_W1.png: Heatmap of W1 (input to hidden) weights")
    logger.info("- raw_weights_W2.png: Heatmap of W2 (hidden to output) weights")


if __name__ == "__main__":
    main()
