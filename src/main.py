"""
Section-Aware Multi-Agent Resume Screening and Skill Mining System

Main entry point for running the full pipeline or individual stages.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Section-Aware Multi-Agent Resume Screening Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["parsing", "features", "mining", "classification", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/",
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--output-path",
        default="results/",
        help="Path to output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stage in ("parsing", "all"):
        print("Stage 1: Parsing resumes and extracting sections...")
        # TODO: Implement HTML parsing and section extraction

    if args.stage in ("features", "all"):
        print("Stage 2: Running section agents and extracting features...")
        # TODO: Implement section-aware feature extraction

    if args.stage in ("mining", "all"):
        print("Stage 3: Mining patterns (clustering, association rules, co-occurrence)...")
        # TODO: Implement clustering, association rules, network analysis

    if args.stage in ("classification", "all"):
        print("Stage 4: Running classification baselines...")
        # TODO: Implement classification and evaluation

    print("Done.")


if __name__ == "__main__":
    main()
