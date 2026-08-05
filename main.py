import argparse

from config import CONFIG
from data_reduction.runner import run_all, run_method


def parse_args():
    parser = argparse.ArgumentParser(description="Data reduction methods runner")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["all", "a", "b", "c", "d"],
        default="all",
        help="Method to run (default: all)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == "all":
        run_all(CONFIG)
    else:
        run_method(args.mode, CONFIG)

if __name__ == "__main__":
    main()