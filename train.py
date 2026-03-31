"""Training entrypoint.

This file delegates the core training logic to :mod:`train_module` so the training
loop can be imported and reused by other scripts (e.g., for hyperparameter sweeps
or unit tests).
"""

import argparse

from train_module import run_training


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train defect classification model.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--summary-only', action='store_true', help='Print only per-epoch loss + defect-type accuracy lines.')
    args = parser.parse_args()

    run_training(num_epochs=args.epochs, summary_only=args.summary_only)
