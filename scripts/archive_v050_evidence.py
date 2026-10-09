"""Seal only candidate integration evidence, with full member readback."""
import argparse
from pathlib import Path
from scripts.run_v050_verification import archive, put

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cloud', type=Path, required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    put(args.out / 'archive-receipt.json', archive(args.out, args.cloud, args.source))
