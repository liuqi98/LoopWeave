"""Split a multi-arm score CSV into reference-vs-each two-arm CSVs.

``research_audit paired`` requires exactly two arms, so multi-arm evidence needs
explicit subsets. Subsets are written next to the input and never overwrite.
"""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--reference', default='full')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    directory = args.output_dir or args.input.parent
    directory.mkdir(parents=True, exist_ok=True)
    with args.input.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    if not rows:
        raise ValueError(f'Empty score table: {args.input}')
    arms = sorted({r['arm'] for r in rows})
    if args.reference not in arms:
        raise ValueError(f'Reference arm {args.reference} not in {arms}')
    written = []
    for arm in arms:
        if arm == args.reference:
            continue
        subset = [r for r in rows if r['arm'] in (args.reference, arm)]
        target = directory / f'paired_{args.reference}_vs_{arm}.csv'
        with target.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)
        written.append(str(target))
    print('\n'.join(written))


if __name__ == '__main__':
    main()
