"""CLI for offline audits; never starts training, inference or parameter updates."""
import argparse
import csv
from pathlib import Path
from .common import read_csv, read_json, write_json, digest
from .protocols import audit_history, collect_metrics
from .statistics import factorial, paired_comparison
from .diagnostics import contacts, drift, cost_summary, baseline_comparison, export_distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('audit-history', help='Read original full-data plans, split IDs and epoch logs')
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    p.add_argument('--verify-artifacts', action='store_true', help='Also hash existing checkpoint/cache files; never deserialize weights')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('collect', help='Convert locked per-complex score files into a common CSV')
    p.add_argument('--input', type=Path, required=True, help='Score import manifest JSON')
    p.add_argument('--output', type=Path, required=True)
    for name in ('factorial', 'paired', 'baselines'):
        p = sub.add_parser(name)
        p.add_argument('--input', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--metric', action='append', required=True, help='metric:min or metric:max; may repeat')
        p.add_argument('--replicates', type=int, default=10000)
        if name == 'factorial':
            p.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
            p.add_argument('--protocol-audit', type=Path)
        else:
            p.add_argument('--reference', required=True)
            p.add_argument('--comparator', required=True)
            if name == 'paired':
                p.add_argument('--long-min', type=int, default=16)
    p = sub.add_parser('contacts', help='Score stored distance matrices, optionally at consecutive stages')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--thresholds', type=float, nargs='+', default=[6, 7, 8, 9, 10])
    p = sub.add_parser('export-distances', help='Read already saved backbones and trusted local evaluation tensors')
    p.add_argument('--input', type=Path, required=True, help='Prediction import manifest JSON')
    p.add_argument('--targets', type=Path, required=True, help='Trusted local .pt evaluation dataset, never a checkpoint')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('drift', help='Measure already exported numerical repeats; does not rerun a model')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--metric', action='append', required=True)
    p = sub.add_parser('costs')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'audit-history':
            result = audit_history(args.workspace, args.seeds, args.verify_artifacts)
        elif args.command == 'collect':
            rows = collect_metrics(args.input)
            fields = list(dict.fromkeys(k for r in rows for k in r))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            print(f'Imported {len(rows)} scored observations to {args.output}')
            return
        elif args.command == 'factorial':
            result = factorial(read_csv(args.input), args.metric, args.seeds, args.replicates)
            if args.protocol_audit:
                audit = read_json(args.protocol_audit)
                if not audit.get('complete'):
                    raise ValueError('Protocol audit is incomplete; cannot attach it as evidence for the four-arm analysis')
                result['protocol_audit_sha256'] = digest(args.protocol_audit)
                result['protocol_audit_note'] = 'Attached audit is contextual evidence; this tool does not certify runtime training equivalence.'
        elif args.command in ('paired', 'baselines'):
            method = paired_comparison if args.command == 'paired' else baseline_comparison
            extra = {'long_min': args.long_min} if args.command == 'paired' else {}
            result = method(read_csv(args.input), args.metric, args.reference, args.comparator,
                            args.replicates, **extra)
        elif args.command == 'contacts':
            result = contacts(read_json(args.input), args.thresholds)
        elif args.command == 'export-distances':
            import torch
            manifest = read_json(args.input)
            if manifest.get('targets_sha256') != digest(args.targets):
                raise ValueError('Missing/mismatched evaluation tensor hash')
            targets = torch.load(args.targets, map_location='cpu', weights_only=False)
            result = export_distances(targets, manifest, args.input.resolve().parent)
            write_json(args.output, result)
            print(f'Exported {len(result)} saved prediction/stage records to {args.output}')
            return
        elif args.command == 'drift':
            result = drift(read_json(args.input), args.metric)
        else:
            result = cost_summary(read_csv(args.input))
        if hasattr(args, 'input'):
            result['input_sha256'] = digest(args.input)
        result['analysis_code_sha256'] = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        write_json(args.output, result)
        print(f'Wrote {args.output}')
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
