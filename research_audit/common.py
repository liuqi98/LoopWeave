"""Small, strict input/output helpers shared by offline analyses."""
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Empty input: {path}')
    return rows


def write_json(path, value):
    """Refuse to replace an existing report, including on accidental reruns."""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(payload)


def finite(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'Nonfinite value: {label}')
    return result


def metric_spec(spec):
    name, direction = spec.rsplit(':', 1)
    if not name or direction not in ('min', 'max'):
        raise ValueError('Use metric:min or metric:max, e.g. aligned_rmsd:min')
    return name, -1 if direction == 'min' else 1


def membership(groups):
    result = {}
    for group, samples in groups.items():
        if not samples:
            raise ValueError(f'Empty dependence group: {group}')
        for sid in samples:
            if sid in result:
                raise ValueError(f'Duplicate group membership: {sid}')
            result[sid] = group
    return result
