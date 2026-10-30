"""Check the refined Figure-3 coordinates and provenance against frozen data."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
FIGS = HERE / 'figures'
SOURCE = HERE.parent / 'cle_v2_local_first_multiseed_20261026' / 'PAPER_TABLE.csv'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', default='v2')
    revision = parser.parse_args().revision
    if not revision.startswith('v') or not revision[1:].isdigit():
        raise ValueError('Invalid revision')
    stem = FIGS / ('fig_local_first_multiseed_' + revision)
    record = json.loads(stem.with_suffix('.json').read_text(encoding='utf-8'))
    assert record['source_sha256'] == sha(SOURCE)
    for name, expected in record['outputs_sha256'].items():
        assert sha(FIGS / name) == expected, name
    rows = list(csv.DictReader(SOURCE.open(encoding='utf-8-sig', newline='')))
    seeds = [row for row in rows if row['train_seed'] in {'0', '1', '2'}]
    exported = list(csv.DictReader(stem.with_suffix('.csv').open(encoding='utf-8-sig', newline='')))
    assert seeds == exported, 'Exported scientific rows changed'
    assert stem.with_suffix('.csv').read_bytes() == (FIGS / 'fig_local_first_multiseed_v1.csv').read_bytes()
    for key, field in [('local', 'delta_local'), ('hfl', 'delta_hfl'), ('comm', 'delta_comm')]:
        expected = [float(row[field]) for row in seeds]
        actual = record['plot_coordinates'][key + '_x']
        assert len(actual) == 3
        assert all(math.isclose(a, b, abs_tol=2e-10) for a, b in zip(actual, expected)), key
        mean = sum(expected) / 3
        sd = math.sqrt(sum((x-mean)**2 for x in expected) / 2)
        assert math.isclose(record['summary_statistics'][key]['mean'], mean, abs_tol=2e-10)
        assert math.isclose(record['summary_statistics'][key]['sample_sd'], sd, abs_tol=2e-10)
    assert record['plot_coordinates']['left_xlim'] == [0.0, 0.16]
    assert record['plot_coordinates']['right_xlim'] == [-0.0205, 0.023]
    assert record['plot_coordinates']['comm_y'] == [2, 1, 0]
    assert record['seed_2_original_verdict'] == seeds[2]['verdict']
    assert (FIGS / 'fig_local_first_multiseed_v1.py').read_bytes() == (FIGS / 'generate_local_first_multiseed.py').read_bytes()
    original = json.loads((FIGS / 'fig_local_first_multiseed_v1.json').read_text())
    for name, expected in original['outputs_sha256'].items():
        assert sha(FIGS / name) == expected, 'Original artifact changed: ' + name
    print('PASS: nine x-coordinates, both x-scales, mean/sample SD, exact CSV, seed-2 verdict, output hashes, original v1 artifacts.')

if __name__ == '__main__':
    main()
