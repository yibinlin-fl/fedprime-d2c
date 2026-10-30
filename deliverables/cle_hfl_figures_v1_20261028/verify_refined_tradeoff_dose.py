"""Independent integrity and frozen-data checks for Figure 4/5 candidates."""
from pathlib import Path
import csv
import hashlib
import json
import math

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
FIGS = HERE / 'figures'
REPO = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-12, rel_tol=0), (a, b)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def metadata(stem):
    record = json.loads((FIGS / (stem + '.json')).read_text(encoding='utf-8'))
    for name, expected in record['outputs_sha256'].items():
        assert sha(FIGS / name) == expected, name
    assert sha(REPO / record['shared_helper']['path']) == record['shared_helper']['sha256']
    for entry in record['sources']:
        assert sha(REPO / entry['path']) == entry['sha256'], entry['path']
    pdf = PdfReader(FIGS / (stem + '.pdf'))
    assert len(pdf.pages) == 1
    assert abs(float(pdf.pages[0].mediabox.width) / 72 - 7.06) < 0.001
    assert (pdf.pages[0].extract_text() or '').strip(), 'Missing vector text'
    return record


def main():
    obj = metadata('fig_objective_tradeoff_v3')
    rows = read_csv(FIGS / 'fig_objective_tradeoff_v3.csv')
    assert rows == read_csv(FIGS / 'fig_objective_tradeoff_v2.csv')
    assert len(rows) == 12
    assert obj['axis_ranges'] == {'x': [17.2, 29.4], 'y': [0., 0.29]}
    for actual, row in zip(obj['plot_coordinates'], rows):
        assert actual['arm'] == row['arm'] and actual['train_seed'] == int(row['train_seed'])
        close(actual['dsa'], float(row['dsa']))
        close(actual['grid_accuracy'], float(row['grid_accuracy']))
    for arm, summary in obj['statistics'].items():
        for field, prefix in [('dsa', 'dsa'), ('grid_accuracy', 'accuracy')]:
            values = [float(r[field]) for r in rows if r['arm'] == arm]
            mean = sum(values) / 3
            sd = math.sqrt(sum((x-mean)**2 for x in values) / 2)
            close(summary[prefix+'_mean'], mean)
            close(summary[prefix+'_sd'], sd)
    dose = metadata('fig_dose_response_v4')
    rows = read_csv(FIGS / 'fig_dose_response_v4.csv')
    assert rows == read_csv(FIGS / 'fig_dose_response_v3.csv')
    erm = [r for r in rows if r['method'] != 'AsymHFL+PEW+BER']
    ber = next(r for r in rows if r['method'] == 'AsymHFL+PEW+BER')
    assert dose['plot_coordinates']['erm_gamma'] == [0., .5, .9]
    for actual, row in zip(dose['plot_coordinates']['erm_dsa'], erm):
        close(actual, float(row['dsa']))
    close(dose['plot_coordinates']['ber_dsa'], float(ber['dsa']))
    assert dose['plot_coordinates']['ber_gamma'] == .9
    for key, value in dose['utility_changes_pp'].items():
        close(value, float(ber[key]) - float(erm[2][key]))
    close(dose['dsa_reduction'], float(erm[2]['dsa']) - float(ber['dsa']))
    assert dose['axis_ranges'] == {'left_x': [-.05, 1.3], 'left_y': [-.015, .215],
                                  'right_x': [-6., 2.3], 'right_y': [-.65, 5.65]}
    # Every old revision and its archived generator retain their original bytes.
    for stem in ['fig_objective_tradeoff_v1', 'fig_objective_tradeoff_v2',
                 'fig_dose_response_v1', 'fig_dose_response_v2', 'fig_dose_response_v3']:
        record = json.loads((FIGS / (stem + '.json')).read_text(encoding='utf-8'))
        for name, expected in record['outputs_sha256'].items():
            assert sha(FIGS / name) == expected, 'Old revision changed: ' + name
    print('PASS: all 12 seed rows, four means/sample SDs, four dose points, six utility changes, unchanged axes, hashes, vector PDFs and five old revisions.')


if __name__ == '__main__':
    main()
