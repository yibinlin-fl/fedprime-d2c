"""Measure existing diagram geometry without editing or regenerating figures."""
from pathlib import Path
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / 'figures'
STEMS = ['fig_cle_paired_dsa_v4', 'fig_pew_ber_execution_v3',
         'fig_local_first_multiseed_v1', 'fig_objective_tradeoff_v2', 'fig_dose_response_v3']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()

def main():
    # Both diagram generators use the default subplot rectangle in a 9x3.75in
    # figure, unit data extents, and aspect='auto'. Measure the actual transform.
    fig, ax = plt.subplots(figsize=(9.0, 3.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.canvas.draw()
    origin = ax.transData.transform((0, 0))
    unit = ax.transData.transform((1, 1)) - origin
    ratio = float(unit[0] / unit[1])
    definitions = [
        ('fig1_paired_32x32', .070, .174),
        ('fig2_public_source_32x32', .072, .145),
        ('fig2_public_variants_32x32', .063, .126),
        ('fig2_private_32x32', .045, .105),
        ('fig2_circle_patch', 1, 1),
    ]
    shapes = []
    for name, w, h in definitions:
        rendered = ratio * w / h
        shapes.append({'element': name, 'display_width_over_height': rendered,
                       'relative_to_square_percent': (rendered - 1) * 100})
    plt.close(fig)
    figures = []
    for stem, factor in zip(STEMS, [1, 1, .96, .94, .96]):
        svg = (FIGS / (stem + '.svg')).read_text(encoding='utf-8')
        dimensions = re.search(r'<svg[^>]*width="([\d.]+)pt"[^>]*height="([\d.]+)pt"', svg)
        native_width = float(dimensions.group(1)) / 72
        figures.append({'stem': stem, 'native_width_in': native_width,
                        'manuscript_width_in': 7.06 * factor,
                        'font_scale_at_inclusion': 7.06 * factor / native_width,
                        'sha256': {ext: sha(FIGS / (stem + ext)) for ext in ['.pdf', '.svg', '.png']}})
    report = {'date': '2026-10-30', 'mode': 'audit_only_no_figure_edits',
              'axes_x_to_y_pixel_scale': ratio, 'geometry': shapes, 'figures': figures,
              'rotation_check': 'No rotate/rotation transform calls in the two diagram generators; this does not establish the pose of objects in source images',
              'matplotlib': matplotlib.__version__}
    (ROOT / 'GEOMETRY_AUDIT_20261030.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'geometry': shapes, 'font_scales': [r['font_scale_at_inclusion'] for r in figures]}, indent=2))

if __name__ == '__main__':
    main()
