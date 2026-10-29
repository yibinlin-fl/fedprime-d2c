"""Compile immutable TikZ revisions, render PNG, and export true vector SVG."""
from pathlib import Path
import argparse
import base64
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
FIGS = HERE / 'figures'
REPO = HERE.parents[1]
BUILD = REPO / 'local_runs' / 'tikz_review_20261030'
TEX = Path('D:/texlive/2026/bin/windows')

def run(args):
    result = subprocess.run([str(x) for x in args], cwd=FIGS, capture_output=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError('\n'.join((result.stdout + result.stderr).splitlines()[-25:]))
    return result

def embedded_svg(source, target):
    """Embed source PNG bytes; do not resample or change vector coordinates."""
    svg = source.read_text(encoding='utf-8')
    def replace(match):
        image = (FIGS / match.group(1)).resolve()
        if image.parent != (FIGS / 'assets_20261030').resolve():
            raise ValueError('Unexpected SVG bitmap path')
        return "xlink:href='data:image/png;base64," + base64.b64encode(image.read_bytes()).decode('ascii') + "'"
    svg = re.sub(r"xlink:href='(assets_20261030/[^']+\.png)'", replace, svg)
    target.write_text(svg, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stem', action='append')
    args = parser.parse_args()
    stems = args.stem or ['tikz_pew_ber_execution_v7', 'tikz_cle_paired_dsa_v7']
    BUILD.mkdir(exist_ok=True, parents=True)
    for stem in stems:
        if not stem.replace('_', '').isalnum() or not (FIGS / (stem + '.tex')).is_file():
            raise ValueError('Expected an existing named TikZ source')
        for ext in ['.pdf', '.png', '.svg']:
            if (FIGS / (stem + ext)).exists():
                raise FileExistsError('Existing revision is immutable: ' + stem + ext)
        options = ['-interaction=nonstopmode', '-halt-on-error', '-file-line-error', '-output-directory=' + str(BUILD)]
        run([TEX / 'pdflatex.exe', *options, stem + '.tex'])
        log = (BUILD / (stem + '.log')).read_text(encoding='utf-8', errors='replace')
        if 'Overfull' in log or 'Undefined control sequence' in log:
            raise ValueError('Compilation diagnostics require review: ' + stem)
        shutil.copyfile(BUILD / (stem + '.pdf'), FIGS / (stem + '.pdf'))
        run(['pdftoppm', '-singlefile', '-r', '200', '-png', FIGS / (stem + '.pdf'), FIGS / stem])
        # PDF→SVG requires an unavailable compatible Ghostscript. The native
        # PGF dvisvgm backend preserves the vector diagram and embeds bitmaps.
        driver = r'\def\pgfsysdriver{pgfsys-dvisvgm.def}\PassOptionsToPackage{dvisvgm}{graphicx}\input{' + stem + '.tex}'
        job = 'svg_' + stem
        run([TEX / 'latex.exe', *options, '-jobname=' + job, driver])
        raw_svg = BUILD / (stem + '_linked.svg')
        run([TEX / 'dvisvgm.exe', '--embed-bitmaps', '--no-fonts', '--output=' + str(raw_svg), BUILD / (job + '.dvi')])
        embedded_svg(raw_svg, FIGS / (stem + '.svg'))
        print(stem + ': PDF/PNG/SVG compiled; old revisions preserved.')

if __name__ == '__main__':
    main()
