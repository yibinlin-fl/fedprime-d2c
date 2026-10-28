"""Audit this immutable figure-integration snapshot and build its Overleaf ZIP."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = ROOT.parent / 'cle_hfl_latex_v0_10_20261027' / 'source'
SRC = ROOT / 'source'
QA = REPO / 'local_runs' / 'paper_qa_v0_11_20261028' / 'build'
SELECTED = [
    'fig_cle_paired_dsa_v4.pdf', 'fig_pew_ber_execution_v3.pdf',
    'fig_local_first_multiseed_v1.pdf', 'fig_objective_tradeoff_v2.pdf',
    'fig_dose_response_v3.pdf',
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()

def main():
    from pypdf import PdfReader
    unchanged = ['references.bib', 'main.tex'] + [
        f'sections/{name}.tex' for name in
        ['abstract', 'introduction', 'related_work', 'discussion', 'conclusion', 'appendix']
    ]
    for name in unchanged:
        assert sha(SRC / name) == sha(OLD / name), name
    table_paths = sorted((OLD / 'tables').rglob('*.tex'))
    for path in table_paths:
        rel = path.relative_to(OLD)
        old, new = path.read_text(encoding='utf-8'), (SRC / rel).read_text(encoding='utf-8')
        if path.name == 'seed_uncertainty.tex':
            old = old.replace(r'\begin{tabular}{llll}', '__BEGIN__').replace(r'\end{tabular}', '__END__')
            new = new.replace(r'\begin{tabularx}{\textwidth}{@{}p{0.17\textwidth}p{0.18\textwidth}X p{0.23\textwidth}@{}}', '__BEGIN__').replace(r'\end{tabularx}', '__END__')
        assert old == new, str(rel)
    texts = []
    for base in [OLD, SRC]:
        texts.append('\n'.join((base / 'sections' / f'{n}.tex').read_text(encoding='utf-8') for n in ['method', 'experiments']))
    math = lambda t: Counter(re.sub(r'\s+', '', s) for s in re.findall(r'\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)', t))
    assert math(texts[0]) == math(texts[1]), 'Protected inline/display math differs'
    cites = lambda t: Counter(re.findall(r'\\cite\w*\{([^}]+)\}', t))
    assert cites(texts[0]) == cites(texts[1]), 'Citation keys differ'
    for name in SELECTED:
        assert sha(SRC / 'figures' / name) == sha(ROOT.parent / 'cle_hfl_figures_v1_20261028' / 'figures' / name)
    tex = '\n'.join(p.read_text(encoding='utf-8') for p in (SRC / 'sections').glob('*.tex'))
    assert tex.count(r'\begin{figure*}') == 5
    log = (QA / 'main.log').read_text(encoding='utf-8', errors='replace')
    assert not re.search(r'Overfull \\hbox|undefined references|multiply defined|Undefined control sequence', log)
    reader = PdfReader(QA / 'main.pdf')
    pages = {str(i + 1): [n for n in range(1, 6) if f'Figure {n}:' in (p.extract_text() or '')] for i, p in enumerate(reader.pages)}
    pdf = ROOT / 'CLE_HFL_LATEX_V0_11_FIGURES_20261028.pdf'
    pdf.write_bytes((QA / 'main.pdf').read_bytes())
    archive = ROOT / 'CLE_HFL_LATEX_V0_11_FIGURES_OVERLEAF_20261028.zip'
    package_files = [p for p in SRC.rglob('*') if p.is_file() and (
        p.suffix.lower() in {'.tex', '.bib', '.sty', '.cls', '.md'} or
        p.parent == SRC / 'figures' and p.name in SELECTED)]
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package_files):
            z.write(p, p.relative_to(SRC).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and 'main.tex' in z.namelist()
    report = {
        'source_version': 'V0.10', 'integration_version': 'V0.11',
        'unchanged_files_checked': unchanged, 'all_tables_content_unchanged': True,
        'method_experiment_math_multiset_unchanged': True, 'citation_keys_unchanged': True,
        'figures_equal_workbench_pdfs': {n: sha(SRC / 'figures' / n) for n in SELECTED},
        'page_count': len(reader.pages), 'figure_pages': pages,
        'overfull_or_unresolved_references': False,
        'pdf': {'name': pdf.name, 'bytes': pdf.stat().st_size, 'sha256': sha(pdf)},
        'zip': {'name': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha(archive), 'file_count': len(package_files)},
    }
    (ROOT / 'RELEASE_AUDIT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
