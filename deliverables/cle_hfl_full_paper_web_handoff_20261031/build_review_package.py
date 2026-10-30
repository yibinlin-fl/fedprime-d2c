"""Build an explicit, read-only-source V0.11 web review package."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER = REPO / 'deliverables/cle_hfl_latex_v0_11_20261028'
WORKBENCH = REPO / 'deliverables/cle_hfl_figures_v1_20261028'
RETURN = REPO / 'local_runs/paper_web_review_returns/v0_11_20261031'
FILES = {
    'CLE_HFL_LATEX_V0_11_FIGURES_20261028.pdf': PAPER / 'CLE_HFL_LATEX_V0_11_FIGURES_20261028.pdf',
    'CLE_HFL_LATEX_V0_11_FIGURES_OVERLEAF_20261028.zip': PAPER / 'CLE_HFL_LATEX_V0_11_FIGURES_OVERLEAF_20261028.zip',
    'references.bib': PAPER / 'source/references.bib',
    'CITATION_AUDIT_ZH.md': PAPER / 'source/CITATION_AUDIT_ZH.md',
    'evidence/local_first_three_seeds.csv': REPO / 'deliverables/cle_v2_local_first_multiseed_20261026/PAPER_TABLE.csv',
    'evidence/main_evaluation_dose_and_method.csv': REPO / 'deliverables/cle_v2_moderate_axis2_dose_formal_seed0_20261025/PAPER_TABLE.csv',
    'figure_candidates/01_cle_paired_dsa.png': WORKBENCH / 'figures/tikz_cle_paired_dsa_v7.png',
    'figure_candidates/02_pew_ber_execution.png': WORKBENCH / 'figures/tikz_pew_ber_execution_v7.png',
    'figure_candidates/03_local_first.png': WORKBENCH / 'figures/fig_local_first_multiseed_v2.png',
    'figure_candidates/04_objective_tradeoff.png': WORKBENCH / 'figures/fig_objective_tradeoff_v3.png',
    'figure_candidates/05_dose_response.png': WORKBENCH / 'figures/fig_dose_response_v4.png',
}
AUTHORED = ['README_ZH.md', 'CLE_HFL_FULL_PAPER_WEB_HANDOFF_ZH.md',
            'CURRENT_EVIDENCE_AND_REVIEW_LOCKS_ZH.md', 'WEB_REVIEW_PROMPTS_ZH.md',
            'build_review_package.py', '.gitattributes']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RETURN.mkdir(parents=True, exist_ok=True)
    for name, source in FILES.items():
        target = HERE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != sha(source):
            raise FileExistsError(f'Preserving different existing file: {target}')
        if not target.exists():
            target.write_bytes(source.read_bytes())
        assert sha(target) == sha(source)
    with (HERE / 'evidence/local_first_three_seeds.csv').open(encoding='utf-8-sig', newline='') as stream:
        seeds = list(csv.DictReader(stream))
    assert {'0', '1', '2'}.issubset({r['train_seed'] for r in seeds})
    with (HERE / 'evidence/main_evaluation_dose_and_method.csv').open(encoding='utf-8-sig', newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 4
    names = sorted([*AUTHORED, *FILES])
    manifest = {
        'date': '2026-10-31', 'manuscript': 'V0.11 original bytes; candidate figures not integrated',
        'facts': 'No new experiments; scientific handoff inherited with current version/task corrections',
        'files': [{ 'path': name, 'bytes': (HERE/name).stat().st_size, 'sha256': sha(HERE/name),
                    **({'source': FILES[name].relative_to(REPO).as_posix()} if name in FILES else {}) }
                  for name in names],
    }
    manifest_path = HERE / 'PACKAGE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    archive = HERE / 'CLE_HFL_WEB_REVIEW_V0_11_20261031.zip'
    if archive.exists():
        raise FileExistsError('Preserving existing package; create a new dated revision')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in [*names, 'PACKAGE_MANIFEST.json']:
            bundle.write(HERE/name, arcname=name)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        for item in manifest['files']:
            assert hashlib.sha256(bundle.read(item['path'])).hexdigest() == item['sha256']
    audit = {'zip': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha(archive),
             'entries': len(names)+1, 'source_pdf_and_zip_unchanged': True,
             'all_explicit_source_copies_match': True, 'zip_crc_and_entry_sha256': 'PASS',
             'return_folder': str(RETURN)}
    (HERE/'PACKAGE_AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
