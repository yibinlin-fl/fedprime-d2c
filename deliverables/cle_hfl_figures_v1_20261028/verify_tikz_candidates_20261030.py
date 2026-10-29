"""Verify native PDF image geometry, source pixels, and revision provenance."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import xml.etree.ElementTree as ET

from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / 'figures'
ASSETS = FIGS / 'assets_20261030'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()

def pixels(image):
    return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()

def main():
    definitions = {
        'tikz_pew_ber_execution_v7': ['public_clean', 'public_noise', 'public_blur', 'public_digital', 'private_0', 'private_1'],
        'tikz_cle_paired_dsa_v7': ['paired_gaussian_noise', 'paired_defocus_blur', 'paired_snow', 'paired_pixelate'],
    }
    asset_manifest = json.loads((ASSETS / 'ASSET_MANIFEST.json').read_text())
    for row in asset_manifest['images']:
        assert hashlib.sha256((ASSETS / row['path']).read_bytes()).hexdigest() == row['sha256']
    for stem, expected in definitions.items():
        page = PdfReader(FIGS / (stem + '.pdf')).pages[0]
        objects = page['/Resources']['/XObject'].get_object()
        geometry = []
        def visitor(operator, args, matrix, textmatrix):
            if operator != b'Do':
                return
            obj = objects[args[0]].get_object()
            if obj.get('/Subtype') != '/Image':
                return
            width = math.hypot(matrix[0], matrix[1])
            height = math.hypot(matrix[2], matrix[3])
            assert abs(width / height - 1) < 1e-4
            assert abs(matrix[1]) < 1e-6 and abs(matrix[2]) < 1e-6
            assert matrix[0] > 0 and matrix[3] > 0, 'Unexpected image reflection'
            geometry.append({'object': str(args[0]), 'native_pixels': [int(obj['/Width']), int(obj['/Height'])],
                             'rendered_width_pt': width, 'rendered_height_pt': height, 'ratio': width / height})
        page.extract_text(visitor_operand_before=visitor)
        assert len(geometry) == len(expected), geometry
        actual_pixels = Counter(pixels(item.image) for item in page.images)
        expected_pixels = Counter(pixels(Image.open(ASSETS / (name + '.png'))) for name in expected)
        assert actual_pixels == expected_pixels, 'Image orientation or pixels differ'
        svg_root = ET.parse(FIGS / (stem + '.svg')).getroot()
        svg_images = svg_root.findall('.//{http://www.w3.org/2000/svg}image')
        assert len(svg_images) == len(expected)
        for item in svg_images:
            assert item.attrib['{http://www.w3.org/1999/xlink}href'].startswith('data:image/')
        record = {
            'date': '2026-10-30', 'figure': stem, 'renderer': 'TikZ/TeX Live 2026',
            'native_pdf_size_pt': [float(page.mediabox.width), float(page.mediabox.height)],
            'images': geometry, 'source_pixels_identical': True, 'no_rotation_or_stretch': True,
            'svg_images_embedded': True, 'independent_candidate_not_in_v0_11': True,
            'sources': {name: sha(FIGS / name) for name in [stem + '.tex', 'assets_20261030/ASSET_MANIFEST.json', 'assets_20261030/binding_labels_v2.tex']},
            'outputs_sha256': {stem + ext: sha(FIGS / (stem + ext)) for ext in ['.pdf', '.png', '.svg']},
        }
        (FIGS / (stem + '.json')).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        print(stem + ': ' + str(len(expected)) + ' images, exact pixels, all displayed ratios=1.0; embedded SVG assets verified.')

if __name__ == '__main__':
    main()
