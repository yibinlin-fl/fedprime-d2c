"""Export unchanged CIFAR thumbnails and exact binding labels for TikZ."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json

from PIL import Image

HERE = Path(__file__).resolve().parent
FIGS = HERE / 'figures'
OUT = FIGS / 'assets_20261030'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def keep(path, data):
    if path.exists() and path.read_bytes() != data:
        raise FileExistsError('Versioned asset differs: ' + str(path))
    path.write_bytes(data)

def main():
    OUT.mkdir(exist_ok=True)
    diag = load('diagnosis_snapshot', FIGS / 'fig_cle_paired_dsa_v4.py')
    execution = load('execution_snapshot', FIGS / 'fig_pew_ber_execution_v3.py')
    metadata = {'date': '2026-10-30', 'resize_rotate_crop': False, 'images': []}
    def image(name, array, source, label):
        img = Image.fromarray(array)
        if img.size != (32, 32):
            raise ValueError('Expected original 32x32 image')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        path = OUT / (name + '.png')
        keep(path, buf.getvalue())
        metadata['images'].append({'path': path.name, 'pixels': [32, 32], 'source': source,
                                   'label': label, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    public = execution.public_image()
    image('public_clean', public, 'public/cifar-100-python.tar.gz train index42', 'clean')
    for arr, label in execution.variants(public):
        image('public_' + label, arr, 'schematic synthetic view of public index42; not a measured training result', label)
    for i, (arr, label) in enumerate(execution.private_images()):
        image('private_' + str(i), arr, 'gamma09 client0 private fit; original frozen bytes', label)
    arrays, label_id = diag.load_paired_source(3, [0, 3, 7, 13])
    for arr, operator in zip(arrays, ['gaussian_noise', 'defocus_blur', 'snow', 'pixelate']):
        image('paired_' + operator, arr, 'paired_dsa source3 severity3', operator)
    bindings, families = diag.load_protocol()
    metadata.update(source_label=diag.CLASS_NAMES[label_id], source_id=3, client_binding_map=bindings)
    lines = ['% Generated from frozen metadata; three example classes per client.']
    for client, start_y, model in [(0, 5.75, 'ResNet10'), (1, 3.65, 'ResNet12')]:
        lines.append(r'\node[anchor=west,font=\bfseries\fontsize{8}{9}\selectfont] at (0.15,' + str(start_y) + ') {Client ' + str(client) + r' \quad ' + model + '};')
        for j, cls in enumerate([0, 1, 2]):
            y = start_y - .49 - .43*j
            operator = bindings[str(client)][str(cls)]
            label = diag.SHORT_OPERATOR.get(operator, operator.replace('_', ' ').title()).replace('Jpeg', 'JPEG')
            lines += [r'\node[anchor=west] at (0.15,' + f'{y:.2f}' + ') {' + diag.CLASS_NAMES[cls] + '};',
                      r'\draw[arr] (1.52,' + f'{y:.2f}' + ') -- (1.94,' + f'{y:.2f}' + ');',
                      r'\node[anchor=west] at (2.04,' + f'{y:.2f}' + ') {' + label + '};']
    keep(OUT / 'binding_labels.tex', ('\n'.join(lines) + '\n').encode('utf-8'))
    metadata['asset_preparer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    keep(OUT / 'ASSET_MANIFEST.json', (json.dumps(metadata, indent=2) + '\n').encode('utf-8'))
    print('Exported', len(metadata['images']), 'square source images; no resize, rotation or crop.')

if __name__ == '__main__':
    main()
