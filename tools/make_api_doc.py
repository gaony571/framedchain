"""Write docs/API.md from the docstrings of the installed (or in-tree) package.

Usage (from the repository root):  python tools/make_api_doc.py
Needs numba for the rulers module; without it that section is skipped with a note.
"""
import importlib
import inspect
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import framedchain  # noqa: E402

MODULES = ['frames', 'distance', 'certify', 'operations', 'table', 'transport', 'parts', 'reduced', 'ensemble',
           'geometry', 'rulers']
LOW_LEVEL = {'dp_score', 'dp_score_band', 'align_band', 'band_windows', 'center_path', 'self_rotations', 'hat',
             'expmap', 'procrustes', 'polar', 'quat_to_mat', 'mat_to_quat', 'quat_mul', 'quat_conj', 'rotvec_to_quat'}


def main():
    out = [f'# framedchain {framedchain.__version__} — API reference', '',
           'Generated from the docstrings by `tools/make_api_doc.py`. Every name listed here except the `rulers` module is',
           'importable from the top level (`import framedchain as fc; fc.distance(...)`); `rulers` needs numba and is',
           'imported explicitly (`from framedchain import rulers`).', '',
           'Conventions: a structure with L residues is an array (L − 1, 3, 3) of edge coordinates (one 3 × 3 term per',
           'Cα–Cα edge); batches add a leading axis and are zero-padded to a common length (`pad_stack`); ensembles are',
           '(frames, n, 3, 3). Distances are in units of the coordinate norm (a trans edge has squared norm 0.75); Cα',
           'positions and PDB coordinates are in Å. High-level functions accept numpy arrays or torch tensors; functions',
           'marked *(tensor)* are low-level building blocks that expect torch tensors. float64 is recommended.', '']
    for name in MODULES:
        try:
            mod = importlib.import_module(f'framedchain.{name}')
        except ImportError as e:
            out += [f'## `{name}`', '', f'(not documented here: {e})', '']
            continue
        out += [f'## `{name}`', '', inspect.getdoc(mod) or '', '']
        for attr, obj in vars(mod).items():
            if attr.startswith('_') or getattr(obj, '__module__', None) != mod.__name__:
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = '(...)'
            tag = ' *(tensor)*' if attr in LOW_LEVEL else ''
            out += [f'### `{attr}{sig}`{tag}', '', inspect.getdoc(obj) or '(no docstring)', '']
            if inspect.isclass(obj):
                for m, f in vars(obj).items():
                    if inspect.isfunction(f) and not m.startswith('_'):
                        out += [f'#### `{attr}.{m}{inspect.signature(f)}`', '', inspect.getdoc(f) or '', '']
    path = os.path.join(ROOT, 'docs', 'API.md')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(out).rstrip() + '\n')
    print('wrote', path)


if __name__ == '__main__':
    main()
