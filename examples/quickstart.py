"""Quick start: runs in under a minute on a laptop CPU and needs no input files.

    python examples/quickstart.py

It builds two small synthetic backbones (Cα, C, O atoms), turns them into coordinates of the space, and shows the
distance, its certificate, the matched/gap split, the normalized distance, and the energy statistic between two
small ensembles. Replace `synthetic_backbone` with `fc.coordinates_from_pdb('your.pdb')` for real structures.
"""
import numpy as np
import torch

import framedchain as fc


def synthetic_backbone(n_res, seed):
    """Cα, C, O positions (n_res, 3) of a smooth chain with 3.8 Å Cα steps (a stand-in for a PDB file)."""
    rng = np.random.default_rng(seed)
    d, CA = np.array([1.0, 0.0, 0.0]), [np.zeros(3)]
    for _ in range(n_res - 1):
        d = d + 0.5 * rng.normal(size=3)
        d /= np.linalg.norm(d)
        CA.append(CA[-1] + 3.8 * d)
    CA = np.array(CA)
    nxt = np.roll(CA, -1, 0)
    C = CA + 0.4 * (nxt - CA) + 0.3 * rng.normal(size=CA.shape)
    O = C + 1.2 * np.cross(nxt - CA + 1e-3, [0.0, 0.0, 1.0]) / 3.8
    return CA, C, O


def main():
    x = fc.coordinates_from_backbone(*synthetic_backbone(40, 1))     # (39, 3, 3): one 3 × 3 term per Cα–Cα edge
    y = fc.coordinates_from_backbone(*synthetic_backbone(46, 2))     # a different length is fine
    print(f'x: {x.shape[0]} edges, y: {y.shape[0]} edges')

    d, al = fc.distance(x, y)                                        # 128 starts; an upper bound on d
    print(f'distance d(x, y)      = {float(d):.4f}   ({int(al.match.sum())} matched edges)')
    cert = fc.bnb(x, y)                                              # branch and bound: certified within rel_tol
    print(f'certified             = {cert["certified"]}  (S in [{cert["S_lb"]:.4f}, {cert["S_ub"]:.4f}])')
    sp = fc.split_sq_distance(x, y, alignment=al)
    print(f'matched + gap parts   = {float(sp["matched"]):.4f} + {float(sp["gap"]):.4f} = d^2 = {float(d) ** 2:.4f}')
    print(f'normalized D_hat      = {float(fc.normalized_distance(x, y, d=d)):.4f}  (in [0, 1])')

    # ensembles: 3 runs × 20 frames each, frames concatenated run by run
    def runs(base, sigma, seed):
        g = torch.Generator().manual_seed(seed)
        W = fc.expmap(sigma * torch.randn(60, len(base), 3, generator=g, dtype=torch.float64)).numpy()
        return np.einsum('nxy,fnyz->fnxz', base, W)
    st = fc.ensemble_statistic(runs(x, 0.08, 3), runs(x, 0.16, 4), n_per_run=20, k=200)
    print(f'ensemble D2/sqrt(n)   = {st["D2_sqrt_n"]:.4f}  (D2 = {st["D2"]:.4f}, ratio = {st["ratio"]:.3f})')


if __name__ == '__main__':
    main()
