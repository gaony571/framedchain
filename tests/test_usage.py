"""User-facing paths: numpy and mixed inputs, PDB files on disk, batching with zero padding, ensembles of
different lengths (pytest)."""
import numpy as np
import torch

import framespace as fs

LOC_C, LOC_O = np.array([1.42, 0.53, 0.01]), np.array([1.65, 1.74, 0.01])


def chain(n, seed):
    g = torch.Generator().manual_seed(seed)
    step = fs.expmap(torch.tensor([0.3, 1.2, 0.4], dtype=torch.float64))
    R = [torch.eye(3, dtype=torch.float64)]
    for _ in range(n - 1):
        R.append(R[-1] @ step @ fs.expmap(0.15 * torch.randn(3, generator=g, dtype=torch.float64)))
    return (torch.stack(R) / 2).numpy()


def ensemble(base, F, sigma, seed):
    g = torch.Generator().manual_seed(seed)
    W = fs.expmap(sigma * torch.randn(F, len(base), 3, generator=g, dtype=torch.float64)).numpy()
    return np.einsum('nxy,fnyz->fnxz', base, W)


def write_pdb(path, models):
    with open(path, 'w') as f:
        for mi, x in enumerate(models):
            CA = fs.ca_trace(x)
            R, _ = fs.frames_of(x)
            C = np.concatenate([CA[:-1] + R @ LOC_C, CA[-1:] + 1.0])
            O = np.concatenate([CA[:-1] + R @ LOC_O, CA[-1:] + 2.0])
            f.write(f'MODEL     {mi + 1}\n')
            k = 0
            for i in range(len(CA)):
                for name, xyz in (('N', CA[i] - 1.0), ('CA', CA[i]), ('C', C[i]), ('O', O[i])):
                    k += 1
                    f.write('ATOM  %5d  %-3s ALA A%4d    %8.3f%8.3f%8.3f  1.00  0.00\n' % (k, name, i + 1, *xyz))
            f.write('ENDMDL\n')


def test_numpy_and_mixed_inputs_give_the_same_distance():
    x, y = chain(20, 1), chain(24, 2)
    ref = float(fs.distance(torch.as_tensor(x), torch.as_tensor(y))[0])
    assert abs(float(fs.distance(x, y)[0]) - ref) < 1e-12
    assert abs(float(fs.distance(torch.as_tensor(x), y)[0]) - ref) < 1e-12
    a = fs.align(x, y)                                                     # numpy straight into align
    assert abs(float((fs.sq_norm(torch.as_tensor(x)) + fs.sq_norm(torch.as_tensor(y)) - 2 * a.score).sqrt()) - ref) < 1e-12
    a32 = fs.align(x.astype(np.float32), torch.as_tensor(y))               # y follows the dtype of x
    assert a32.score.dtype == torch.float32 and abs(float(a32.score) - float(a.score)) < 1e-3


def test_pdb_file_to_distance_and_certificate(tmp_path):
    x, y = chain(18, 3), chain(21, 4)
    pa, pb = tmp_path / 'a.pdb', tmp_path / 'b.pdb'
    write_pdb(pa, list(ensemble(x, 3, 0.05, 5)))
    write_pdb(pb, list(ensemble(y, 3, 0.05, 6)))
    A, B = fs.coordinates_from_pdb(str(pa)), fs.coordinates_from_pdb(str(pb))
    assert A.shape == (3, 18, 3, 3) and B.shape == (3, 21, 3, 3)
    d, _ = fs.distance(A[0], B[0])
    cert = fs.bnb(A[0], B[0])
    assert cert['certified'] and cert['S_lb'] - 1e-9 <= float(fs.align(A[0], B[0]).score) <= cert['S_ub'] + 1e-9
    assert 0 < float(fs.normalized_distance(A[0], B[0])) < 1 and float(d) > 0


def test_batched_pairs_with_zero_padding():
    xs, ys = [chain(n, 10 + n) for n in (12, 15, 18)], [chain(n, 20 + n) for n in (14, 13, 16)]
    d2, _ = fs.sq_distance(fs.pad_stack(xs), fs.pad_stack(ys))
    single = [float(fs.sq_distance(a, b)[0]) for a, b in zip(xs, ys)]
    assert np.allclose(d2.numpy(), single, atol=1e-9)


def test_ensemble_statistic_from_numpy_and_different_lengths():
    base_a, base_b = chain(16, 7), chain(19, 8)
    EA = np.concatenate([ensemble(base_a, 10, 0.08, 30 + r) for r in range(3)])
    EB = np.concatenate([ensemble(base_b, 10, 0.08, 40 + r) for r in range(3)])
    st = fs.ensemble_statistic(EA, EB, n_per_run=10, k=40)
    assert abs(st['D2'] - (2 * st['C'] - st['B_P'] - st['B_Q'])) < 1e-12 and st['D2'] > 0
    assert abs(st['n_edges'] - 17.5) < 1e-12 and abs(st['D2_sqrt_n'] - st['D2'] / np.sqrt(17.5)) < 1e-12
