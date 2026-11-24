"""SIS, normalized distance correlation and biased HSIC with bounded memory.

All observations are used. Kernel/distance matrices are traversed in row blocks;
centered inner products use sufficient sums, with no row subsampling. The scalar
median positive pair distance uses an exact order-statistic search up to float64
precision, avoiding an n x n bandwidth matrix. Target type is explicit.
"""
from __future__ import annotations

import time
import numpy as np
import torch
from scipy.spatial.distance import pdist


class ScreeningTimeout(TimeoutError):
    pass


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise ScreeningTimeout()


def scalar_bandwidth(x, deadline=None):
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    check_deadline(deadline)
    if len(x) <= 2048:
        d = pdist(x[:, None])
        d = d[d > 0]
        return float(np.partition(d, (len(d) - 1) // 2)[(len(d) - 1) // 2]) if len(d) else 1.0
    n = len(x)
    _, counts = np.unique(x, return_counts=True)
    zeros = int(np.sum(counts * (counts - 1) // 2))
    positive = n * (n - 1) // 2 - zeros
    if positive == 0:
        return 1.0
    rank = (positive + 1) // 2
    low, high = 0.0, float(x[-1] - x[0])
    indices = np.arange(n)
    for _ in range(64):
        check_deadline(deadline)
        mid = (low + high) / 2
        if mid == low or mid == high:
            break
        count = int(np.sum(np.searchsorted(x, x + mid, side='right') - indices - 1)) - zeros
        if count >= rank:
            high = mid
        else:
            low = mid
    return max(high, np.finfo(float).tiny)


def target_tensor(y, problem_type, device):
    if problem_type in ('binary', 'multiclass'):
        _, codes = np.unique(np.asarray(y), return_inverse=True)
        return torch.nn.functional.one_hot(torch.tensor(codes, device=device)).float()
    if problem_type != 'regression':
        raise ValueError(f'Explicit target type required, got {problem_type}')
    return torch.as_tensor(np.asarray(y, dtype=np.float32), device=device).reshape(-1, 1)


def sis_score(z, target):
    z = z.double()
    target = target.double()
    z = z - z.mean(0)
    target = target - target.mean(0)
    denom = z.square().sum(0).sqrt()[:, None] * target.square().sum(0).sqrt()[None, :]
    corr = (z.T @ target).abs() / denom.clamp_min(1e-30)
    corr = torch.where(denom > 1e-20, corr, 0)
    # max over one-vs-rest classes, mean over embedding coordinates (d=1 for Raw).
    return float(corr.max(dim=1).values.mean().cpu())


@torch.inference_mode()
def kernel_score(z, target, utility, *, sigma_x=1.0, sigma_y=1.0,
                 categorical=False, pace=False, row_block=256, deadline=None):
    n = len(z)
    if n < 2:
        return 0.0
    # Centering inputs reduces cancellation in the Euclidean distance calculation.
    z = z - z.mean(0)
    target = target - target.mean(0)
    if utility == 'dc':
        z = z / z.square().mean().sqrt().clamp_min(1e-12)
        target = target / target.square().mean().sqrt().clamp_min(1e-12)
    rows = max(1, min(row_block, 2**20 // n))
    rz = torch.zeros(n, device=z.device, dtype=torch.float64)
    ry = torch.zeros_like(rz)
    sums = torch.zeros(3, device=z.device, dtype=torch.float64)
    for start in range(0, n, rows):
        check_deadline(deadline)
        stop = min(n, start + rows)
        a = torch.cdist(z[start:stop], z)
        b = torch.cdist(target[start:stop], target)
        idx = torch.arange(stop - start, device=z.device)
        a[idx, idx + start] = 0
        b[idx, idx + start] = 0
        if utility == 'hsic':
            a = torch.exp(-a.square() / (2 * sigma_x**2))
            if categorical and not pace:
                # One-hot Euclidean distances are zero exactly within a class.
                b = (b < 0.5).float()
            else:
                b = torch.exp(-b.square() / (2 * sigma_y**2))
        elif utility != 'dc':
            raise ValueError(utility)
        a, b = a.double(), b.double()
        rz[start:stop] = a.sum(1)
        ry[start:stop] = b.sum(1)
        sums[0] += (a * b).sum()
        if utility == 'dc':
            sums[1] += a.square().sum()
            sums[2] += b.square().sum()
    sz, sy = rz.sum(), ry.sum()
    cross = sums[0] - 2 * (rz * ry).sum() / n + sz * sy / n**2
    if utility == 'hsic':
        return float((cross / n**2).clamp_min(0).cpu())
    vz = (sums[1] - 2 * rz.square().sum() / n + sz.square() / n**2).clamp_min(0)
    vy = (sums[2] - 2 * ry.square().sum() / n + sy.square() / n**2).clamp_min(0)
    denom = (vz * vy).sqrt()
    value = torch.where(denom > 1e-20, (cross / denom.clamp_min(1e-30)).clamp(0, 1).sqrt(), 0)
    return float(value.cpu())
