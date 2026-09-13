from __future__ import annotations

import numpy as np

from scripts.validate_proxy_nonidentifiability import balanced_proxy, dependence_tv


def test_same_observable_proxy_admits_incompatible_latent_worlds() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 100)
    proxy = balanced_proxy(labels, environments=5)
    world_independent = proxy.copy()
    world_bound = labels % 5

    assert dependence_tv(labels, proxy) <= 1.0e-12
    assert dependence_tv(labels, world_independent) <= 1.0e-12
    assert np.isclose(dependence_tv(labels, world_bound), 0.8)


def test_dependence_tv_matches_binary_deterministic_closed_form() -> None:
    labels = np.repeat(np.arange(4, dtype=np.int64), 25)
    environments = labels % 2
    assert np.isclose(dependence_tv(labels, environments), 0.5)


def test_balanced_proxy_preserves_alignment_and_support() -> None:
    labels = np.repeat(np.arange(3, dtype=np.int64), 10)
    proxy = balanced_proxy(labels, environments=5)
    assert proxy.shape == labels.shape
    for label in np.unique(labels):
        counts = np.bincount(proxy[labels == label], minlength=5)
        assert counts.tolist() == [2, 2, 2, 2, 2]
