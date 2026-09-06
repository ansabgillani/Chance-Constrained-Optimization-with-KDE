import numpy as np
import pytest
from kde_cco.uncertainty.samplers import (
    sample_gaussian, sample_bimodal, sample_skewed, sample_heavy_tailed,
    sample_renewable_error,
)
from kde_cco.uncertainty.splits import split_samples


@pytest.mark.parametrize("sampler", [sample_gaussian, sample_bimodal, sample_skewed,
                                      sample_heavy_tailed, sample_renewable_error])
def test_seeded_samplers_reproduce_and_report_metadata(sampler):
    a, meta_a = sampler(32, seed=7)
    b, meta_b = sampler(32, seed=7)
    c, _ = sampler(32, seed=8)
    assert a.shape == (32,)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert isinstance(meta_a, dict) and meta_a["distribution"]
    assert meta_a == meta_b


def test_samplers_are_independent_of_legacy_global_state():
    np.random.seed(1)
    first, _ = sample_gaussian(24, seed=91)
    np.random.seed(999)
    second, _ = sample_gaussian(24, seed=91)
    assert np.array_equal(first, second)


def test_renewable_metadata_declares_signed_modes():
    _, metadata = sample_renewable_error(4, seed=12)
    modes = metadata["mixture_components"]
    assert metadata["distribution"] == "renewable_error"
    assert [mode["label"] for mode in modes] == ["underproduction", "overproduction"]
    assert modes[0]["mean"] < 0 < modes[1]["mean"]
    assert np.isclose(sum(mode["weight"] for mode in modes), 1.0)


def test_split_returns_disjoint_copies_and_indices():
    samples = np.arange(20)
    train, test, indices = split_samples(samples, 8, seed=3)
    assert train.shape == (8,) and test.shape == (12,)
    assert np.intersect1d(train, test).size == 0
    assert len(indices) == 8 and len(np.unique(indices)) == 8
    train[0] = -1
    assert samples[indices[0]] != -1


def test_split_preserves_trailing_dimensions_and_is_reproducible():
    samples = np.arange(30).reshape(10, 3)
    train_a, test_a, idx_a = split_samples(samples, 4, seed=17)
    train_b, test_b, idx_b = split_samples(samples, 4, seed=17)
    assert train_a.shape == (4, 3) and test_a.shape == (6, 3)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(test_a, test_b)
    assert np.array_equal(idx_a, idx_b)
    test_a[0, 0] = -1
    assert not np.any(samples == -1)


def test_split_validates_train_size():
    with pytest.raises(ValueError):
        split_samples(np.arange(3), 0, seed=1)
    with pytest.raises(ValueError):
        split_samples(np.arange(3), 3, seed=1)
    with pytest.raises(ValueError):
        split_samples(np.arange(3), True, seed=1)
    with pytest.raises(ValueError):
        split_samples(np.arange(3), 1, seed=True)
    with pytest.raises(TypeError):
        split_samples(np.arange(3), 1, seed="1")
