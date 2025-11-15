"""Pytest configuration and common fixtures."""

import pytest
import numpy as np
from src.test_data import TestData


@pytest.fixture
def clear_difference_data():
    """明確な差があるテストデータ."""
    return TestData(n_a=1000, conv_a=100, n_b=1000, conv_b=150)


@pytest.fixture
def subtle_difference_data():
    """微妙な差があるテストデータ."""
    return TestData(n_a=1000, conv_a=100, n_b=1000, conv_b=115)


@pytest.fixture
def no_difference_data():
    """差がないテストデータ."""
    return TestData(n_a=1000, conv_a=100, n_b=1000, conv_b=100)


@pytest.fixture
def small_sample_data():
    """小サンプルのテストデータ."""
    return TestData(n_a=50, conv_a=10, n_b=50, conv_b=15)


@pytest.fixture
def extreme_difference_data():
    """極端な差があるテストデータ."""
    return TestData(n_a=1000, conv_a=50, n_b=1000, conv_b=200)


@pytest.fixture
def zero_conversion_a_data():
    """グループAのコンバージョンが0のテストデータ."""
    return TestData(n_a=100, conv_a=0, n_b=100, conv_b=10)


@pytest.fixture
def zero_conversion_both_data():
    """両グループのコンバージョンが0のテストデータ."""
    return TestData(n_a=100, conv_a=0, n_b=100, conv_b=0)


@pytest.fixture
def perfect_conversion_b_data():
    """グループBのコンバージョンが100%のテストデータ."""
    return TestData(n_a=100, conv_a=50, n_b=100, conv_b=100)


@pytest.fixture(autouse=True)
def reset_random_seed():
    """各テストの前に乱数シードをリセット."""
    np.random.seed(42)
    yield
    np.random.seed(None)
