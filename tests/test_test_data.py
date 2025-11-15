"""Tests for TestData class."""

import pytest
from src.test_data import TestData, TestMethod


class TestTestDataValidation:
    """TestDataのバリデーションテスト."""

    def test_valid_data(self):
        """有効なデータで初期化できることを確認."""
        data = TestData(n_a=100, conv_a=10, n_b=100, conv_b=20)
        assert data.n_a == 100
        assert data.conv_a == 10
        assert data.n_b == 100
        assert data.conv_b == 20

    def test_zero_sample_size_a_raises_error(self):
        """グループAのサンプルサイズが0の場合エラー."""
        with pytest.raises(ValueError, match="サンプルサイズは正の整数である必要があります"):
            TestData(n_a=0, conv_a=0, n_b=100, conv_b=10)

    def test_zero_sample_size_b_raises_error(self):
        """グループBのサンプルサイズが0の場合エラー."""
        with pytest.raises(ValueError, match="サンプルサイズは正の整数である必要があります"):
            TestData(n_a=100, conv_a=10, n_b=0, conv_b=0)

    def test_negative_sample_size_raises_error(self):
        """負のサンプルサイズでエラー."""
        with pytest.raises(ValueError, match="サンプルサイズは正の整数である必要があります"):
            TestData(n_a=-1, conv_a=0, n_b=100, conv_b=10)

    def test_negative_conversion_a_raises_error(self):
        """グループAの負のコンバージョン数でエラー."""
        with pytest.raises(ValueError, match="コンバージョン数は非負整数である必要があります"):
            TestData(n_a=100, conv_a=-1, n_b=100, conv_b=10)

    def test_negative_conversion_b_raises_error(self):
        """グループBの負のコンバージョン数でエラー."""
        with pytest.raises(ValueError, match="コンバージョン数は非負整数である必要があります"):
            TestData(n_a=100, conv_a=10, n_b=100, conv_b=-1)

    def test_conversion_exceeds_sample_size_a_raises_error(self):
        """グループAのコンバージョン数がサンプルサイズを超える場合エラー."""
        with pytest.raises(ValueError, match="コンバージョン数.*はサンプルサイズ.*を超えることはできません"):
            TestData(n_a=100, conv_a=101, n_b=100, conv_b=10)

    def test_conversion_exceeds_sample_size_b_raises_error(self):
        """グループBのコンバージョン数がサンプルサイズを超える場合エラー."""
        with pytest.raises(ValueError, match="コンバージョン数.*はサンプルサイズ.*を超えることはできません"):
            TestData(n_a=100, conv_a=10, n_b=100, conv_b=101)

    def test_zero_conversions_valid(self):
        """コンバージョン数が0でも有効."""
        data = TestData(n_a=100, conv_a=0, n_b=100, conv_b=0)
        assert data.conv_a == 0
        assert data.conv_b == 0

    def test_perfect_conversion_valid(self):
        """コンバージョン率が100%でも有効."""
        data = TestData(n_a=100, conv_a=100, n_b=100, conv_b=100)
        assert data.conv_a == 100
        assert data.conv_b == 100


class TestTestDataProperties:
    """TestDataのプロパティ計算テスト."""

    def test_cvr_a_calculation(self):
        """グループAのCVR計算が正確."""
        data = TestData(n_a=1000, conv_a=100, n_b=1000, conv_b=150)
        assert data.cvr_a == 0.1
        assert data.cvr_a == pytest.approx(0.1, abs=1e-10)

    def test_cvr_b_calculation(self):
        """グループBのCVR計算が正確."""
        data = TestData(n_a=1000, conv_a=100, n_b=1000, conv_b=150)
        assert data.cvr_b == 0.15
        assert data.cvr_b == pytest.approx(0.15, abs=1e-10)

    def test_cvr_diff_calculation(self):
        """CVRの差の計算が正確."""
        data = TestData(n_a=1000, conv_a=100, n_b=1000, conv_b=150)
        assert data.cvr_diff == pytest.approx(0.05, abs=1e-10)

    def test_cvr_diff_negative(self):
        """CVRの差が負の場合も正確."""
        data = TestData(n_a=1000, conv_a=150, n_b=1000, conv_b=100)
        assert data.cvr_diff == pytest.approx(-0.05, abs=1e-10)

    def test_cvr_zero_conversions(self):
        """コンバージョンが0の場合のCVR."""
        data = TestData(n_a=100, conv_a=0, n_b=100, conv_b=10)
        assert data.cvr_a == 0.0
        assert data.cvr_b == 0.1

    def test_cvr_perfect_conversion(self):
        """コンバージョンが100%の場合のCVR."""
        data = TestData(n_a=100, conv_a=100, n_b=100, conv_b=100)
        assert data.cvr_a == 1.0
        assert data.cvr_b == 1.0

    def test_cvr_decimal_precision(self):
        """小数点以下の精度が保たれる."""
        data = TestData(n_a=3, conv_a=1, n_b=3, conv_b=2)
        assert data.cvr_a == pytest.approx(1/3, abs=1e-10)
        assert data.cvr_b == pytest.approx(2/3, abs=1e-10)
        assert data.cvr_diff == pytest.approx(1/3, abs=1e-10)


class TestTestMethod:
    """TestMethod列挙型のテスト."""

    def test_test_method_values(self):
        """TestMethodの値が正しい."""
        assert TestMethod.Z_TEST.value == "z_test"
        assert TestMethod.T_TEST.value == "t_test"
        assert TestMethod.CHI_SQUARE.value == "chi_square"

    def test_test_method_enumeration(self):
        """TestMethodが列挙型として機能する."""
        methods = list(TestMethod)
        assert len(methods) == 3
        assert TestMethod.Z_TEST in methods
        assert TestMethod.T_TEST in methods
        assert TestMethod.CHI_SQUARE in methods
