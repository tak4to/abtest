"""Tests for FrequentistABTest class."""

import pytest
import numpy as np
from scipy import stats

from src.frequentist import FrequentistABTest
from src.test_data import TestData, TestMethod


class TestFrequentistZTest:
    """Z検定のテスト."""

    def test_z_test_clear_difference(self, clear_difference_data):
        """明確な差がある場合のZ検定."""
        test = FrequentistABTest(clear_difference_data)
        result = test.z_test()

        assert result.method == TestMethod.Z_TEST
        assert result.is_significant == True
        assert result.p_value < 0.05
        assert result.test_statistic != 0

    def test_z_test_no_difference(self, no_difference_data):
        """差がない場合のZ検定."""
        test = FrequentistABTest(no_difference_data)
        result = test.z_test()

        assert result.method == TestMethod.Z_TEST
        assert result.is_significant == False
        assert result.p_value > 0.05
        assert abs(result.test_statistic) < 0.1

    def test_z_test_statistic_sign(self, clear_difference_data):
        """Z統計量の符号が正しい."""
        test = FrequentistABTest(clear_difference_data)
        result = test.z_test()

        # グループBの方が高いのでZ統計量は正
        assert result.test_statistic > 0

    def test_z_test_statistic_manual_calculation(self):
        """Z統計量の手動計算と一致."""
        data = TestData(n_a=100, conv_a=10, n_b=100, conv_b=20)
        test = FrequentistABTest(data)
        result = test.z_test()

        # 手動計算
        p_a = 10 / 100
        p_b = 20 / 100
        p_pool = (10 + 20) / (100 + 100)
        se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/100 + 1/100))
        expected_z = (p_b - p_a) / se_pool

        assert result.test_statistic == pytest.approx(expected_z, abs=1e-10)

    def test_z_test_p_value_range(self, clear_difference_data):
        """p値が0-1の範囲内."""
        test = FrequentistABTest(clear_difference_data)
        result = test.z_test()

        assert 0 <= result.p_value <= 1

    def test_z_test_confidence_interval_contains_diff(self, clear_difference_data):
        """信頼区間が実際の差を含む（概ね）."""
        test = FrequentistABTest(clear_difference_data, confidence_level=0.95)
        result = test.z_test()

        actual_diff = clear_difference_data.cvr_diff
        # 大サンプルなので、実際の差を含むはず
        assert result.ci_lower <= actual_diff <= result.ci_upper

    def test_z_test_confidence_level_effect(self, clear_difference_data):
        """信頼水準が高いほど信頼区間が広い."""
        test_90 = FrequentistABTest(clear_difference_data, confidence_level=0.90)
        test_99 = FrequentistABTest(clear_difference_data, confidence_level=0.99)

        result_90 = test_90.z_test()
        result_99 = test_99.z_test()

        width_90 = result_90.ci_upper - result_90.ci_lower
        width_99 = result_99.ci_upper - result_99.ci_lower

        assert width_99 > width_90

    def test_z_test_pooled_proportion(self, clear_difference_data):
        """プールされた比率が正しい."""
        test = FrequentistABTest(clear_difference_data)
        result = test.z_test()

        expected_pool = (clear_difference_data.conv_a + clear_difference_data.conv_b) / \
                        (clear_difference_data.n_a + clear_difference_data.n_b)

        assert result.additional_info["pooled_proportion"] == pytest.approx(expected_pool, abs=1e-10)

    def test_z_test_symmetry(self):
        """AとBを入れ替えても統計量の絶対値は同じ."""
        data1 = TestData(n_a=100, conv_a=10, n_b=100, conv_b=20)
        data2 = TestData(n_a=100, conv_a=20, n_b=100, conv_b=10)

        result1 = FrequentistABTest(data1).z_test()
        result2 = FrequentistABTest(data2).z_test()

        assert abs(result1.test_statistic) == pytest.approx(abs(result2.test_statistic), abs=1e-10)
        assert result1.p_value == pytest.approx(result2.p_value, abs=1e-10)


class TestFrequentistTTest:
    """t検定のテスト."""

    def test_t_test_clear_difference(self, clear_difference_data):
        """明確な差がある場合のt検定."""
        test = FrequentistABTest(clear_difference_data)
        result = test.t_test()

        assert result.method == TestMethod.T_TEST
        assert result.is_significant == True
        assert result.p_value < 0.05

    def test_t_test_no_difference(self, no_difference_data):
        """差がない場合のt検定."""
        test = FrequentistABTest(no_difference_data)
        result = test.t_test()

        assert result.method == TestMethod.T_TEST
        assert result.is_significant == False
        assert result.p_value > 0.05

    def test_t_test_small_sample(self, small_sample_data):
        """小サンプルでもt検定が動作."""
        test = FrequentistABTest(small_sample_data)
        result = test.t_test()

        assert result.method == TestMethod.T_TEST
        assert 0 <= result.p_value <= 1

    def test_t_test_degrees_of_freedom(self):
        """Welchの自由度が正しく計算される."""
        data = TestData(n_a=100, conv_a=10, n_b=100, conv_b=20)
        test = FrequentistABTest(data)
        result = test.t_test()

        # 自由度が0より大きい
        assert result.additional_info["degrees_of_freedom"] > 0
        # 自由度が最大値（n_a + n_b - 2）以下
        assert result.additional_info["degrees_of_freedom"] <= 100 + 100 - 2

    def test_t_test_manual_calculation(self):
        """t統計量の手動計算と一致."""
        data = TestData(n_a=100, conv_a=10, n_b=100, conv_b=20)
        test = FrequentistABTest(data)
        result = test.t_test()

        # 手動計算
        p_a = 10 / 100
        p_b = 20 / 100
        var_a = p_a * (1 - p_a) / 100
        var_b = p_b * (1 - p_b) / 100
        expected_t = (p_b - p_a) / np.sqrt(var_a + var_b)

        assert result.test_statistic == pytest.approx(expected_t, abs=1e-10)

    def test_t_test_variance_info(self, clear_difference_data):
        """分散情報が正しく格納される."""
        test = FrequentistABTest(clear_difference_data)
        result = test.t_test()

        assert "variance_a" in result.additional_info
        assert "variance_b" in result.additional_info
        assert result.additional_info["variance_a"] > 0
        assert result.additional_info["variance_b"] > 0

    def test_t_test_confidence_interval_width(self, small_sample_data):
        """小サンプルでは信頼区間が広い（Z検定より）."""
        test = FrequentistABTest(small_sample_data)

        t_result = test.t_test()
        z_result = test.z_test()

        t_width = t_result.ci_upper - t_result.ci_lower
        z_width = z_result.ci_upper - z_result.ci_lower

        # t検定の方がやや広いか同等
        assert t_width >= z_width * 0.95  # 誤差を考慮


class TestFrequentistChiSquareTest:
    """カイ二乗検定のテスト."""

    def test_chi_square_clear_difference(self, clear_difference_data):
        """明確な差がある場合のカイ二乗検定."""
        test = FrequentistABTest(clear_difference_data)
        result = test.chi_square_test()

        assert result.method == TestMethod.CHI_SQUARE
        assert result.is_significant == True
        assert result.p_value < 0.05
        assert result.test_statistic > 0

    def test_chi_square_no_difference(self, no_difference_data):
        """差がない場合のカイ二乗検定."""
        test = FrequentistABTest(no_difference_data)
        result = test.chi_square_test()

        assert result.method == TestMethod.CHI_SQUARE
        assert result.is_significant == False
        assert result.p_value > 0.05

    def test_chi_square_statistic_non_negative(self, clear_difference_data):
        """カイ二乗統計量は非負."""
        test = FrequentistABTest(clear_difference_data)
        result = test.chi_square_test()

        assert result.test_statistic >= 0

    def test_chi_square_degrees_of_freedom(self, clear_difference_data):
        """自由度が正しい（2x2分割表では1）."""
        test = FrequentistABTest(clear_difference_data)
        result = test.chi_square_test()

        assert result.additional_info["degrees_of_freedom"] == 1

    def test_chi_square_observed_table(self, clear_difference_data):
        """観測度数表が正しく格納される."""
        test = FrequentistABTest(clear_difference_data)
        result = test.chi_square_test()

        observed = result.additional_info["observed"]
        assert observed[0][0] == clear_difference_data.conv_a
        assert observed[0][1] == clear_difference_data.n_a - clear_difference_data.conv_a
        assert observed[1][0] == clear_difference_data.conv_b
        assert observed[1][1] == clear_difference_data.n_b - clear_difference_data.conv_b

    def test_chi_square_expected_frequencies(self, clear_difference_data):
        """期待度数が正しく計算される."""
        test = FrequentistABTest(clear_difference_data)
        result = test.chi_square_test()

        expected = result.additional_info["expected"]
        # 期待度数の合計が観測度数の合計と一致
        assert np.sum(expected) == pytest.approx(
            clear_difference_data.n_a + clear_difference_data.n_b,
            abs=1e-8
        )

    def test_chi_square_yates_correction(self, small_sample_data):
        """Yates補正版も計算される."""
        test = FrequentistABTest(small_sample_data)
        result = test.chi_square_test()

        assert "chi2_yates" in result.additional_info
        assert "p_value_yates" in result.additional_info

        # Yates補正は通常より保守的（p値が大きい）
        # ただし、常にそうとは限らないので、単に存在確認のみ

    def test_chi_square_wilson_ci(self, clear_difference_data):
        """Wilson信頼区間が計算される."""
        test = FrequentistABTest(clear_difference_data)
        result = test.chi_square_test()

        assert "ci_a" in result.additional_info
        assert "ci_b" in result.additional_info

        # 信頼区間が妥当な範囲
        ci_a = result.additional_info["ci_a"]
        ci_b = result.additional_info["ci_b"]

        assert 0 <= ci_a[0] <= ci_a[1] <= 1
        assert 0 <= ci_b[0] <= ci_b[1] <= 1

    def test_chi_square_vs_scipy(self):
        """scipyのカイ二乗検定と一致."""
        data = TestData(n_a=100, conv_a=10, n_b=100, conv_b=20)
        test = FrequentistABTest(data)
        result = test.chi_square_test()

        # scipyで直接計算
        observed = np.array([
            [10, 90],
            [20, 80]
        ])
        chi2_stat, p_value, _, _ = stats.chi2_contingency(observed, correction=False)

        assert result.test_statistic == pytest.approx(chi2_stat, abs=1e-10)
        assert result.p_value == pytest.approx(p_value, abs=1e-10)


class TestFrequentistRun:
    """run()メソッドのテスト."""

    def test_run_z_test(self, clear_difference_data):
        """Z検定を実行."""
        test = FrequentistABTest(clear_difference_data)
        result = test.run(TestMethod.Z_TEST)

        assert result.method == TestMethod.Z_TEST

    def test_run_t_test(self, clear_difference_data):
        """t検定を実行."""
        test = FrequentistABTest(clear_difference_data)
        result = test.run(TestMethod.T_TEST)

        assert result.method == TestMethod.T_TEST

    def test_run_chi_square(self, clear_difference_data):
        """カイ二乗検定を実行."""
        test = FrequentistABTest(clear_difference_data)
        result = test.run(TestMethod.CHI_SQUARE)

        assert result.method == TestMethod.CHI_SQUARE

    def test_run_unknown_method_raises_error(self, clear_difference_data):
        """未知の検定方法でエラー."""
        test = FrequentistABTest(clear_difference_data)

        # TestMethodではない値を渡す（型チェック回避のため文字列で）
        with pytest.raises(ValueError, match="Unknown test method"):
            test.run("unknown")


class TestFrequentistMethodComparison:
    """検定方法間の比較テスト."""

    def test_all_methods_agree_on_clear_difference(self, clear_difference_data):
        """明確な差がある場合、すべての方法で有意."""
        test = FrequentistABTest(clear_difference_data)

        z_result = test.z_test()
        t_result = test.t_test()
        chi_result = test.chi_square_test()

        assert z_result.is_significant == True
        assert t_result.is_significant == True
        assert chi_result.is_significant == True

    def test_all_methods_agree_on_no_difference(self, no_difference_data):
        """差がない場合、すべての方法で非有意."""
        test = FrequentistABTest(no_difference_data)

        z_result = test.z_test()
        t_result = test.t_test()
        chi_result = test.chi_square_test()

        assert z_result.is_significant == False
        assert t_result.is_significant == False
        assert chi_result.is_significant == False

    def test_p_values_similar_large_sample(self, clear_difference_data):
        """大サンプルではp値が類似."""
        test = FrequentistABTest(clear_difference_data)

        z_result = test.z_test()
        t_result = test.t_test()
        chi_result = test.chi_square_test()

        # 大サンプルでは、p値が近い（誤差10%以内）
        assert z_result.p_value == pytest.approx(t_result.p_value, rel=0.1)
        assert z_result.p_value == pytest.approx(chi_result.p_value, rel=0.1)

    def test_z_squared_equals_chi_square(self):
        """Z統計量の二乗がカイ二乗統計量にほぼ等しい."""
        data = TestData(n_a=1000, conv_a=100, n_b=1000, conv_b=150)
        test = FrequentistABTest(data)

        z_result = test.z_test()
        chi_result = test.chi_square_test()

        # Z^2 ≈ χ^2（大サンプルで）
        assert z_result.test_statistic ** 2 == pytest.approx(chi_result.test_statistic, abs=0.1)


class TestFrequentistEdgeCases:
    """エッジケースのテスト."""

    def test_zero_conversion_both_groups(self, zero_conversion_both_data):
        """両グループともコンバージョンが0."""
        test = FrequentistABTest(zero_conversion_both_data)

        # Z検定のみテスト
        # （t検定は分散が0、カイ二乗検定は期待度数が0のため数値エラー）
        z_result = test.z_test()

        # NaNになっても許容（0/0のため）
        # または非有意であることを確認
        if not np.isnan(z_result.p_value):
            assert z_result.is_significant == False

    def test_perfect_conversion_b(self, perfect_conversion_b_data):
        """グループBのコンバージョンが100%."""
        test = FrequentistABTest(perfect_conversion_b_data)

        # すべての検定で有意
        assert test.z_test().is_significant == True
        assert test.t_test().is_significant == True
        assert test.chi_square_test().is_significant == True

    def test_small_sample_all_methods_work(self, small_sample_data):
        """小サンプルでもすべての検定が動作."""
        test = FrequentistABTest(small_sample_data)

        z_result = test.z_test()
        t_result = test.t_test()
        chi_result = test.chi_square_test()

        # すべて結果が返される
        assert z_result is not None
        assert t_result is not None
        assert chi_result is not None

    def test_extreme_difference_all_significant(self, extreme_difference_data):
        """極端な差ではすべての検定で有意."""
        test = FrequentistABTest(extreme_difference_data)

        assert test.z_test().is_significant == True
        assert test.t_test().is_significant == True
        assert test.chi_square_test().is_significant == True

        # p値が非常に小さい
        assert test.z_test().p_value < 0.001
        assert test.t_test().p_value < 0.001
        assert test.chi_square_test().p_value < 0.001


class TestFrequentistConfidenceInterval:
    """信頼区間のテスト."""

    def test_confidence_interval_contains_zero_no_difference(self, no_difference_data):
        """差がない場合、信頼区間が0を含む."""
        test = FrequentistABTest(no_difference_data)

        z_result = test.z_test()
        t_result = test.t_test()

        assert z_result.ci_lower <= 0 <= z_result.ci_upper
        assert t_result.ci_lower <= 0 <= t_result.ci_upper

    def test_confidence_interval_excludes_zero_clear_difference(self, clear_difference_data):
        """明確な差がある場合、信頼区間が0を含まない."""
        test = FrequentistABTest(clear_difference_data)

        z_result = test.z_test()
        t_result = test.t_test()

        # BがAより優れているので、下限が0より大きい
        assert z_result.ci_lower > 0
        assert t_result.ci_lower > 0

    def test_confidence_interval_width_proportional_to_sample_size(self):
        """サンプルサイズが大きいほど信頼区間が狭い."""
        data_small = TestData(n_a=100, conv_a=10, n_b=100, conv_b=15)
        data_large = TestData(n_a=1000, conv_a=100, n_b=1000, conv_b=150)

        test_small = FrequentistABTest(data_small)
        test_large = FrequentistABTest(data_large)

        result_small = test_small.z_test()
        result_large = test_large.z_test()

        width_small = result_small.ci_upper - result_small.ci_lower
        width_large = result_large.ci_upper - result_large.ci_lower

        # 大サンプルの方が狭い
        assert width_large < width_small

    def test_confidence_interval_symmetry(self):
        """信頼区間が推定値を中心にほぼ対称."""
        data = TestData(n_a=100, conv_a=10, n_b=100, conv_b=20)
        test = FrequentistABTest(data)
        result = test.z_test()

        diff = data.cvr_diff
        lower_distance = diff - result.ci_lower
        upper_distance = result.ci_upper - diff

        # ほぼ対称（誤差1%以内）
        assert lower_distance == pytest.approx(upper_distance, rel=0.01)
