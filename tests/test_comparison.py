"""Tests for ABTestComparison class (integration tests)."""

import pytest
from src.comparison import ABTestComparison
from src.test_data import TestMethod


class TestABTestComparisonInitialization:
    """ABTestComparison初期化のテスト."""

    def test_initialization_default_params(self, clear_difference_data):
        """デフォルトパラメータで初期化."""
        comparison = ABTestComparison(clear_difference_data)

        assert comparison.data == clear_difference_data
        assert comparison.frequentist is not None
        assert comparison.bayesian is not None

    def test_initialization_custom_confidence_level(self, clear_difference_data):
        """カスタム信頼水準で初期化."""
        comparison = ABTestComparison(clear_difference_data, confidence_level=0.99)

        # FrequentistとBayesianの信頼水準が一致
        assert comparison.frequentist.confidence_level == 0.99
        assert comparison.bayesian.credible_level == 0.99

    def test_initialization_custom_prior(self, clear_difference_data):
        """カスタム事前分布で初期化."""
        comparison = ABTestComparison(
            clear_difference_data,
            alpha_prior=2.0,
            beta_prior=2.0
        )

        assert comparison.bayesian.alpha_prior == 2.0
        assert comparison.bayesian.beta_prior == 2.0


class TestABTestComparisonRunAll:
    """run_all()メソッドのテスト."""

    def test_run_all_returns_both_results(self, clear_difference_data):
        """run_all()が両方の結果を返す."""
        comparison = ABTestComparison(clear_difference_data)
        freq_result, bayes_result = comparison.run_all()

        assert freq_result is not None
        assert bayes_result is not None

    def test_run_all_with_z_test(self, clear_difference_data):
        """Z検定でrun_all()を実行."""
        comparison = ABTestComparison(clear_difference_data)
        freq_result, bayes_result = comparison.run_all(test_method=TestMethod.Z_TEST)

        assert freq_result.method == TestMethod.Z_TEST

    def test_run_all_with_t_test(self, clear_difference_data):
        """t検定でrun_all()を実行."""
        comparison = ABTestComparison(clear_difference_data)
        freq_result, bayes_result = comparison.run_all(test_method=TestMethod.T_TEST)

        assert freq_result.method == TestMethod.T_TEST

    def test_run_all_with_chi_square(self, clear_difference_data):
        """カイ二乗検定でrun_all()を実行."""
        comparison = ABTestComparison(clear_difference_data)
        freq_result, bayes_result = comparison.run_all(test_method=TestMethod.CHI_SQUARE)

        assert freq_result.method == TestMethod.CHI_SQUARE

    def test_run_all_bayesian_result_structure(self, clear_difference_data):
        """ベイジアン結果の構造が正しい."""
        comparison = ABTestComparison(clear_difference_data)
        _, bayes_result = comparison.run_all()

        assert hasattr(bayes_result, 'prob_b_better')
        assert hasattr(bayes_result, 'prob_a_better')
        assert hasattr(bayes_result, 'mean_a')
        assert hasattr(bayes_result, 'mean_b')

    def test_run_all_frequentist_result_structure(self, clear_difference_data):
        """頻度主義結果の構造が正しい."""
        comparison = ABTestComparison(clear_difference_data)
        freq_result, _ = comparison.run_all()

        assert hasattr(freq_result, 'method')
        assert hasattr(freq_result, 'p_value')
        assert hasattr(freq_result, 'test_statistic')
        assert hasattr(freq_result, 'is_significant')


class TestABTestComparisonCompareResults:
    """compare_results()メソッドのテスト."""

    def test_compare_results_structure(self, clear_difference_data):
        """compare_results()が正しい構造を返す."""
        comparison = ABTestComparison(clear_difference_data)
        freq_result, bayes_result = comparison.run_all()
        comparison_dict = comparison.compare_results(freq_result, bayes_result)

        assert "data" in comparison_dict
        assert "frequentist" in comparison_dict
        assert "bayesian" in comparison_dict
        assert "agreement" in comparison_dict

    def test_compare_results_data_section(self, clear_difference_data):
        """dataセクションが正しい."""
        comparison = ABTestComparison(clear_difference_data)
        freq_result, bayes_result = comparison.run_all()
        comparison_dict = comparison.compare_results(freq_result, bayes_result)

        data_section = comparison_dict["data"]
        assert "cvr_a" in data_section
        assert "cvr_b" in data_section
        assert "cvr_diff" in data_section

        assert data_section["cvr_a"] == clear_difference_data.cvr_a
        assert data_section["cvr_b"] == clear_difference_data.cvr_b
        assert data_section["cvr_diff"] == clear_difference_data.cvr_diff

    def test_compare_results_frequentist_section(self, clear_difference_data):
        """frequentistセクションが正しい."""
        comparison = ABTestComparison(clear_difference_data)
        freq_result, bayes_result = comparison.run_all()
        comparison_dict = comparison.compare_results(freq_result, bayes_result)

        freq_section = comparison_dict["frequentist"]
        assert "method" in freq_section
        assert "p_value" in freq_section
        assert "is_significant" in freq_section
        assert "ci" in freq_section

    def test_compare_results_bayesian_section(self, clear_difference_data):
        """bayesianセクションが正しい."""
        comparison = ABTestComparison(clear_difference_data)
        freq_result, bayes_result = comparison.run_all()
        comparison_dict = comparison.compare_results(freq_result, bayes_result)

        bayes_section = comparison_dict["bayesian"]
        assert "prob_b_better" in bayes_section
        assert "diff_mean" in bayes_section
        assert "credible_interval" in bayes_section

    def test_compare_results_agreement_clear_difference(self, clear_difference_data):
        """明確な差がある場合、両アプローチが一致."""
        comparison = ABTestComparison(clear_difference_data)
        freq_result, bayes_result = comparison.run_all()
        comparison_dict = comparison.compare_results(freq_result, bayes_result)

        # 両方とも有意/高確率なので一致
        assert comparison_dict["agreement"] == True

    def test_compare_results_agreement_no_difference(self, no_difference_data):
        """差がない場合、両アプローチが一致."""
        comparison = ABTestComparison(no_difference_data)
        freq_result, bayes_result = comparison.run_all()
        comparison_dict = comparison.compare_results(freq_result, bayes_result)

        # 両方とも非有意/低確率なので一致
        assert comparison_dict["agreement"] == True


class TestABTestComparisonIntegration:
    """統合テスト."""

    def test_full_workflow_clear_difference(self, clear_difference_data):
        """明確な差がある場合のフルワークフロー."""
        comparison = ABTestComparison(clear_difference_data)
        freq_result, bayes_result = comparison.run_all()
        comparison_dict = comparison.compare_results(freq_result, bayes_result)

        # 頻度主義: 有意
        assert freq_result.is_significant == True

        # ベイジアン: 高確率
        assert bayes_result.prob_b_better > 0.95

        # 一致
        assert comparison_dict["agreement"] == True

    def test_full_workflow_no_difference(self, no_difference_data):
        """差がない場合のフルワークフロー."""
        comparison = ABTestComparison(no_difference_data)
        freq_result, bayes_result = comparison.run_all()
        comparison_dict = comparison.compare_results(freq_result, bayes_result)

        # 頻度主義: 非有意
        assert freq_result.is_significant == False

        # ベイジアン: ほぼ50%
        assert 0.4 < bayes_result.prob_b_better < 0.6

        # 一致
        assert comparison_dict["agreement"] == True

    def test_full_workflow_subtle_difference(self, subtle_difference_data):
        """微妙な差がある場合のフルワークフロー."""
        comparison = ABTestComparison(subtle_difference_data)
        freq_result, bayes_result = comparison.run_all()
        comparison_dict = comparison.compare_results(freq_result, bayes_result)

        # 結果が得られることを確認
        assert freq_result is not None
        assert bayes_result is not None
        assert comparison_dict is not None

    def test_full_workflow_small_sample(self, small_sample_data):
        """小サンプルの場合のフルワークフロー."""
        comparison = ABTestComparison(small_sample_data)
        freq_result, bayes_result = comparison.run_all()
        comparison_dict = comparison.compare_results(freq_result, bayes_result)

        # 結果が得られることを確認
        assert freq_result is not None
        assert bayes_result is not None
        assert comparison_dict is not None

    def test_different_test_methods_consistency(self, clear_difference_data):
        """異なる検定方法でも一貫性がある."""
        comparison = ABTestComparison(clear_difference_data)

        freq_z, bayes_z = comparison.run_all(TestMethod.Z_TEST)
        freq_t, bayes_t = comparison.run_all(TestMethod.T_TEST)
        freq_chi, bayes_chi = comparison.run_all(TestMethod.CHI_SQUARE)

        # ベイジアンは検定方法によらず同じ（乱数があるので完全一致はしない）
        assert bayes_z.prob_b_better == pytest.approx(bayes_t.prob_b_better, abs=0.01)
        assert bayes_z.prob_b_better == pytest.approx(bayes_chi.prob_b_better, abs=0.01)

        # 頻度主義は全て有意
        assert freq_z.is_significant == True
        assert freq_t.is_significant == True
        assert freq_chi.is_significant == True

    def test_credible_and_confidence_intervals_similar(self, clear_difference_data):
        """信頼区間と確信区間が類似（大サンプル）."""
        comparison = ABTestComparison(clear_difference_data)
        freq_result, bayes_result = comparison.run_all()

        # 大サンプルでは区間が類似
        freq_width = freq_result.ci_upper - freq_result.ci_lower
        bayes_width = bayes_result.diff_ci_upper - bayes_result.diff_ci_lower

        # おおよそ同じ幅（誤差20%以内）
        assert freq_width == pytest.approx(bayes_width, rel=0.2)

    def test_multiple_runs_consistency(self, clear_difference_data):
        """複数回実行しても一貫性がある."""
        comparison = ABTestComparison(clear_difference_data)

        # 頻度主義は決定的
        freq1, _ = comparison.run_all()
        freq2, _ = comparison.run_all()

        assert freq1.p_value == freq2.p_value
        assert freq1.test_statistic == freq2.test_statistic

        # ベイジアンは乱数シード固定で一致
        import numpy as np
        np.random.seed(42)
        _, bayes1 = comparison.run_all()

        np.random.seed(42)
        _, bayes2 = comparison.run_all()

        assert bayes1.prob_b_better == bayes2.prob_b_better


class TestABTestComparisonEdgeCases:
    """エッジケースの統合テスト."""

    def test_zero_conversion_both(self, zero_conversion_both_data):
        """両グループともコンバージョンが0."""
        comparison = ABTestComparison(zero_conversion_both_data)

        # Z検定を使用（t検定は分散0でエラーの可能性）
        freq_result, bayes_result = comparison.run_all(test_method=TestMethod.Z_TEST)

        # ベイジアン: ほぼ50%
        assert 0.4 < bayes_result.prob_b_better < 0.6

    def test_perfect_conversion_b(self, perfect_conversion_b_data):
        """グループBのコンバージョンが100%."""
        comparison = ABTestComparison(perfect_conversion_b_data)
        freq_result, bayes_result = comparison.run_all()

        # 頻度主義: 有意
        assert freq_result.is_significant == True

        # ベイジアン: 高確率
        assert bayes_result.prob_b_better > 0.95

    def test_extreme_difference(self, extreme_difference_data):
        """極端な差がある場合."""
        comparison = ABTestComparison(extreme_difference_data)
        freq_result, bayes_result = comparison.run_all()

        # 頻度主義: 有意
        assert freq_result.is_significant == True
        assert freq_result.p_value < 0.001

        # ベイジアン: 非常に高確率
        assert bayes_result.prob_b_better > 0.9999
