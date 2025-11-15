"""Tests for result classes."""

import pytest
from src.results import BayesianResult, FrequentistResult
from src.test_data import TestMethod


class TestBayesianResult:
    """BayesianResultのテスト."""

    def test_bayesian_result_creation(self):
        """BayesianResultが正しく生成される."""
        result = BayesianResult(
            prob_b_better=0.95,
            prob_a_better=0.05,
            mean_a=0.1,
            mean_b=0.15,
            diff_mean=0.05,
            diff_ci_lower=0.03,
            diff_ci_upper=0.07,
            credible_level=0.95,
            alpha_post_a=101.0,
            beta_post_a=901.0,
            alpha_post_b=151.0,
            beta_post_b=851.0,
            n_samples=100000
        )

        assert result.prob_b_better == 0.95
        assert result.prob_a_better == 0.05
        assert result.mean_a == 0.1
        assert result.mean_b == 0.15
        assert result.diff_mean == 0.05
        assert result.diff_ci_lower == 0.03
        assert result.diff_ci_upper == 0.07
        assert result.credible_level == 0.95
        assert result.alpha_post_a == 101.0
        assert result.beta_post_a == 901.0
        assert result.alpha_post_b == 151.0
        assert result.beta_post_b == 851.0
        assert result.n_samples == 100000

    def test_bayesian_result_with_optional_fields(self):
        """オプションフィールド付きのBayesianResult."""
        result = BayesianResult(
            prob_b_better=0.95,
            prob_a_better=0.05,
            mean_a=0.1,
            mean_b=0.15,
            diff_mean=0.05,
            diff_ci_lower=0.03,
            diff_ci_upper=0.07,
            credible_level=0.95,
            alpha_post_a=101.0,
            beta_post_a=901.0,
            alpha_post_b=151.0,
            beta_post_b=851.0,
            n_samples=100000,
            expected_loss_a=0.001,
            expected_loss_b=0.049,
            bayes_factor=19.0
        )

        assert result.expected_loss_a == 0.001
        assert result.expected_loss_b == 0.049
        assert result.bayes_factor == 19.0

    def test_bayesian_result_summary(self):
        """summary()メソッドが文字列を返す."""
        result = BayesianResult(
            prob_b_better=0.95,
            prob_a_better=0.05,
            mean_a=0.1,
            mean_b=0.15,
            diff_mean=0.05,
            diff_ci_lower=0.03,
            diff_ci_upper=0.07,
            credible_level=0.95,
            alpha_post_a=101.0,
            beta_post_a=901.0,
            alpha_post_b=151.0,
            beta_post_b=851.0,
            n_samples=100000,
            expected_loss_a=0.001,
            expected_loss_b=0.049,
            bayes_factor=19.0
        )

        summary = result.summary()
        assert isinstance(summary, str)
        assert "95%" in summary
        assert "0.05" in summary

    def test_bayesian_result_summary_without_optional(self):
        """オプションフィールドなしでもsummary()が動作."""
        result = BayesianResult(
            prob_b_better=0.95,
            prob_a_better=0.05,
            mean_a=0.1,
            mean_b=0.15,
            diff_mean=0.05,
            diff_ci_lower=0.03,
            diff_ci_upper=0.07,
            credible_level=0.95,
            alpha_post_a=101.0,
            beta_post_a=901.0,
            alpha_post_b=151.0,
            beta_post_b=851.0,
            n_samples=100000
        )

        summary = result.summary()
        assert isinstance(summary, str)

    def test_bayesian_result_probabilities_sum_to_one(self):
        """確率の合計が1になることは保証されない（データクラスのみ）."""
        # BayesianResultは単なるデータクラスなので、バリデーションはしない
        result = BayesianResult(
            prob_b_better=0.6,
            prob_a_better=0.4,  # 合計1.0
            mean_a=0.1,
            mean_b=0.15,
            diff_mean=0.05,
            diff_ci_lower=0.03,
            diff_ci_upper=0.07,
            credible_level=0.95,
            alpha_post_a=101.0,
            beta_post_a=901.0,
            alpha_post_b=151.0,
            beta_post_b=851.0,
            n_samples=100000
        )

        assert result.prob_b_better + result.prob_a_better == pytest.approx(1.0)


class TestFrequentistResult:
    """FrequentistResultのテスト."""

    def test_frequentist_result_creation(self):
        """FrequentistResultが正しく生成される."""
        result = FrequentistResult(
            method=TestMethod.Z_TEST,
            test_statistic=3.5,
            p_value=0.0005,
            ci_lower=0.03,
            ci_upper=0.07,
            confidence_level=0.95,
            is_significant=True
        )

        assert result.method == TestMethod.Z_TEST
        assert result.test_statistic == 3.5
        assert result.p_value == 0.0005
        assert result.ci_lower == 0.03
        assert result.ci_upper == 0.07
        assert result.confidence_level == 0.95
        assert result.is_significant is True

    def test_frequentist_result_with_additional_info(self):
        """additional_info付きのFrequentistResult."""
        result = FrequentistResult(
            method=TestMethod.T_TEST,
            test_statistic=2.5,
            p_value=0.01,
            ci_lower=0.01,
            ci_upper=0.05,
            confidence_level=0.95,
            is_significant=True,
            additional_info={
                "degrees_of_freedom": 198.5,
                "variance_a": 0.0009,
                "variance_b": 0.00105
            }
        )

        assert result.additional_info["degrees_of_freedom"] == 198.5
        assert result.additional_info["variance_a"] == 0.0009
        assert result.additional_info["variance_b"] == 0.00105

    def test_frequentist_result_summary(self):
        """summary()メソッドが文字列を返す."""
        result = FrequentistResult(
            method=TestMethod.Z_TEST,
            test_statistic=3.5,
            p_value=0.0005,
            ci_lower=0.03,
            ci_upper=0.07,
            confidence_level=0.95,
            is_significant=True
        )

        summary = result.summary()
        assert isinstance(summary, str)
        assert "z_test" in summary
        assert "3.5" in summary or "3.50" in summary
        assert "0.0005" in summary

    def test_frequentist_result_summary_significant(self):
        """有意な場合のsummary()."""
        result = FrequentistResult(
            method=TestMethod.Z_TEST,
            test_statistic=3.5,
            p_value=0.0005,
            ci_lower=0.03,
            ci_upper=0.07,
            confidence_level=0.95,
            is_significant=True
        )

        summary = result.summary()
        assert "✅" in summary or "有意" in summary

    def test_frequentist_result_summary_not_significant(self):
        """非有意な場合のsummary()."""
        result = FrequentistResult(
            method=TestMethod.Z_TEST,
            test_statistic=1.0,
            p_value=0.3,
            ci_lower=-0.02,
            ci_upper=0.06,
            confidence_level=0.95,
            is_significant=False
        )

        summary = result.summary()
        assert "❌" in summary or "有意差なし" in summary

    def test_frequentist_result_different_methods(self):
        """異なる検定方法のFrequentistResult."""
        methods = [TestMethod.Z_TEST, TestMethod.T_TEST, TestMethod.CHI_SQUARE]

        for method in methods:
            result = FrequentistResult(
                method=method,
                test_statistic=2.0,
                p_value=0.05,
                ci_lower=0.0,
                ci_upper=0.05,
                confidence_level=0.95,
                is_significant=False
            )

            assert result.method == method

    def test_frequentist_result_significance_threshold(self):
        """有意性の判定が正しい."""
        # p < α で有意
        result_sig = FrequentistResult(
            method=TestMethod.Z_TEST,
            test_statistic=2.5,
            p_value=0.01,
            ci_lower=0.01,
            ci_upper=0.05,
            confidence_level=0.95,
            is_significant=True
        )

        # p >= α で非有意
        result_not_sig = FrequentistResult(
            method=TestMethod.Z_TEST,
            test_statistic=1.5,
            p_value=0.15,
            ci_lower=-0.01,
            ci_upper=0.05,
            confidence_level=0.95,
            is_significant=False
        )

        assert result_sig.is_significant is True
        assert result_not_sig.is_significant is False


class TestResultsConsistency:
    """結果クラス間の一貫性テスト."""

    def test_both_results_have_summary_method(self):
        """両方の結果クラスにsummary()メソッドがある."""
        bayesian = BayesianResult(
            prob_b_better=0.95,
            prob_a_better=0.05,
            mean_a=0.1,
            mean_b=0.15,
            diff_mean=0.05,
            diff_ci_lower=0.03,
            diff_ci_upper=0.07,
            credible_level=0.95,
            alpha_post_a=101.0,
            beta_post_a=901.0,
            alpha_post_b=151.0,
            beta_post_b=851.0,
            n_samples=100000
        )

        frequentist = FrequentistResult(
            method=TestMethod.Z_TEST,
            test_statistic=3.5,
            p_value=0.0005,
            ci_lower=0.03,
            ci_upper=0.07,
            confidence_level=0.95,
            is_significant=True
        )

        assert hasattr(bayesian, 'summary')
        assert hasattr(frequentist, 'summary')
        assert callable(bayesian.summary)
        assert callable(frequentist.summary)

    def test_confidence_and_credible_levels(self):
        """信頼水準と確信水準が同じ値を取れる."""
        bayesian = BayesianResult(
            prob_b_better=0.95,
            prob_a_better=0.05,
            mean_a=0.1,
            mean_b=0.15,
            diff_mean=0.05,
            diff_ci_lower=0.03,
            diff_ci_upper=0.07,
            credible_level=0.95,
            alpha_post_a=101.0,
            beta_post_a=901.0,
            alpha_post_b=151.0,
            beta_post_b=851.0,
            n_samples=100000
        )

        frequentist = FrequentistResult(
            method=TestMethod.Z_TEST,
            test_statistic=3.5,
            p_value=0.0005,
            ci_lower=0.03,
            ci_upper=0.07,
            confidence_level=0.95,
            is_significant=True
        )

        assert bayesian.credible_level == frequentist.confidence_level
