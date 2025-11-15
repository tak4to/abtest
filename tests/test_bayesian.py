"""Tests for BayesianABTest class."""

import pytest
import numpy as np
from scipy import stats

from src.bayesian import BayesianABTest
from src.test_data import TestData


class TestBayesianPosteriorParameters:
    """事後分布のパラメータ計算テスト."""

    def test_posterior_parameters_clear_difference(self, clear_difference_data):
        """明確な差がある場合の事後分布パラメータ."""
        test = BayesianABTest(clear_difference_data)

        # 事前分布: Beta(1, 1)
        # データ: グループA = 100/1000, グループB = 150/1000
        # 事後分布: A = Beta(1+100, 1+900), B = Beta(1+150, 1+850)
        assert test.alpha_post_a == 101.0
        assert test.beta_post_a == 901.0
        assert test.alpha_post_b == 151.0
        assert test.beta_post_b == 851.0

    def test_posterior_parameters_no_difference(self, no_difference_data):
        """差がない場合の事後分布パラメータ."""
        test = BayesianABTest(no_difference_data)

        assert test.alpha_post_a == 101.0
        assert test.beta_post_a == 901.0
        assert test.alpha_post_b == 101.0
        assert test.beta_post_b == 901.0

    def test_posterior_parameters_with_custom_prior(self, clear_difference_data):
        """カスタム事前分布での事後分布パラメータ."""
        test = BayesianABTest(clear_difference_data, alpha_prior=2.0, beta_prior=2.0)

        # 事前分布: Beta(2, 2)
        # 事後分布: A = Beta(2+100, 2+900), B = Beta(2+150, 2+850)
        assert test.alpha_post_a == 102.0
        assert test.beta_post_a == 902.0
        assert test.alpha_post_b == 152.0
        assert test.beta_post_b == 852.0

    def test_posterior_parameters_zero_conversion(self, zero_conversion_a_data):
        """コンバージョンが0の場合の事後分布パラメータ."""
        test = BayesianABTest(zero_conversion_a_data)

        assert test.alpha_post_a == 1.0  # 1 + 0
        assert test.beta_post_a == 101.0  # 1 + 100
        assert test.alpha_post_b == 11.0  # 1 + 10
        assert test.beta_post_b == 91.0  # 1 + 90


class TestBayesianSampling:
    """サンプリング機能のテスト."""

    def test_sample_posterior_shape(self, clear_difference_data):
        """サンプルの形状が正しい."""
        test = BayesianABTest(clear_difference_data, n_samples=1000)
        samples_a, samples_b = test.sample_posterior()

        assert samples_a.shape == (1000,)
        assert samples_b.shape == (1000,)

    def test_sample_posterior_range(self, clear_difference_data):
        """サンプルが0-1の範囲内."""
        test = BayesianABTest(clear_difference_data, n_samples=1000)
        samples_a, samples_b = test.sample_posterior()

        assert np.all(samples_a >= 0)
        assert np.all(samples_a <= 1)
        assert np.all(samples_b >= 0)
        assert np.all(samples_b <= 1)

    def test_sample_posterior_mean_matches_theory(self, clear_difference_data):
        """サンプルの平均が理論値と一致."""
        test = BayesianABTest(clear_difference_data, n_samples=100000)
        samples_a, samples_b = test.sample_posterior()

        # Beta分布の期待値: α / (α + β)
        expected_mean_a = test.alpha_post_a / (test.alpha_post_a + test.beta_post_a)
        expected_mean_b = test.alpha_post_b / (test.alpha_post_b + test.beta_post_b)

        assert np.mean(samples_a) == pytest.approx(expected_mean_a, abs=0.001)
        assert np.mean(samples_b) == pytest.approx(expected_mean_b, abs=0.001)

    def test_sample_posterior_variance_matches_theory(self, clear_difference_data):
        """サンプルの分散が理論値と一致."""
        test = BayesianABTest(clear_difference_data, n_samples=100000)
        samples_a, samples_b = test.sample_posterior()

        # Beta分布の分散: αβ / ((α+β)^2 * (α+β+1))
        def beta_variance(alpha, beta):
            return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

        expected_var_a = beta_variance(test.alpha_post_a, test.beta_post_a)
        expected_var_b = beta_variance(test.alpha_post_b, test.beta_post_b)

        assert np.var(samples_a) == pytest.approx(expected_var_a, abs=0.0001)
        assert np.var(samples_b) == pytest.approx(expected_var_b, abs=0.0001)


class TestBayesianProbabilityCalculation:
    """確率計算のテスト."""

    def test_probability_clear_difference(self, clear_difference_data):
        """明確な差がある場合の確率計算."""
        test = BayesianABTest(clear_difference_data, n_samples=100000)
        samples_a, samples_b = test.sample_posterior()
        prob_b_better, prob_a_better = test.calculate_probability(samples_a, samples_b)

        # BがAより明らかに優れている
        assert prob_b_better > 0.99
        assert prob_a_better < 0.01
        assert prob_b_better + prob_a_better == pytest.approx(1.0, abs=1e-10)

    def test_probability_no_difference(self, no_difference_data):
        """差がない場合の確率計算."""
        test = BayesianABTest(no_difference_data, n_samples=100000)
        samples_a, samples_b = test.sample_posterior()
        prob_b_better, prob_a_better = test.calculate_probability(samples_a, samples_b)

        # ほぼ50/50
        assert prob_b_better == pytest.approx(0.5, abs=0.05)
        assert prob_a_better == pytest.approx(0.5, abs=0.05)
        assert prob_b_better + prob_a_better == pytest.approx(1.0, abs=1e-10)

    def test_probability_sum_to_one(self, subtle_difference_data):
        """確率の合計が1."""
        test = BayesianABTest(subtle_difference_data, n_samples=100000)
        samples_a, samples_b = test.sample_posterior()
        prob_b_better, prob_a_better = test.calculate_probability(samples_a, samples_b)

        assert prob_b_better + prob_a_better == pytest.approx(1.0, abs=1e-10)

    def test_probability_analytical_vs_monte_carlo(self, clear_difference_data):
        """解析的計算とモンテカルロの一致."""
        test = BayesianABTest(clear_difference_data, n_samples=100000)

        # モンテカルロ
        samples_a, samples_b = test.sample_posterior()
        prob_b_better_mc, _ = test.calculate_probability(samples_a, samples_b)

        # 解析的計算
        prob_b_better_analytical = test.probability_analytical()

        # 差が0.01未満であることを確認（モンテカルロの誤差を考慮）
        assert prob_b_better_mc == pytest.approx(prob_b_better_analytical, abs=0.01)

    def test_probability_analytical_extreme_difference(self, extreme_difference_data):
        """極端な差がある場合の解析的計算."""
        test = BayesianABTest(extreme_difference_data)
        prob_b_better = test.probability_analytical()

        # ほぼ確実にBが優れている
        assert prob_b_better > 0.9999


class TestBayesianExpectedLoss:
    """期待損失のテスト."""

    def test_expected_loss_non_negative(self, clear_difference_data):
        """期待損失は非負."""
        test = BayesianABTest(clear_difference_data, n_samples=100000)
        samples_a, samples_b = test.sample_posterior()
        loss_a, loss_b = test.calculate_expected_loss(samples_a, samples_b)

        assert loss_a >= 0
        assert loss_b >= 0

    def test_expected_loss_clear_difference(self, clear_difference_data):
        """明確な差がある場合、Aを選ぶと大きな損失."""
        test = BayesianABTest(clear_difference_data, n_samples=100000)
        samples_a, samples_b = test.sample_posterior()
        loss_a, loss_b = test.calculate_expected_loss(samples_a, samples_b)

        # Aを選ぶと損失が大きい
        assert loss_a > loss_b
        assert loss_a > 0.04  # 理論的には約0.05

    def test_expected_loss_no_difference(self, no_difference_data):
        """差がない場合、期待損失はほぼ同じ."""
        test = BayesianABTest(no_difference_data, n_samples=100000)
        samples_a, samples_b = test.sample_posterior()
        loss_a, loss_b = test.calculate_expected_loss(samples_a, samples_b)

        # ほぼ同じ
        assert loss_a == pytest.approx(loss_b, abs=0.005)

    def test_expected_loss_sum_property(self, clear_difference_data):
        """期待損失の計算が正しい（数学的性質の確認）."""
        test = BayesianABTest(clear_difference_data, n_samples=100000)
        samples_a, samples_b = test.sample_posterior()
        loss_a, loss_b = test.calculate_expected_loss(samples_a, samples_b)

        # E[max(B-A, 0)] + E[max(A-B, 0)] = E[|B-A|]
        diff_samples = samples_b - samples_a
        expected_abs_diff = np.mean(np.abs(diff_samples))

        assert loss_a + loss_b == pytest.approx(expected_abs_diff, abs=0.001)


class TestBayesianBayesFactor:
    """ベイズファクター（オッズ比）のテスト."""

    def test_bayes_factor_clear_difference(self):
        """明確な差がある場合のベイズファクター."""
        test = BayesianABTest(TestData(n_a=1000, conv_a=100, n_b=1000, conv_b=150))

        # P(B > A) ≈ 0.9996
        bf = test.calculate_bayes_factor(0.9996)

        # BF = 0.9996 / 0.0004 = 2499
        assert bf > 1000  # 強い証拠

    def test_bayes_factor_no_difference(self):
        """差がない場合のベイズファクター."""
        bf = BayesianABTest(TestData(n_a=100, conv_a=10, n_b=100, conv_b=10)).calculate_bayes_factor(0.5)

        # BF = 0.5 / 0.5 = 1
        assert bf == pytest.approx(1.0, abs=1e-10)

    def test_bayes_factor_zero_probability(self):
        """確率が0の場合のベイズファクター."""
        bf = BayesianABTest(TestData(n_a=100, conv_a=10, n_b=100, conv_b=10)).calculate_bayes_factor(0.0)

        assert bf == 0.0

    def test_bayes_factor_one_probability(self):
        """確率が1の場合のベイズファクター."""
        bf = BayesianABTest(TestData(n_a=100, conv_a=10, n_b=100, conv_b=10)).calculate_bayes_factor(1.0)

        assert bf == float('inf')

    def test_bayes_factor_interpretation(self):
        """ベイズファクターの解釈範囲."""
        test = BayesianABTest(TestData(n_a=100, conv_a=10, n_b=100, conv_b=10))

        # 弱い証拠: 1 < BF < 3
        bf_weak = test.calculate_bayes_factor(0.75)  # 0.75 / 0.25 = 3
        assert 1 < bf_weak <= 3

        # 中程度の証拠: 3 < BF < 10
        bf_moderate = test.calculate_bayes_factor(0.833)  # 0.833 / 0.167 ≈ 5
        assert 3 < bf_moderate <= 10

        # 強い証拠: BF > 10
        bf_strong = test.calculate_bayes_factor(0.95)  # 0.95 / 0.05 = 19
        assert bf_strong > 10


class TestBayesianRun:
    """run()メソッドの統合テスト."""

    def test_run_returns_correct_structure(self, clear_difference_data):
        """run()が正しい構造の結果を返す."""
        test = BayesianABTest(clear_difference_data)
        result = test.run()

        assert hasattr(result, 'prob_b_better')
        assert hasattr(result, 'prob_a_better')
        assert hasattr(result, 'mean_a')
        assert hasattr(result, 'mean_b')
        assert hasattr(result, 'diff_mean')
        assert hasattr(result, 'diff_ci_lower')
        assert hasattr(result, 'diff_ci_upper')
        assert hasattr(result, 'expected_loss_a')
        assert hasattr(result, 'expected_loss_b')
        assert hasattr(result, 'bayes_factor')

    def test_run_mean_calculation(self, clear_difference_data):
        """run()の平均計算が正確."""
        test = BayesianABTest(clear_difference_data)
        result = test.run()

        # 事後分布の期待値
        expected_mean_a = test.alpha_post_a / (test.alpha_post_a + test.beta_post_a)
        expected_mean_b = test.alpha_post_b / (test.alpha_post_b + test.beta_post_b)

        assert result.mean_a == pytest.approx(expected_mean_a, abs=1e-10)
        assert result.mean_b == pytest.approx(expected_mean_b, abs=1e-10)

    def test_run_credible_interval_contains_mean(self, clear_difference_data):
        """確信区間が平均を含む."""
        test = BayesianABTest(clear_difference_data)
        result = test.run()

        assert result.diff_ci_lower <= result.diff_mean <= result.diff_ci_upper

    def test_run_credible_interval_width(self, clear_difference_data):
        """確信区間の幅が妥当."""
        test = BayesianABTest(clear_difference_data, credible_level=0.95)
        result = test.run()

        # 95%確信区間なので、幅が0より大きい
        width = result.diff_ci_upper - result.diff_ci_lower
        assert width > 0

    def test_run_without_loss(self, clear_difference_data):
        """期待損失を計算しない場合."""
        test = BayesianABTest(clear_difference_data)
        result = test.run(calculate_loss=False)

        assert result.expected_loss_a is None
        assert result.expected_loss_b is None

    def test_run_without_bayes_factor(self, clear_difference_data):
        """ベイズファクターを計算しない場合."""
        test = BayesianABTest(clear_difference_data)
        result = test.run(calculate_bf=False)

        assert result.bayes_factor is None

    def test_run_consistency_multiple_calls(self, clear_difference_data):
        """複数回実行しても一貫性がある（乱数シード固定）."""
        test = BayesianABTest(clear_difference_data, n_samples=10000)

        np.random.seed(42)
        result1 = test.run()

        np.random.seed(42)
        result2 = test.run()

        # 同じシードなので、結果が一致
        assert result1.prob_b_better == result2.prob_b_better
        assert result1.diff_mean == result2.diff_mean


class TestBayesianEdgeCases:
    """エッジケースのテスト."""

    def test_zero_conversion_both_groups(self, zero_conversion_both_data):
        """両グループともコンバージョンが0."""
        test = BayesianABTest(zero_conversion_both_data)
        result = test.run()

        # ほぼ同じ
        assert result.prob_b_better == pytest.approx(0.5, abs=0.1)
        assert result.mean_a == pytest.approx(result.mean_b, abs=0.01)

    def test_perfect_conversion_b(self, perfect_conversion_b_data):
        """グループBのコンバージョンが100%."""
        test = BayesianABTest(perfect_conversion_b_data)
        result = test.run()

        # Bが明らかに優れている
        assert result.prob_b_better > 0.99

    def test_small_sample_stability(self, small_sample_data):
        """小サンプルでも安定した結果."""
        test = BayesianABTest(small_sample_data, n_samples=100000)
        result = test.run()

        # 確率の合計が1
        assert result.prob_b_better + result.prob_a_better == pytest.approx(1.0, abs=1e-10)

        # 期待損失が非負
        assert result.expected_loss_a >= 0
        assert result.expected_loss_b >= 0

    def test_custom_credible_level(self, clear_difference_data):
        """カスタム確信水準."""
        test_90 = BayesianABTest(clear_difference_data, credible_level=0.90)
        test_99 = BayesianABTest(clear_difference_data, credible_level=0.99)

        result_90 = test_90.run()
        result_99 = test_99.run()

        # 99%確信区間の方が広い
        width_90 = result_90.diff_ci_upper - result_90.diff_ci_lower
        width_99 = result_99.diff_ci_upper - result_99.diff_ci_lower

        assert width_99 > width_90
