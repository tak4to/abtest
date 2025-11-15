import numpy as np
from scipy import stats, integrate
from typing import Tuple

from src.test_data import TestData
from src.results import BayesianResult


class BayesianABTest:
    """
    ベイジアンA/Bテスト（Beta-Binomial共役モデル）

    Beta分布を事前分布として使用し、二項分布の尤度と組み合わせることで、
    事後分布が解析的にBeta分布として求まります（共役性）。

    **MCMCが不要な理由:**
    Beta-Binomial共役性により、事後分布が解析的に求まるため、
    MCMCやその他の近似手法は不要です。Beta分布から直接サンプリング
    することで、正確かつ高速に確率的推論を行うことができます。

    **数式:**
    - 事前分布: p ~ Beta(α_prior, β_prior)
    - 尤度: conversions ~ Binomial(n, p)
    - 事後分布: p | data ~ Beta(α_prior + conversions, β_prior + n - conversions)

    Attributes
    ----------
    data : TestData
        A/Bテストのデータ
    alpha_prior : float
        事前分布のαパラメータ（デフォルト1.0は無情報事前分布）
    beta_prior : float
        事前分布のβパラメータ（デフォルト1.0は無情報事前分布）
    credible_level : float
        確信水準
    n_samples : int
        モンテカルロサンプル数
    alpha_post_a : float
        グループAの事後分布のαパラメータ
    beta_post_a : float
        グループAの事後分布のβパラメータ
    alpha_post_b : float
        グループBの事後分布のαパラメータ
    beta_post_b : float
        グループBの事後分布のβパラメータ
    """
    
    def __init__(
        self,
        data: TestData,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        credible_level: float = 0.95,
        n_samples: int = 100000
    ):
        """
        Parameters
        ----------
        data : TestData
            A/Bテストのデータ
        alpha_prior : float, optional
            事前分布のαパラメータ（デフォルト: 1.0 = 無情報事前分布）
        beta_prior : float, optional
            事前分布のβパラメータ（デフォルト: 1.0 = 無情報事前分布）
        credible_level : float, optional
            確信水準（デフォルト: 0.95）
        n_samples : int, optional
            モンテカルロシミュレーションのサンプル数（デフォルト: 100000）
        """
        self.data = data
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.credible_level = credible_level
        self.n_samples = n_samples
        
        # 事後分布のパラメータを計算
        self.alpha_post_a = alpha_prior + data.conv_a
        self.beta_post_a = beta_prior + (data.n_a - data.conv_a)
        self.alpha_post_b = alpha_prior + data.conv_b
        self.beta_post_b = beta_prior + (data.n_b - data.conv_b)
    
    def sample_posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        事後分布からサンプリング
        
        Returns
        -------
        samples_a : np.ndarray
            グループAのサンプル
        samples_b : np.ndarray
            グループBのサンプル
        """
        samples_a = np.random.beta(self.alpha_post_a, self.beta_post_a, self.n_samples)
        samples_b = np.random.beta(self.alpha_post_b, self.beta_post_b, self.n_samples)
        return samples_a, samples_b
    
    def calculate_probability(
        self,
        samples_a: np.ndarray,
        samples_b: np.ndarray
    ) -> Tuple[float, float]:
        """
        BがAより優れている確率を計算
        
        Parameters
        ----------
        samples_a : np.ndarray
            グループAのサンプル
        samples_b : np.ndarray
            グループBのサンプル
        
        Returns
        -------
        prob_b_better : float
            BがAより優れている確率
        prob_a_better : float
            AがBより優れている確率
        """
        prob_b_better = np.mean(samples_b > samples_a)
        prob_a_better = 1 - prob_b_better
        return prob_b_better, prob_a_better
    
    def calculate_expected_loss(
        self,
        samples_a: np.ndarray,
        samples_b: np.ndarray
    ) -> Tuple[float, float]:
        """
        期待損失を計算
        
        各選択肢を選んだ場合の期待される損失を計算します。
        損失は「選ばなかった方が良かった場合」の差分の期待値です。
        
        Parameters
        ----------
        samples_a : np.ndarray
            グループAのサンプル
        samples_b : np.ndarray
            グループBのサンプル
        
        Returns
        -------
        loss_choose_a : float
            Aを選んだ場合の期待損失
        loss_choose_b : float
            Bを選んだ場合の期待損失
        """
        loss_choose_a = np.mean(np.maximum(samples_b - samples_a, 0))
        loss_choose_b = np.mean(np.maximum(samples_a - samples_b, 0))
        return loss_choose_a, loss_choose_b
    
    def calculate_bayes_factor(self, prob_b_better: float) -> float:
        """
        ベイズファクター（オッズ比）を計算

        注意: この実装は簡易版で、オッズ比 P(B > A) / P(A > B) を計算します。
        厳密なベイズファクターは、モデル比較 P(Data|H1) / P(Data|H0) ですが、
        実用上はこのオッズ比が「BがAより優れている」証拠の強さの指標として有用です。

        解釈の目安:
        - BF < 1: Aが優れている証拠
        - BF = 1: どちらとも言えない
        - 1 < BF < 3: Bが優れている弱い証拠
        - 3 < BF < 10: Bが優れている中程度の証拠
        - BF > 10: Bが優れている強い証拠

        Parameters
        ----------
        prob_b_better : float
            BがAより優れている確率

        Returns
        -------
        float
            ベイズファクター（オッズ比）
        """
        if prob_b_better == 0:
            return 0.0
        if prob_b_better == 1:
            return float('inf')
        return prob_b_better / (1 - prob_b_better)
    
    def probability_analytical(self) -> float:
        """
        解析的にP(B > A)を計算（オプション）
        
        数値積分を使用して正確な確率を計算します。
        モンテカルロよりも計算時間がかかりますが、より正確です。
        
        Returns
        -------
        float
            BがAより優れている確率
        """
        def integrand(x):
            return (
                stats.beta.pdf(x, self.alpha_post_a, self.beta_post_a) * 
                stats.beta.cdf(x, self.alpha_post_b, self.beta_post_b)
            )
        
        result, _ = integrate.quad(integrand, 0, 1)
        return 1 - result
    
    def run(
        self,
        calculate_loss: bool = True,
        calculate_bf: bool = True
    ) -> BayesianResult:
        """
        ベイジアン分析を実行
        
        Parameters
        ----------
        calculate_loss : bool, optional
            期待損失を計算するか（デフォルト: True）
        calculate_bf : bool, optional
            ベイズファクターを計算するか（デフォルト: True）
        
        Returns
        -------
        BayesianResult
            分析結果
        """
        # サンプリング
        samples_a, samples_b = self.sample_posterior()
        
        # 確率の計算
        prob_b_better, prob_a_better = self.calculate_probability(samples_a, samples_b)
        
        # 期待値
        mean_a = self.alpha_post_a / (self.alpha_post_a + self.beta_post_a)
        mean_b = self.alpha_post_b / (self.alpha_post_b + self.beta_post_b)
        
        # 差の分布
        diff_samples = samples_b - samples_a
        diff_mean = np.mean(diff_samples)
        diff_ci = np.percentile(
            diff_samples,
            [(1 - self.credible_level) * 100 / 2, (1 + self.credible_level) * 100 / 2]
        )
        
        # オプション計算
        expected_loss_a = None
        expected_loss_b = None
        if calculate_loss:
            expected_loss_a, expected_loss_b = self.calculate_expected_loss(
                samples_a, samples_b
            )
        
        bayes_factor = None
        if calculate_bf:
            bayes_factor = self.calculate_bayes_factor(prob_b_better)
        
        return BayesianResult(
            prob_b_better=prob_b_better,
            prob_a_better=prob_a_better,
            mean_a=mean_a,
            mean_b=mean_b,
            diff_mean=diff_mean,
            diff_ci_lower=diff_ci[0],
            diff_ci_upper=diff_ci[1],
            credible_level=self.credible_level,
            alpha_post_a=self.alpha_post_a,
            beta_post_a=self.beta_post_a,
            alpha_post_b=self.alpha_post_b,
            beta_post_b=self.beta_post_b,
            n_samples=self.n_samples,
            expected_loss_a=expected_loss_a,
            expected_loss_b=expected_loss_b,
            bayes_factor=bayes_factor
        )
