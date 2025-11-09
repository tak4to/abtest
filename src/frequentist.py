import numpy as np
from scipy import stats

from src.test_data import TestData, TestMethod
from src.results import FrequentistResult


class FrequentistABTest:
    """
    頻度論的A/Bテスト
    
    統計仮説検定を使用してA/Bテストを実行します。
    z検定、t検定（Welchの方法）、カイ二乗検定をサポートしています。
    
    Attributes
    ----------
    data : TestData
        A/Bテストのデータ
    confidence_level : float
        信頼水準（デフォルト: 0.95）
    alpha : float
        有意水準（1 - confidence_level）
    """
    
    def __init__(self, data: TestData, confidence_level: float = 0.95):
        """
        Parameters
        ----------
        data : TestData
            A/Bテストのデータ
        confidence_level : float, optional
            信頼水準（デフォルト: 0.95）
        """
        self.data = data
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def z_test(self) -> FrequentistResult:
        """
        z検定（正規近似）を実行
        
        大サンプルサイズの場合に適しています（n ≥ 30が目安）。
        
        Returns
        -------
        FrequentistResult
            検定結果
        """
        p_a = self.data.cvr_a
        p_b = self.data.cvr_b
        n_a = self.data.n_a
        n_b = self.data.n_b
        
        # プールされた比率
        p_pool = (self.data.conv_a + self.data.conv_b) / (n_a + n_b)
        se_pool = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        
        # z統計量とp値
        z_score = (p_b - p_a) / se_pool
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # 信頼区間
        se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        ci_lower = (p_b - p_a) - z_critical * se_diff
        ci_upper = (p_b - p_a) + z_critical * se_diff
        
        return FrequentistResult(
            method=TestMethod.Z_TEST,
            test_statistic=z_score,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            is_significant=p_value < self.alpha,
            additional_info={
                "pooled_proportion": p_pool,
                "standard_error": se_diff
            }
        )
    
    def t_test(self) -> FrequentistResult:
        """
        t検定（Welchの方法）を実行
        
        小〜中サンプルサイズの場合や、
        2群の分散が等しくない場合に適しています。
        
        Returns
        -------
        FrequentistResult
            検定結果
        """
        p_a = self.data.cvr_a
        p_b = self.data.cvr_b
        n_a = self.data.n_a
        n_b = self.data.n_b
        
        # 各グループの分散
        var_a = p_a * (1 - p_a) / n_a
        var_b = p_b * (1 - p_b) / n_b
        
        # t統計量
        t_score = (p_b - p_a) / np.sqrt(var_a + var_b)
        
        # Welchの自由度
        df = (var_a + var_b)**2 / (var_a**2 / (n_a - 1) + var_b**2 / (n_b - 1))
        
        # p値
        p_value = 2 * (1 - stats.t.cdf(abs(t_score), df))
        
        # 信頼区間
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        se_diff = np.sqrt(var_a + var_b)
        ci_lower = (p_b - p_a) - t_critical * se_diff
        ci_upper = (p_b - p_a) + t_critical * se_diff
        
        return FrequentistResult(
            method=TestMethod.T_TEST,
            test_statistic=t_score,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            is_significant=p_value < self.alpha,
            additional_info={
                "degrees_of_freedom": df,
                "variance_a": var_a,
                "variance_b": var_b
            }
        )
    
    def chi_square_test(self) -> FrequentistResult:
        """
        カイ二乗検定を実行
        
        カテゴリカルデータの独立性を直接検定します。
        
        Returns
        -------
        FrequentistResult
            検定結果
        """
        # 分割表
        observed = np.array([
            [self.data.conv_a, self.data.n_a - self.data.conv_a],
            [self.data.conv_b, self.data.n_b - self.data.conv_b]
        ])
        
        # カイ二乗検定
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(
            observed, correction=False
        )
        
        # Yates補正版
        chi2_yates, p_value_yates, _, _ = stats.chi2_contingency(
            observed, correction=True
        )
        
        # Wilson score methodによる信頼区間
        def wilson_ci(x, n):
            p_hat = x / n
            z = stats.norm.ppf(1 - self.alpha / 2)
            denominator = 1 + z**2 / n
            center = (p_hat + z**2 / (2 * n)) / denominator
            margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
            return center - margin, center + margin
        
        ci_a = wilson_ci(self.data.conv_a, self.data.n_a)
        ci_b = wilson_ci(self.data.conv_b, self.data.n_b)
        
        # 差の信頼区間（近似）
        ci_lower = ci_b[0] - ci_a[1]
        ci_upper = ci_b[1] - ci_a[0]
        
        return FrequentistResult(
            method=TestMethod.CHI_SQUARE,
            test_statistic=chi2_stat,
            p_value=p_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            is_significant=p_value < self.alpha,
            additional_info={
                "degrees_of_freedom": dof,
                "chi2_yates": chi2_yates,
                "p_value_yates": p_value_yates,
                "observed": observed.tolist(),
                "expected": expected.tolist(),
                "ci_a": ci_a,
                "ci_b": ci_b
            }
        )
    
    def run(self, method: TestMethod) -> FrequentistResult:
        """
        指定された方法で検定を実行
        
        Parameters
        ----------
        method : TestMethod
            検定方法（Z_TEST, T_TEST, CHI_SQUARE）
        
        Returns
        -------
        FrequentistResult
            検定結果
        
        Raises
        ------
        ValueError
            未知の検定方法が指定された場合
        """
        if method == TestMethod.Z_TEST:
            return self.z_test()
        elif method == TestMethod.T_TEST:
            return self.t_test()
        elif method == TestMethod.CHI_SQUARE:
            return self.chi_square_test()
        else:
            raise ValueError(f"Unknown test method: {method}")