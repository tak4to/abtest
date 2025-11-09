from typing import Tuple, Dict, Any

from src.test_data import TestData, TestMethod
from src.results import FrequentistResult, BayesianResult
from src.frequentist import FrequentistABTest
from src.bayesian import BayesianABTest


class ABTestComparison:
    """
    A/Bテストの総合比較クラス
    
    頻度論的検定とベイジアン分析の両方を実行し、
    結果を比較するための統合インターフェースを提供します。
    
    Attributes
    ----------
    data : TestData
        A/Bテストのデータ
    frequentist : FrequentistABTest
        頻度論的検定のインスタンス
    bayesian : BayesianABTest
        ベイジアン分析のインスタンス
    """
    
    def __init__(
        self,
        data: TestData,
        confidence_level: float = 0.95,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0
    ):
        """
        Parameters
        ----------
        data : TestData
            A/Bテストのデータ
        confidence_level : float, optional
            信頼水準/確信水準（デフォルト: 0.95）
        alpha_prior : float, optional
            ベイジアンの事前分布αパラメータ（デフォルト: 1.0）
        beta_prior : float, optional
            ベイジアンの事前分布βパラメータ（デフォルト: 1.0）
        """
        self.data = data
        self.frequentist = FrequentistABTest(data, confidence_level)
        self.bayesian = BayesianABTest(
            data,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            credible_level=confidence_level
        )
    
    def run_all(
        self,
        test_method: TestMethod = TestMethod.Z_TEST
    ) -> Tuple[FrequentistResult, BayesianResult]:
        """
        頻度論的検定とベイジアン分析の両方を実行
        
        Parameters
        ----------
        test_method : TestMethod, optional
            頻度論的検定の方法（デフォルト: Z_TEST）
        
        Returns
        -------
        freq_result : FrequentistResult
            頻度論的検定の結果
        bayes_result : BayesianResult
            ベイジアン分析の結果
        """
        freq_result = self.frequentist.run(test_method)
        bayes_result = self.bayesian.run()
        return freq_result, bayes_result
    
    def compare_results(
        self,
        freq_result: FrequentistResult,
        bayes_result: BayesianResult
    ) -> Dict[str, Any]:
        """
        両方の結果を比較
        
        頻度論的アプローチとベイジアンアプローチの結果を比較し、
        一致度や主要な指標をまとめた辞書を返します。
        
        Parameters
        ----------
        freq_result : FrequentistResult
            頻度論的検定の結果
        bayes_result : BayesianResult
            ベイジアン分析の結果
        
        Returns
        -------
        dict
            比較結果を含む辞書
            
            - data: 基本データ（CVR等）
            - frequentist: 頻度論的検定の主要指標
            - bayesian: ベイジアン分析の主要指標
            - agreement: 両アプローチの結論が一致しているか
        """
        return {
            "data": {
                "cvr_a": self.data.cvr_a,
                "cvr_b": self.data.cvr_b,
                "cvr_diff": self.data.cvr_diff
            },
            "frequentist": {
                "method": freq_result.method.value,
                "p_value": freq_result.p_value,
                "is_significant": freq_result.is_significant,
                "ci": (freq_result.ci_lower, freq_result.ci_upper)
            },
            "bayesian": {
                "prob_b_better": bayes_result.prob_b_better,
                "diff_mean": bayes_result.diff_mean,
                "credible_interval": (
                    bayes_result.diff_ci_lower,
                    bayes_result.diff_ci_upper
                )
            },
            "agreement": freq_result.is_significant == (bayes_result.prob_b_better > 0.95)
        }