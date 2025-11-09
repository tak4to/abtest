from dataclasses import dataclass
from typing import Optional, Dict, Any
from src.test_data import TestMethod


@dataclass
class FrequentistResult:
    """
    頻度論的検定の結果
    
    Attributes
    ----------
    method : TestMethod
        使用した検定方法
    test_statistic : float
        検定統計量
    p_value : float
        p値
    ci_lower : float
        信頼区間の下限
    ci_upper : float
        信頼区間の上限
    confidence_level : float
        信頼水準
    is_significant : bool
        統計的に有意かどうか
    additional_info : Optional[Dict[str, Any]]
        追加情報（自由度、分散など）
    """
    method: TestMethod
    test_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    is_significant: bool
    additional_info: Optional[Dict[str, Any]] = None
    
    def summary(self) -> str:
        """
        結果のサマリーを文字列で返す
        
        Returns
        -------
        str
            結果のサマリー
        """
        lines = [
            f"検定方法: {self.method.value}",
            f"検定統計量: {self.test_statistic:.4f}",
            f"p値: {self.p_value:.6f}",
            f"信頼区間 ({self.confidence_level:.0%}): [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"判定: {'統計的に有意 ✅' if self.is_significant else '有意差なし ❌'}"
        ]
        return "\n".join(lines)


@dataclass
class BayesianResult:
    """
    ベイジアン検定の結果
    
    Attributes
    ----------
    prob_b_better : float
        BがAより優れている確率
    prob_a_better : float
        AがBより優れている確率
    mean_a : float
        グループAの事後分布の期待値
    mean_b : float
        グループBの事後分布の期待値
    diff_mean : float
        差の期待値
    diff_ci_lower : float
        差の確信区間の下限
    diff_ci_upper : float
        差の確信区間の上限
    credible_level : float
        確信水準
    alpha_post_a : float
        グループAの事後分布のαパラメータ
    beta_post_a : float
        グループAの事後分布のβパラメータ
    alpha_post_b : float
        グループBの事後分布のαパラメータ
    beta_post_b : float
        グループBの事後分布のβパラメータ
    n_samples : int
        モンテカルロサンプル数
    expected_loss_a : Optional[float]
        Aを選んだ場合の期待損失
    expected_loss_b : Optional[float]
        Bを選んだ場合の期待損失
    bayes_factor : Optional[float]
        ベイズファクター
    """
    prob_b_better: float
    prob_a_better: float
    mean_a: float
    mean_b: float
    diff_mean: float
    diff_ci_lower: float
    diff_ci_upper: float
    credible_level: float
    alpha_post_a: float
    beta_post_a: float
    alpha_post_b: float
    beta_post_b: float
    n_samples: int
    expected_loss_a: Optional[float] = None
    expected_loss_b: Optional[float] = None
    bayes_factor: Optional[float] = None
    
    def summary(self) -> str:
        """
        結果のサマリーを文字列で返す
        
        Returns
        -------
        str
            結果のサマリー
        """
        lines = [
            f"BがAより優れている確率: {self.prob_b_better:.2%}",
            f"AがBより優れている確率: {self.prob_a_better:.2%}",
            f"差の期待値 (B - A): {self.diff_mean:.4f}",
            f"差の{self.credible_level:.0%}確信区間: [{self.diff_ci_lower:.4f}, {self.diff_ci_upper:.4f}]",
        ]
        
        if self.bayes_factor is not None:
            lines.append(f"ベイズファクター: {self.bayes_factor:.2f}")
        
        if self.expected_loss_a is not None and self.expected_loss_b is not None:
            lines.append(f"期待損失 (Aを選択): {self.expected_loss_a:.4f}")
            lines.append(f"期待損失 (Bを選択): {self.expected_loss_b:.4f}")
        
        return "\n".join(lines)
