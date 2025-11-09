"""
A/Bテスト結果の可視化

ベイジアンA/Bテストと頻度論的A/Bテストの結果を可視化するための関数を提供します。
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Optional, Tuple
import matplotlib.patches as mpatches

from src.test_data import TestData
from src.results import BayesianResult, FrequentistResult
from src.bayesian import BayesianABTest
from src.frequentist import FrequentistABTest


# seabornのスタイル設定
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_bayesian_distributions(
    bayesian_test: BayesianABTest,
    result: BayesianResult,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    ベイジアンA/Bテストの事後分布を可視化

    Parameters
    ----------
    bayesian_test : BayesianABTest
        ベイジアンA/Bテストのインスタンス
    result : BayesianResult
        ベイジアン分析の結果
    figsize : Tuple[int, int], optional
        図のサイズ（デフォルト: (14, 10)）

    Returns
    -------
    plt.Figure
        matplotlibのfigureオブジェクト
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # x軸の範囲を設定
    x = np.linspace(0, 1, 1000)

    # 1. 事後分布のプロット
    ax1 = axes[0, 0]

    # グループAの事後分布
    posterior_a = stats.beta.pdf(
        x, result.alpha_post_a, result.beta_post_a
    )
    ax1.plot(x, posterior_a, label=f'Group A (CVR={result.mean_a:.3f})',
             linewidth=2, color='#1f77b4')
    ax1.fill_between(x, posterior_a, alpha=0.3, color='#1f77b4')

    # グループBの事後分布
    posterior_b = stats.beta.pdf(
        x, result.alpha_post_b, result.beta_post_b
    )
    ax1.plot(x, posterior_b, label=f'Group B (CVR={result.mean_b:.3f})',
             linewidth=2, color='#ff7f0e')
    ax1.fill_between(x, posterior_b, alpha=0.3, color='#ff7f0e')

    ax1.set_xlabel('Conversion Rate', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Posterior Distributions (Beta Distribution)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. 差の分布のプロット
    ax2 = axes[0, 1]

    # サンプリング
    samples_a, samples_b = bayesian_test.sample_posterior()
    diff_samples = samples_b - samples_a

    ax2.hist(diff_samples, bins=100, density=True, alpha=0.7,
             color='#2ca02c', edgecolor='black')

    # 確信区間をハイライト
    ax2.axvline(result.diff_ci_lower, color='red', linestyle='--',
                linewidth=2, label=f'{result.credible_level:.0%} Credible Interval')
    ax2.axvline(result.diff_ci_upper, color='red', linestyle='--', linewidth=2)
    ax2.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.5,
                label='No Difference')
    ax2.axvline(result.diff_mean, color='blue', linestyle='-', linewidth=2,
                label=f'Mean Diff = {result.diff_mean:.4f}')

    ax2.set_xlabel('Difference in CVR (B - A)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Distribution of Difference (B - A)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. 確率の可視化
    ax3 = axes[1, 0]

    probabilities = [result.prob_a_better, result.prob_b_better]
    labels = [f'A is Better\n({result.prob_a_better:.1%})',
              f'B is Better\n({result.prob_b_better:.1%})']
    colors = ['#1f77b4', '#ff7f0e']

    bars = ax3.bar(labels, probabilities, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Probability', fontsize=12)
    ax3.set_title('Probability of Being Better', fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 1.0])
    ax3.grid(True, alpha=0.3, axis='y')

    # バーに値を表示
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 4. 統計サマリー
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    Bayesian A/B Test Summary
    {'=' * 45}

    Group A:
      - Posterior: Beta({result.alpha_post_a:.1f}, {result.beta_post_a:.1f})
      - Mean CVR: {result.mean_a:.4f}

    Group B:
      - Posterior: Beta({result.alpha_post_b:.1f}, {result.beta_post_b:.1f})
      - Mean CVR: {result.mean_b:.4f}

    Difference (B - A):
      - Mean: {result.diff_mean:.4f}
      - {result.credible_level:.0%} Credible Interval:
        [{result.diff_ci_lower:.4f}, {result.diff_ci_upper:.4f}]

    Probability B is Better: {result.prob_b_better:.2%}
    """

    if result.bayes_factor is not None:
        summary_text += f"\n    Bayes Factor: {result.bayes_factor:.2f}"

    if result.expected_loss_a is not None and result.expected_loss_b is not None:
        summary_text += f"""

    Expected Loss:
      - If choosing A: {result.expected_loss_a:.4f}
      - If choosing B: {result.expected_loss_b:.4f}
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    return fig


def plot_frequentist_results(
    data: TestData,
    result: FrequentistResult,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    頻度論的A/Bテストの結果を可視化

    Parameters
    ----------
    data : TestData
        A/Bテストのデータ
    result : FrequentistResult
        頻度論的検定の結果
    figsize : Tuple[int, int], optional
        図のサイズ（デフォルト: (14, 8)）

    Returns
    -------
    plt.Figure
        matplotlibのfigureオブジェクト
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. コンバージョン率の比較
    ax1 = axes[0]

    groups = ['Group A', 'Group B']
    cvrs = [data.cvr_a, data.cvr_b]
    colors = ['#1f77b4', '#ff7f0e']

    bars = ax1.bar(groups, cvrs, color=colors, alpha=0.7, edgecolor='black')

    # エラーバー（信頼区間）を追加
    # Wilson score methodによる信頼区間
    def wilson_ci(x, n, confidence_level):
        p_hat = x / n
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
        return center - margin, center + margin

    ci_a = wilson_ci(data.conv_a, data.n_a, result.confidence_level)
    ci_b = wilson_ci(data.conv_b, data.n_b, result.confidence_level)

    errors = [
        [data.cvr_a - ci_a[0], ci_a[1] - data.cvr_a],
        [data.cvr_b - ci_b[0], ci_b[1] - data.cvr_b]
    ]

    ax1.errorbar(groups, cvrs, yerr=np.array(errors).T, fmt='none',
                 color='black', capsize=10, capthick=2, linewidth=2)

    ax1.set_ylabel('Conversion Rate', fontsize=12)
    ax1.set_title('Conversion Rate Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # バーに値を表示
    for bar, cvr in zip(bars, cvrs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{cvr:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. 統計サマリー
    ax2 = axes[1]
    ax2.axis('off')

    significance_text = "✅ Significant" if result.is_significant else "❌ Not Significant"

    summary_text = f"""
    Frequentist A/B Test Summary
    {'=' * 50}

    Test Method: {result.method.value}

    Group A:
      - Sample Size: {data.n_a}
      - Conversions: {data.conv_a}
      - CVR: {data.cvr_a:.4f}
      - {result.confidence_level:.0%} CI: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]

    Group B:
      - Sample Size: {data.n_b}
      - Conversions: {data.conv_b}
      - CVR: {data.cvr_b:.4f}
      - {result.confidence_level:.0%} CI: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]

    Statistical Test:
      - Test Statistic: {result.test_statistic:.4f}
      - P-value: {result.p_value:.6f}
      - Significance Level α: {1 - result.confidence_level:.2f}
      - Result: {significance_text}

    Difference (B - A):
      - Point Estimate: {data.cvr_diff:.4f}
      - {result.confidence_level:.0%} CI:
        [{result.ci_lower:.4f}, {result.ci_upper:.4f}]
    """

    ax2.text(0.1, 0.9, summary_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    return fig


def plot_comparison(
    data: TestData,
    bayesian_result: BayesianResult,
    frequentist_result: FrequentistResult,
    figsize: Tuple[int, int] = (16, 6)
) -> plt.Figure:
    """
    ベイジアンと頻度論的アプローチを比較

    Parameters
    ----------
    data : TestData
        A/Bテストのデータ
    bayesian_result : BayesianResult
        ベイジアン分析の結果
    frequentist_result : FrequentistResult
        頻度論的検定の結果
    figsize : Tuple[int, int], optional
        図のサイズ（デフォルト: (16, 6)）

    Returns
    -------
    plt.Figure
        matplotlibのfigureオブジェクト
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. 差の分布と信頼区間/確信区間の比較
    ax1 = axes[0]

    # ベイジアンの差の分布をサンプリング
    samples_a = np.random.beta(bayesian_result.alpha_post_a,
                               bayesian_result.beta_post_a, 100000)
    samples_b = np.random.beta(bayesian_result.alpha_post_b,
                               bayesian_result.beta_post_b, 100000)
    diff_samples = samples_b - samples_a

    ax1.hist(diff_samples, bins=100, density=True, alpha=0.5,
             color='purple', label='Bayesian (Posterior)', edgecolor='black')

    # ベイジアンの確信区間
    ax1.axvline(bayesian_result.diff_ci_lower, color='purple', linestyle='--',
                linewidth=2, label=f'Bayesian {bayesian_result.credible_level:.0%} CI')
    ax1.axvline(bayesian_result.diff_ci_upper, color='purple', linestyle='--', linewidth=2)

    # 頻度論的の信頼区間
    ax1.axvline(frequentist_result.ci_lower, color='green', linestyle=':',
                linewidth=3, label=f'Frequentist {frequentist_result.confidence_level:.0%} CI')
    ax1.axvline(frequentist_result.ci_upper, color='green', linestyle=':', linewidth=3)

    # 差がゼロのライン
    ax1.axvline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)

    ax1.set_xlabel('Difference in CVR (B - A)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Interval Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. 結論の比較
    ax2 = axes[1]

    # ベイジアンの結論
    bayesian_conclusion = "B is Better" if bayesian_result.prob_b_better > 0.95 else \
                         "A is Better" if bayesian_result.prob_a_better > 0.95 else \
                         "Inconclusive"

    # 頻度論的の結論
    if frequentist_result.is_significant:
        if data.cvr_b > data.cvr_a:
            freq_conclusion = "B is Better"
        else:
            freq_conclusion = "A is Better"
    else:
        freq_conclusion = "No Difference"

    conclusions = ['Bayesian', 'Frequentist']
    results = [bayesian_conclusion, freq_conclusion]

    # 色の設定
    color_map = {
        "B is Better": '#ff7f0e',
        "A is Better": '#1f77b4',
        "Inconclusive": '#gray',
        "No Difference": '#gray'
    }
    colors = [color_map[r] for r in results]

    y_pos = np.arange(len(conclusions))
    ax2.barh(y_pos, [1, 1], color=colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(conclusions, fontsize=11)
    ax2.set_xlim([0, 1])
    ax2.set_xticks([])
    ax2.set_title('Conclusions', fontsize=14, fontweight='bold')

    # 結論のテキストを表示
    for i, (conclusion, result) in enumerate(zip(conclusions, results)):
        ax2.text(0.5, i, result, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    # 3. 主要メトリクスの比較
    ax3 = axes[2]
    ax3.axis('off')

    agreement = (bayesian_conclusion == freq_conclusion) or \
                (bayesian_conclusion == "Inconclusive" and freq_conclusion == "No Difference")

    agreement_text = "✅ Agreement" if agreement else "⚠️ Disagreement"

    comparison_text = f"""
    Bayesian vs Frequentist Comparison
    {'=' * 48}

    Bayesian Approach:
      - P(B > A): {bayesian_result.prob_b_better:.2%}
      - {bayesian_result.credible_level:.0%} Credible Interval:
        [{bayesian_result.diff_ci_lower:.4f},
         {bayesian_result.diff_ci_upper:.4f}]
      - Conclusion: {bayesian_conclusion}

    Frequentist Approach:
      - P-value: {frequentist_result.p_value:.6f}
      - {frequentist_result.confidence_level:.0%} Confidence Interval:
        [{frequentist_result.ci_lower:.4f},
         {frequentist_result.ci_upper:.4f}]
      - Conclusion: {freq_conclusion}

    {'=' * 48}
    Agreement: {agreement_text}
    """

    ax3.text(0.1, 0.9, comparison_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.tight_layout()
    return fig


def create_distribution_table(
    data: TestData,
    bayesian_result: BayesianResult,
    frequentist_result: FrequentistResult
) -> str:
    """
    分布の統計情報をテーブル形式で表示

    Parameters
    ----------
    data : TestData
        A/Bテストのデータ
    bayesian_result : BayesianResult
        ベイジアン分析の結果
    frequentist_result : FrequentistResult
        頻度論的検定の結果

    Returns
    -------
    str
        テーブル形式の文字列
    """
    table = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        A/B Test Distribution Table                           ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  INPUT DATA                                                                   ║
║  ───────────────────────────────────────────────────────────────────────────  ║
║  Group A: {data.n_a:8d} samples, {data.conv_a:8d} conversions (CVR: {data.cvr_a:.4f})      ║
║  Group B: {data.n_b:8d} samples, {data.conv_b:8d} conversions (CVR: {data.cvr_b:.4f})      ║
║  Difference (B - A): {data.cvr_diff:+.4f}                                              ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  BAYESIAN APPROACH                                                            ║
║  ───────────────────────────────────────────────────────────────────────────  ║
║  Group A Posterior: Beta({bayesian_result.alpha_post_a:.1f}, {bayesian_result.beta_post_a:.1f})                                ║
║  Group B Posterior: Beta({bayesian_result.alpha_post_b:.1f}, {bayesian_result.beta_post_b:.1f})                                ║
║                                                                               ║
║  P(B > A): {bayesian_result.prob_b_better:.2%}                                                        ║
║  P(A > B): {bayesian_result.prob_a_better:.2%}                                                        ║
║                                                                               ║
║  Difference (B - A):                                                          ║
║    Mean: {bayesian_result.diff_mean:+.6f}                                                      ║
║    {bayesian_result.credible_level:.0%} Credible Interval: [{bayesian_result.diff_ci_lower:+.6f}, {bayesian_result.diff_ci_upper:+.6f}]          ║
"""

    if bayesian_result.bayes_factor is not None:
        table += f"║    Bayes Factor: {bayesian_result.bayes_factor:.4f}                                                  ║\n"

    if bayesian_result.expected_loss_a is not None:
        table += f"""║                                                                               ║
║  Expected Loss:                                                               ║
║    If choosing A: {bayesian_result.expected_loss_a:.6f}                                             ║
║    If choosing B: {bayesian_result.expected_loss_b:.6f}                                             ║
"""

    table += f"""║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  FREQUENTIST APPROACH                                                         ║
║  ───────────────────────────────────────────────────────────────────────────  ║
║  Test Method: {frequentist_result.method.value:<20s}                                        ║
║  Test Statistic: {frequentist_result.test_statistic:+.6f}                                              ║
║  P-value: {frequentist_result.p_value:.8f}                                                      ║
║  Significance Level (α): {1 - frequentist_result.confidence_level:.2f}                                             ║
║  Result: {'✅ Statistically Significant' if frequentist_result.is_significant else '❌ Not Significant':<30s}                                ║
║                                                                               ║
║  Difference (B - A):                                                          ║
║    Point Estimate: {data.cvr_diff:+.6f}                                                ║
║    {frequentist_result.confidence_level:.0%} Confidence Interval: [{frequentist_result.ci_lower:+.6f}, {frequentist_result.ci_upper:+.6f}]        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

    return table
