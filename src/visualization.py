import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
from typing import Optional, Tuple
import matplotlib.patches as mpatches
import os
import shutil

from src.test_data import TestData
from src.results import BayesianResult, FrequentistResult
from src.bayesian import BayesianABTest
from src.frequentist import FrequentistABTest


# 日本語フォント設定（Streamlit Cloud対応・改善版）
def setup_japanese_font():
    """日本語フォントを設定する（seabornベース）"""

    # 日本語フォントのパスを検索
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc',
        '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc',  # macOS
        'C:\\Windows\\Fonts\\msgothic.ttc',  # Windows
    ]

    font_file = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            font_file = font_path
            break

    # matplotlibのフォント設定
    if font_file:
        try:
            # フォントを登録
            if hasattr(fm.fontManager, 'addfont'):
                fm.fontManager.addfont(font_file)
            font_prop = fm.FontProperties(fname=font_file)
            font_name = font_prop.get_name()

            # seabornとmatplotlibの設定を統合
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [font_name, 'Noto Sans CJK JP', 'Noto Sans JP', 'DejaVu Sans']
        except Exception as e:
            # フォント登録に失敗した場合はフォールバック
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans JP', 'DejaVu Sans', 'Arial']
    else:
        # フォールバック設定（フォントファイルが見つからない場合）
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans JP', 'DejaVu Sans', 'Arial']

    # その他の設定
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # seabornのコンテキスト設定（フォントサイズなど）
    sns.set_context("notebook", font_scale=1.1)

# フォント設定を実行
setup_japanese_font()

# seabornのスタイル設定（日本語フォント対応）
sns.set_style("whitegrid", {
    'font.family': 'sans-serif',
    'font.sans-serif': plt.rcParams['font.sans-serif'],
})

# カラーパレット
COLORS = {
    'group_a': '#3498db',  # 青
    'group_b': '#e74c3c',  # 赤
    'positive': '#2ecc71',  # 緑
    'neutral': '#95a5a6',   # グレー
    'highlight': '#f39c12', # オレンジ
    'credible': '#9b59b6'   # 紫
}


def plot_bayesian_distributions(
    bayesian_test: BayesianABTest,
    result: BayesianResult,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    ベイジアンA/Bテストの事後分布を可視化（改善版）

    Parameters
    ----------
    bayesian_test : BayesianABTest
        ベイジアンA/Bテストのインスタンス
    result : BayesianResult
        ベイジアン分析の結果
    figsize : Tuple[int, int], optional
        図のサイズ（デフォルト: (16, 10)）

    Returns
    -------
    plt.Figure
        matplotlibのfigureオブジェクト
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.4)

    # 1. 事後分布のプロット（大きめに）
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    # x軸の範囲を設定（データに応じて調整）
    x_min = max(0, min(result.mean_a, result.mean_b) - 0.05)
    x_max = min(1, max(result.mean_a, result.mean_b) + 0.05)
    x = np.linspace(x_min, x_max, 1000)

    # グループAの事後分布
    posterior_a = stats.beta.pdf(x, result.alpha_post_a, result.beta_post_a)
    ax1.plot(x, posterior_a, label=f'グループA (CVR={result.mean_a:.3f})',
             linewidth=3, color=COLORS['group_a'], alpha=0.9)
    ax1.fill_between(x, posterior_a, alpha=0.2, color=COLORS['group_a'])

    # グループBの事後分布
    posterior_b = stats.beta.pdf(x, result.alpha_post_b, result.beta_post_b)
    ax1.plot(x, posterior_b, label=f'グループB (CVR={result.mean_b:.3f})',
             linewidth=3, color=COLORS['group_b'], alpha=0.9)
    ax1.fill_between(x, posterior_b, alpha=0.2, color=COLORS['group_b'])

    # 平均値に垂直線を追加
    ax1.axvline(result.mean_a, color=COLORS['group_a'], linestyle='--',
                linewidth=2, alpha=0.7, label=f'A平均値')
    ax1.axvline(result.mean_b, color=COLORS['group_b'], linestyle='--',
                linewidth=2, alpha=0.7, label=f'B平均値')

    ax1.set_xlabel('コンバージョン率', fontsize=13, fontweight='bold')
    ax1.set_ylabel('確率密度', fontsize=13, fontweight='bold')
    ax1.set_title('事後分布の比較 (ベータ分布)', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # 2. 差の分布のプロット（seabornベース）
    ax2 = fig.add_subplot(gs[2, :2])

    # サンプリング
    samples_a, samples_b = bayesian_test.sample_posterior()
    diff_samples = samples_b - samples_a

    # seabornのhistplotを使用（より美しく、日本語フォント対応が確実）
    sns.histplot(diff_samples, bins=80, stat='density', alpha=0.6,
                 color=COLORS['credible'], edgecolor='white', linewidth=0.5, ax=ax2)

    # 追加でKDEも表示
    sns.kdeplot(diff_samples, color=COLORS['credible'], linewidth=2.5, ax=ax2, alpha=0.8)

    # 確信区間をハイライト
    ax2.axvline(result.diff_ci_lower, color=COLORS['credible'], linestyle='--',
                linewidth=2.5, label=f'{result.credible_level:.0%} 確信区間')
    ax2.axvline(result.diff_ci_upper, color=COLORS['credible'], linestyle='--', linewidth=2.5)

    # 区間を塗りつぶし
    y_max = ax2.get_ylim()[1]
    ax2.fill_betweenx([0, y_max], result.diff_ci_lower, result.diff_ci_upper,
                      alpha=0.15, color=COLORS['credible'], label='確信区間範囲')

    # ゼロのラインと平均値
    ax2.axvline(0, color='black', linestyle='-', linewidth=2.5, alpha=0.7,
                label='差なし (0)')
    ax2.axvline(result.diff_mean, color=COLORS['highlight'], linestyle='-', linewidth=3,
                label=f'平均差 = {result.diff_mean:.4f}')

    ax2.set_xlabel('CVRの差 (B - A)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('確率密度', fontsize=13, fontweight='bold')
    ax2.set_title('差の分布 (B - A)', fontsize=14, fontweight='bold', pad=20)
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle=':')

    # 3. 確率の可視化（円グラフ）
    ax3 = fig.add_subplot(gs[0, 2])

    probabilities = [result.prob_a_better, result.prob_b_better]
    labels = [f'Aが優位\n{result.prob_a_better:.1%}',
              f'Bが優位\n{result.prob_b_better:.1%}']
    colors = [COLORS['group_a'], COLORS['group_b']]

    # 円グラフの描画
    wedges, texts, autotexts = ax3.pie(
        probabilities,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05),
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    # パーセント表示を白色に
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)

    ax3.set_title('どちらが優位か？', fontsize=13, fontweight='bold', pad=20)

    # 4. 統計サマリー（より視覚的に）
    ax4 = fig.add_subplot(gs[1:, 2])
    ax4.axis('off')

    # 判定結果
    if result.prob_b_better > 0.95:
        conclusion = "Bが優位"
        conclusion_color = COLORS['group_b']
    elif result.prob_a_better > 0.95:
        conclusion = "Aが優位"
        conclusion_color = COLORS['group_a']
    else:
        conclusion = "判定不能"
        conclusion_color = COLORS['neutral']

    summary_text = f"""
ベイジアンA/Bテスト 結果サマリー
{'─' * 35}

【事後分布】
  グループA: Beta({result.alpha_post_a:.1f}, {result.beta_post_a:.1f})
    → 平均CVR: {result.mean_a:.4f}

  グループB: Beta({result.alpha_post_b:.1f}, {result.beta_post_b:.1f})
    → 平均CVR: {result.mean_b:.4f}

【差の分析 (B - A)】
  平均差: {result.diff_mean:+.4f}
  {result.credible_level:.0%} 確信区間:
    [{result.diff_ci_lower:+.4f}, {result.diff_ci_upper:+.4f}]

【確率】
  P(B > A): {result.prob_b_better:.1%}
  P(A > B): {result.prob_a_better:.1%}
"""

    if result.bayes_factor is not None:
        summary_text += f"\n  ベイズファクター: {result.bayes_factor:.2f}"

    if result.expected_loss_a is not None and result.expected_loss_b is not None:
        summary_text += f"""

【期待損失】
  Aを選択: {result.expected_loss_a:.4f}
  Bを選択: {result.expected_loss_b:.4f}
"""

    summary_text += f"""
{'─' * 35}
判定: {conclusion}
"""

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9,
                     edgecolor=conclusion_color, linewidth=2))

    return fig


def plot_frequentist_results(
    data: TestData,
    result: FrequentistResult,
    figsize: Tuple[int, int] = (16, 6)
) -> plt.Figure:
    """
    頻度論的A/Bテストの結果を可視化（改善版）

    Parameters
    ----------
    data : TestData
        A/Bテストのデータ
    result : FrequentistResult
        頻度論的検定の結果
    figsize : Tuple[int, int], optional
        図のサイズ（デフォルト: (16, 6)）

    Returns
    -------
    plt.Figure
        matplotlibのfigureオブジェクト
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, wspace=0.3)

    # 1. コンバージョン率の比較（エラーバー付き）
    ax1 = fig.add_subplot(gs[0, 0])

    groups = ['グループA', 'グループB']
    cvrs = [data.cvr_a, data.cvr_b]
    colors = [COLORS['group_a'], COLORS['group_b']]

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

    # バーを描画
    bars = ax1.bar(groups, cvrs, color=colors, alpha=0.8, edgecolor='white', linewidth=2)

    # エラーバー（信頼区間）を追加
    ax1.errorbar(groups, cvrs, yerr=np.array(errors).T, fmt='none',
                 color='black', capsize=12, capthick=2.5, linewidth=2.5, alpha=0.7)

    ax1.set_ylabel('コンバージョン率', fontsize=13, fontweight='bold')
    ax1.set_title('CVR比較 (信頼区間付き)', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, axis='y', linestyle=':')

    # バーに値を表示
    for i, (bar, cvr, ci) in enumerate(zip(bars, cvrs, [ci_a, ci_b])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + errors[i][1] + 0.005,
                f'{cvr:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        # 信頼区間を表示
        ax1.text(bar.get_x() + bar.get_width()/2., ci[0] - 0.01,
                f'[{ci[0]:.3f},\n{ci[1]:.3f}]',
                ha='center', va='top', fontsize=9, color='gray')

    # 2. p値の可視化
    ax2 = fig.add_subplot(gs[0, 1])

    # p値を視覚的に表示
    alpha = 1 - result.confidence_level

    # p値と有意水準の比較
    y_values = [result.p_value, alpha]
    labels = [f'p値\n{result.p_value:.4f}', f'有意水準 α\n{alpha:.2f}']

    # 色を決定（有意なら緑、そうでなければグレー）
    bar_colors = [COLORS['positive'] if result.is_significant else COLORS['neutral'],
                  COLORS['highlight']]

    bars = ax2.bar(labels, y_values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)

    # 値をバーに表示
    for bar, val in zip(bars, y_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 判定ライン
    ax2.axhline(alpha, color='red', linestyle='--', linewidth=2, alpha=0.5, label='有意水準')

    ax2.set_ylabel('値', fontsize=13, fontweight='bold')
    ax2.set_title('統計的有意性の判定', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylim([0, max(y_values) * 1.3])
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':')

    # 判定結果を表示
    if result.is_significant:
        judgment = f"有意差あり\n(p < α)"
        judgment_color = COLORS['positive']
    else:
        judgment = f"有意差なし\n(p ≥ α)"
        judgment_color = COLORS['neutral']

    ax2.text(0.5, 0.95, judgment, transform=ax2.transAxes,
            ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=judgment_color, alpha=0.3,
                     edgecolor=judgment_color, linewidth=2))

    # 3. 統計サマリー
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    # 効果量を計算
    pooled_p = (data.conv_a + data.conv_b) / (data.n_a + data.n_b)
    effect_size = data.cvr_diff / np.sqrt(pooled_p * (1 - pooled_p) * (1/data.n_a + 1/data.n_b))

    significance_text = "有意" if result.is_significant else "非有意"
    sig_color = COLORS['positive'] if result.is_significant else COLORS['neutral']

    summary_text = f"""
頻度主義A/Bテスト 結果サマリー
{'─' * 38}

【検定方法】
  {result.method.value}

【データ】
  グループA:
    サンプル数: {data.n_a}
    コンバージョン: {data.conv_a}
    CVR: {data.cvr_a:.4f}
    {result.confidence_level:.0%} 信頼区間:
      [{ci_a[0]:.4f}, {ci_a[1]:.4f}]

  グループB:
    サンプル数: {data.n_b}
    コンバージョン: {data.conv_b}
    CVR: {data.cvr_b:.4f}
    {result.confidence_level:.0%} 信頼区間:
      [{ci_b[0]:.4f}, {ci_b[1]:.4f}]

【統計検定】
  検定統計量: {result.test_statistic:.4f}
  p値: {result.p_value:.6f}
  有意水準 α: {alpha:.2f}
  結果: {significance_text}

【差の分析 (B - A)】
  点推定: {data.cvr_diff:+.4f}
  {result.confidence_level:.0%} 信頼区間:
    [{result.ci_lower:+.4f}, {result.ci_upper:+.4f}]
  効果量: {effect_size:.3f}
{'─' * 38}
"""

    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9,
                     edgecolor=sig_color, linewidth=2))

    return fig


def plot_comparison(
    data: TestData,
    bayesian_result: BayesianResult,
    frequentist_result: FrequentistResult,
    figsize: Tuple[int, int] = (18, 10)
) -> plt.Figure:
    """
    ベイジアンと頻度論的アプローチを比較（改善版）

    Parameters
    ----------
    data : TestData
        A/Bテストのデータ
    bayesian_result : BayesianResult
        ベイジアン分析の結果
    frequentist_result : FrequentistResult
        頻度論的検定の結果
    figsize : Tuple[int, int], optional
        図のサイズ（デフォルト: (18, 10)）

    Returns
    -------
    plt.Figure
        matplotlibのfigureオブジェクト
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # 1. 差の分布と信頼区間/確信区間の比較（seabornベース）
    ax1 = fig.add_subplot(gs[0, :2])

    # ベイジアンの差の分布をサンプリング
    samples_a = np.random.beta(bayesian_result.alpha_post_a,
                               bayesian_result.beta_post_a, 100000)
    samples_b = np.random.beta(bayesian_result.alpha_post_b,
                               bayesian_result.beta_post_b, 100000)
    diff_samples = samples_b - samples_a

    # seabornのhistplotとkdeplotを使用
    sns.histplot(diff_samples, bins=100, stat='density', alpha=0.4,
                 color=COLORS['credible'], label='ベイジアン事後分布',
                 edgecolor='white', linewidth=0.5, ax=ax1)

    # KDEプロットを追加（より滑らかな分布表示）
    sns.kdeplot(diff_samples, color=COLORS['credible'], linewidth=2.5, ax=ax1, alpha=0.7)

    # ベイジアンの確信区間
    y_max = ax1.get_ylim()[1]
    ax1.fill_betweenx([0, y_max], bayesian_result.diff_ci_lower, bayesian_result.diff_ci_upper,
                      alpha=0.2, color=COLORS['credible'], label=f'ベイジアン {bayesian_result.credible_level:.0%} 確信区間')

    ax1.axvline(bayesian_result.diff_ci_lower, color=COLORS['credible'], linestyle='--',
                linewidth=2.5, alpha=0.8)
    ax1.axvline(bayesian_result.diff_ci_upper, color=COLORS['credible'], linestyle='--',
                linewidth=2.5, alpha=0.8)

    # 頻度論的の信頼区間
    ax1.axvline(frequentist_result.ci_lower, color=COLORS['highlight'], linestyle=':',
                linewidth=3.5, label=f'頻度主義 {frequentist_result.confidence_level:.0%} 信頼区間',
                alpha=0.9)
    ax1.axvline(frequentist_result.ci_upper, color=COLORS['highlight'], linestyle=':',
                linewidth=3.5, alpha=0.9)

    # 差がゼロのライン
    ax1.axvline(0, color='black', linestyle='-', linewidth=2.5, alpha=0.7, label='差なし (0)')

    # 平均値
    ax1.axvline(bayesian_result.diff_mean, color=COLORS['credible'], linestyle='-',
                linewidth=2, alpha=0.5, label=f'ベイジアン平均 ({bayesian_result.diff_mean:.4f})')

    ax1.set_xlabel('CVRの差 (B - A)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('確率密度', fontsize=13, fontweight='bold')
    ax1.set_title('区間の比較: ベイジアン vs 頻度主義', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=9, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=':')

    # 2. 結論の比較（改善版）
    ax2 = fig.add_subplot(gs[0, 2])

    # ベイジアンの結論
    if bayesian_result.prob_b_better > 0.95:
        bayesian_conclusion = "Bが優位"
        bayesian_color = COLORS['group_b']
    elif bayesian_result.prob_a_better > 0.95:
        bayesian_conclusion = "Aが優位"
        bayesian_color = COLORS['group_a']
    else:
        bayesian_conclusion = "判定不能"
        bayesian_color = COLORS['neutral']

    # 頻度論的の結論
    if frequentist_result.is_significant:
        if data.cvr_b > data.cvr_a:
            freq_conclusion = "Bが優位"
            freq_color = COLORS['group_b']
        else:
            freq_conclusion = "Aが優位"
            freq_color = COLORS['group_a']
    else:
        freq_conclusion = "有意差なし"
        freq_color = COLORS['neutral']

    conclusions = ['ベイジアン', '頻度主義']
    results = [bayesian_conclusion, freq_conclusion]
    colors = [bayesian_color, freq_color]

    y_pos = np.arange(len(conclusions))
    bars = ax2.barh(y_pos, [1, 1], color=colors, alpha=0.7, edgecolor='white', linewidth=2)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(conclusions, fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.set_xticks([])
    ax2.set_title('結論の比較', fontsize=13, fontweight='bold', pad=20)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # 結論のテキストを表示
    for i, result in enumerate(results):
        ax2.text(0.5, i, f'{result}', ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')

    # 3. 主要メトリクスの比較（改善版）
    ax3 = fig.add_subplot(gs[1, :2])

    # メトリクスの比較をテーブル形式で表示
    metrics = ['確率/P値', '区間下限', '区間上限', '判定']

    bayesian_values = [
        f'P(B>A) = {bayesian_result.prob_b_better:.1%}',
        f'{bayesian_result.diff_ci_lower:.4f}',
        f'{bayesian_result.diff_ci_upper:.4f}',
        bayesian_conclusion
    ]

    freq_values = [
        f'p = {frequentist_result.p_value:.4f}',
        f'{frequentist_result.ci_lower:.4f}',
        f'{frequentist_result.ci_upper:.4f}',
        freq_conclusion
    ]

    # テーブルの作成
    x_pos = np.arange(len(metrics))
    width = 0.35

    bars1 = ax3.bar(x_pos - width/2, [1]*len(metrics), width, label='ベイジアン',
                    color=COLORS['credible'], alpha=0.7, edgecolor='white', linewidth=2)
    bars2 = ax3.bar(x_pos + width/2, [1]*len(metrics), width, label='頻度主義',
                    color=COLORS['highlight'], alpha=0.7, edgecolor='white', linewidth=2)

    # 値をバーに表示
    for i, (bar1, bar2, bval, fval) in enumerate(zip(bars1, bars2, bayesian_values, freq_values)):
        ax3.text(bar1.get_x() + bar1.get_width()/2, 0.5, bval,
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white', rotation=0)
        ax3.text(bar2.get_x() + bar2.get_width()/2, 0.5, fval,
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white', rotation=0)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax3.set_ylim([0, 1.2])
    ax3.set_yticks([])
    ax3.set_title('メトリクスの比較', fontsize=13, fontweight='bold', pad=20)
    ax3.legend(fontsize=10, loc='upper right')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # 4. 一致度と総合サマリー
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')

    # 結論の一致度
    agreement = (bayesian_conclusion == freq_conclusion) or \
                (bayesian_conclusion == "判定不能" and freq_conclusion == "有意差なし")

    if agreement:
        agreement_text = "一致"
        agreement_color = COLORS['positive']
    else:
        agreement_text = "不一致"
        agreement_color = COLORS['highlight']

    summary_text = f"""
比較サマリー
{'─' * 28}

【ベイジアン】
  確率: P(B>A) = {bayesian_result.prob_b_better:.1%}
  {bayesian_result.credible_level:.0%} 確信区間:
    [{bayesian_result.diff_ci_lower:+.4f},
     {bayesian_result.diff_ci_upper:+.4f}]
  結論: {bayesian_conclusion}

【頻度主義】
  p値: {frequentist_result.p_value:.4f}
  {frequentist_result.confidence_level:.0%} 信頼区間:
    [{frequentist_result.ci_lower:+.4f},
     {frequentist_result.ci_upper:+.4f}]
  結論: {freq_conclusion}

{'─' * 28}
結論の一致度: {agreement_text}

【解釈のポイント】
• ベイジアン: 確率的解釈
• 頻度主義: 仮説検定
• 両方の結果を総合判断
  することが重要です
"""

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9,
                     edgecolor=agreement_color, linewidth=2))

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