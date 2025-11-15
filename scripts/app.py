import sys
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from src.test_data import TestData, TestMethod
from src.bayesian import BayesianABTest
from src.frequentist import FrequentistABTest
from src.visualization import (
    plot_bayesian_distributions,
    plot_frequentist_results,
    plot_comparison
)


# ページ設定
st.set_page_config(
    page_title="A/B Test Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# タイトルとイントロダクション
st.title("📊 A/Bテスト分析ツール")

# イントロダクション
st.markdown("""
<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h3 style="color: #1f77b4; margin-top: 0;">👋 ようこそ！</h3>
    <p style="font-size: 16px; line-height: 1.6;">
        このツールでは、<b>ベイジアンA/Bテスト</b>と<b>頻度主義A/Bテスト</b>の両方を体験できます。<br>
        左のサイドバーでデータを入力するか、プリセットを選択して、すぐに分析を開始できます。
    </p>
    <details>
        <summary style="cursor: pointer; color: #1f77b4; font-weight: bold;">📖 A/Bテストとは？</summary>
        <p style="margin-top: 10px; line-height: 1.6;">
            A/Bテストは、2つのバージョン（AとB）を比較して、どちらがより優れているかを判定する統計的手法です。<br>
            例えば、Webサイトの2つのデザインのうち、どちらがより多くのコンバージョンを生み出すかを判定できます。
        </p>
    </details>
    <details style="margin-top: 10px;">
        <summary style="cursor: pointer; color: #1f77b4; font-weight: bold;">🎯 このツールの使い方</summary>
        <ol style="margin-top: 10px; line-height: 1.6;">
            <li><b>プリセットを選択</b>: 左のサイドバーでサンプルデータを選択</li>
            <li><b>データを入力</b>: または、自分のデータを入力</li>
            <li><b>結果を確認</b>: 3つのタブで異なる分析結果を確認</li>
            <li><b>比較</b>: ベイジアンと頻度主義の違いを理解</li>
        </ol>
    </details>
</div>
""", unsafe_allow_html=True)

# サイドバー: データ入力
st.sidebar.header("🔧 データ設定")

# 統計知識レベルの選択
st.sidebar.subheader("👤 あなたの統計知識レベル")
expertise_level = st.sidebar.radio(
    "表示する情報量を選択",
    ["初心者 (シンプル)", "中級者 (標準)", "上級者 (詳細)"],
    help="統計の知識レベルに応じて、表示する情報量を調整します"
)

if expertise_level == "初心者 (シンプル)":
    st.sidebar.markdown("""
    <div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; font-size: 14px;">
            💡 <b>初めての方へ</b><br>
            まずは「明確な差がある例」を選択して、どんな分析ができるか試してみましょう！
        </p>
    </div>
    """, unsafe_allow_html=True)
elif expertise_level == "中級者 (標準)":
    st.sidebar.markdown("""
    <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; font-size: 14px;">
            📊 <b>標準モード</b><br>
            主要な統計指標と解釈を表示します。詳細設定で分析方法をカスタマイズできます。
        </p>
    </div>
    """, unsafe_allow_html=True)
else:  # 上級者
    st.sidebar.markdown("""
    <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
        <p style="margin: 0; font-size: 14px;">
            🎓 <b>上級者モード</b><br>
            すべての統計量と詳細設定を表示します。事前分布のカスタマイズも可能です。
        </p>
    </div>
    """, unsafe_allow_html=True)

# サンプルデータのプリセット
st.sidebar.subheader("📋 プリセット")
preset = st.sidebar.selectbox(
    "サンプルデータを選択",
    [
        "カスタム",
        "明確な差がある例",
        "微妙な差がある例",
        "差がない例",
        "小サンプルの例"
    ],
    help="様々なシナリオのサンプルデータを選択できます。初めての方は「明確な差がある例」がおすすめです。"
)

# プリセットの値を設定
if preset == "明確な差がある例":
    default_n_a = 1000
    default_conv_a = 100
    default_n_b = 1000
    default_conv_b = 150
elif preset == "微妙な差がある例":
    default_n_a = 1000
    default_conv_a = 100
    default_n_b = 1000
    default_conv_b = 115
elif preset == "差がない例":
    default_n_a = 1000
    default_conv_a = 100
    default_n_b = 1000
    default_conv_b = 105
elif preset == "小サンプルの例":
    default_n_a = 50
    default_conv_a = 10
    default_n_b = 50
    default_conv_b = 15
else:  # カスタム
    default_n_a = 1000
    default_conv_a = 100
    default_n_b = 1000
    default_conv_b = 120

# データ入力
st.sidebar.subheader("🅰️ グループA (現行版)")
n_a = st.sidebar.number_input(
    "サンプルサイズ (グループA)",
    min_value=1,
    value=default_n_a,
    step=1,
    help="グループAの訪問者数（例：Webサイトの訪問者数、広告の表示回数など）"
)
conv_a = st.sidebar.number_input(
    "コンバージョン数 (グループA)",
    min_value=0,
    max_value=int(n_a),
    value=min(default_conv_a, int(n_a)),
    step=1,
    help="グループAのコンバージョン数（例：購入数、クリック数など）"
)

st.sidebar.subheader("🅱️ グループB (新バージョン)")
n_b = st.sidebar.number_input(
    "サンプルサイズ (グループB)",
    min_value=1,
    value=default_n_b,
    step=1,
    help="グループBの訪問者数（例：Webサイトの訪問者数、広告の表示回数など）"
)
conv_b = st.sidebar.number_input(
    "コンバージョン数 (グループB)",
    min_value=0,
    max_value=int(n_b),
    value=min(default_conv_b, int(n_b)),
    step=1,
    help="グループBのコンバージョン数（例：購入数、クリック数など）"
)

# 詳細設定（知識レベルに応じて表示）
if expertise_level in ["中級者 (標準)", "上級者 (詳細)"]:
    with st.sidebar.expander("⚙️ 詳細設定", expanded=(expertise_level == "上級者 (詳細)")):
        # ベイジアン設定
        st.markdown("**🎲 ベイジアン設定**")

        if expertise_level == "上級者 (詳細)":
            # 上級者向け：事前分布の入力方法を選択
            prior_input_method = st.radio(
                "事前分布の入力方法",
                ["仮想サンプル数で設定", "αとβで直接設定"],
                help="Beta分布の事前分布を設定する方法を選択します"
            )

            if prior_input_method == "仮想サンプル数で設定":
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 8px; border-radius: 5px; margin-bottom: 10px; font-size: 12px;">
                    💡 「仮想的な過去データ」として事前分布を設定できます。<br>
                    例：過去に100人中10人が成功した場合 → サンプル数:100, 成功数:10
                </div>
                """, unsafe_allow_html=True)

                prior_sample_size = st.number_input(
                    "事前分布：仮想サンプル数",
                    min_value=0,
                    value=2,
                    step=1,
                    help="仮想的な過去データのサンプル数（0の場合は無情報事前分布）"
                )

                if prior_sample_size > 0:
                    prior_successes = st.number_input(
                        "事前分布：仮想成功数",
                        min_value=0,
                        max_value=int(prior_sample_size),
                        value=min(1, int(prior_sample_size)),
                        step=1,
                        help="仮想的な過去データの成功数"
                    )
                    # 成功数 + 1 → α、失敗数 + 1 → β
                    alpha_prior = float(prior_successes + 1)
                    beta_prior = float(prior_sample_size - prior_successes + 1)
                else:
                    # 無情報事前分布: Beta(1, 1) = 一様分布
                    alpha_prior = 1.0
                    beta_prior = 1.0

                st.caption(f"→ Beta分布のパラメータ: α={alpha_prior:.1f}, β={beta_prior:.1f}")

            else:  # αとβで直接設定
                alpha_prior = st.number_input(
                    "事前分布 α",
                    min_value=0.1,
                    value=1.0,
                    step=0.1,
                    help="Beta分布の事前分布パラメータα"
                )
                beta_prior = st.number_input(
                    "事前分布 β",
                    min_value=0.1,
                    value=1.0,
                    step=0.1,
                    help="Beta分布の事前分布パラメータβ"
                )
        else:
            # 中級者向け：デフォルト値を使用（無情報事前分布）
            alpha_prior = 1.0
            beta_prior = 1.0
            st.info("📌 事前分布: Beta(1, 1) = 無情報事前分布を使用")

        credible_level = st.slider(
            "確信水準",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="ベイジアンの確信区間の水準"
        )

        # 頻度主義設定
        st.markdown("**📊 頻度主義設定**")
        confidence_level = st.slider(
            "信頼水準",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01,
            help="信頼区間の水準"
        )
        test_method = st.selectbox(
            "検定方法",
            [TestMethod.Z_TEST, TestMethod.T_TEST, TestMethod.CHI_SQUARE],
            format_func=lambda x: {
                TestMethod.Z_TEST: "Z検定（正規近似）",
                TestMethod.T_TEST: "t検定（Welch法）",
                TestMethod.CHI_SQUARE: "カイ二乗検定"
            }[x],
            help="統計的仮説検定の方法"
        )
else:
    # 初心者向け：デフォルト値を使用
    alpha_prior = 1.0
    beta_prior = 1.0
    credible_level = 0.95
    confidence_level = 0.95
    test_method = TestMethod.Z_TEST

# データの妥当性チェックと分析の実行
try:
    # TestDataオブジェクトを作成
    data = TestData(n_a=int(n_a), conv_a=int(conv_a), n_b=int(n_b), conv_b=int(conv_b))

    # 基本統計の表示
    st.header("📈 基本統計")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="グループA CVR",
            value=f"{data.cvr_a:.2%}",
            delta=None
        )

    with col2:
        st.metric(
            label="グループB CVR",
            value=f"{data.cvr_b:.2%}",
            delta=f"{data.cvr_diff:+.2%}"
        )

    with col3:
        st.metric(
            label="相対的な改善率",
            value=f"{(data.cvr_diff / data.cvr_a * 100):+.1f}%" if data.cvr_a > 0 else "N/A",
            delta=None
        )

    # タブの作成
    tab1, tab2, tab3 = st.tabs(["🎲 ベイジアンアプローチ", "📊 頻度主義アプローチ", "⚖️ 比較"])

    # ベイジアンアプローチ
    with tab1:
        st.header("ベイジアンA/Bテスト")

        # 知識レベル別の説明
        if expertise_level == "初心者 (シンプル)":
            st.markdown("""
            <div style="background-color: #e8f4f8; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="margin-top: 0;">🎲 ベイジアンアプローチとは？</h4>
                <p>「<b>BがAより良い確率</b>」を直接計算できる方法です。</p>
                <p>例：「Bの方が良い確率が95%」というように、直感的に理解しやすい結果が得られます。</p>
            </div>
            """, unsafe_allow_html=True)
        elif expertise_level == "中級者 (標準)":
            st.markdown("""
            ベイジアンアプローチは、確率的な推論を行います。
            「BがAより優れている確率」を直接計算できるのが特徴です。
            """)
        else:  # 上級者
            st.markdown("""
            ベイジアンアプローチは、事後分布を用いた確率的推論を行います。
            事前分布とデータを組み合わせて事後分布を計算し、確信区間や期待損失を推定します。
            """)

        with st.spinner("ベイジアン分析を実行中..."):
            bayesian_test = BayesianABTest(
                data=data,
                alpha_prior=alpha_prior,
                beta_prior=beta_prior,
                credible_level=credible_level,
                n_samples=100000
            )
            bayesian_result = bayesian_test.run()

        # 結果のサマリー（知識レベル別表示）
        st.subheader("📋 結果サマリー")

        if expertise_level == "初心者 (シンプル)":
            # 初心者向け：最も重要な情報のみ
            st.metric(
                label="🎯 BがAより良い確率",
                value=f"{bayesian_result.prob_b_better:.1%}",
                delta=None
            )

            # わかりやすい判定
            if bayesian_result.prob_b_better > 0.95:
                st.success("✅ 結論: **Bを選ぶべき**です（95%以上の確率で優れています）")
            elif bayesian_result.prob_b_better < 0.05:
                st.success("✅ 結論: **Aを選ぶべき**です（95%以上の確率で優れています）")
            elif bayesian_result.prob_b_better > 0.80:
                st.info("📊 結論: **Bの方が良さそう**ですが、確信度は80%程度です")
            elif bayesian_result.prob_b_better < 0.20:
                st.info("📊 結論: **Aの方が良さそう**ですが、確信度は80%程度です")
            else:
                st.warning("⚠️ 結論: **どちらが良いか判断が難しい**です。もっとデータが必要かもしれません")

            # 簡易的な説明
            with st.expander("💡 この結果の意味は？"):
                st.markdown(f"""
                - **BがAより良い確率**: {bayesian_result.prob_b_better:.1%}
                  - これは100回テストを繰り返したら、約{bayesian_result.prob_b_better*100:.0f}回はBの方が良い結果になるという意味です
                - **一般的な判断基準**:
                  - 95%以上 → 自信を持って判断できる
                  - 80〜95% → ある程度自信を持って判断できる
                  - 50〜80% → 判断が難しい、もっとデータが必要
                """)

        elif expertise_level == "中級者 (標準)":
            # 中級者向け：主要な統計量
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="BがAより優れている確率",
                    value=f"{bayesian_result.prob_b_better:.1%}",
                    delta=None
                )

            with col2:
                st.metric(
                    label="差の期待値 (B - A)",
                    value=f"{bayesian_result.diff_mean:+.4f}",
                    delta=None
                )

            with col3:
                if bayesian_result.bayes_factor is not None:
                    st.metric(
                        label="ベイズファクター",
                        value=f"{bayesian_result.bayes_factor:.2f}",
                        delta=None
                    )

            # 確信区間
            st.info(
                f"**{bayesian_result.credible_level:.0%} 確信区間**: "
                f"[{bayesian_result.diff_ci_lower:.4f}, {bayesian_result.diff_ci_upper:.4f}]"
            )

        else:  # 上級者
            # 上級者向け：すべての統計量
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="P(B > A)",
                    value=f"{bayesian_result.prob_b_better:.4f}",
                    delta=None
                )

            with col2:
                st.metric(
                    label="E[B - A]",
                    value=f"{bayesian_result.diff_mean:+.6f}",
                    delta=None
                )

            with col3:
                st.metric(
                    label="Std[B - A]",
                    value=f"{bayesian_result.diff_std:.6f}",
                    delta=None
                )

            with col4:
                if bayesian_result.bayes_factor is not None:
                    st.metric(
                        label="Bayes Factor",
                        value=f"{bayesian_result.bayes_factor:.4f}",
                        delta=None
                    )

            # 確信区間
            st.info(
                f"**{bayesian_result.credible_level:.0%} 確信区間 (HDI)**: "
                f"[{bayesian_result.diff_ci_lower:.6f}, {bayesian_result.diff_ci_upper:.6f}]"
            )

            # 事後分布のパラメータ
            with st.expander("📊 事後分布のパラメータ"):
                post_col1, post_col2 = st.columns(2)
                with post_col1:
                    st.markdown(f"""
                    **グループA**
                    - α (posterior) = {alpha_prior + conv_a:.1f}
                    - β (posterior) = {beta_prior + (n_a - conv_a):.1f}
                    """)
                with post_col2:
                    st.markdown(f"""
                    **グループB**
                    - α (posterior) = {alpha_prior + conv_b:.1f}
                    - β (posterior) = {beta_prior + (n_b - conv_b):.1f}
                    """)

        # 期待損失（中級者以上のみ表示）
        if expertise_level != "初心者 (シンプル)" and bayesian_result.expected_loss_a is not None and bayesian_result.expected_loss_b is not None:
            st.subheader("💰 期待損失")

            if expertise_level == "中級者 (標準)":
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Aを選択した場合の期待損失",
                        value=f"{bayesian_result.expected_loss_a:.6f}",
                        delta=None
                    )

                with col2:
                    st.metric(
                        label="Bを選択した場合の期待損失",
                        value=f"{bayesian_result.expected_loss_b:.6f}",
                        delta=None
                    )

                if bayesian_result.expected_loss_a < bayesian_result.expected_loss_b:
                    st.success("✅ 推奨: **グループA**を選択することをお勧めします")
                else:
                    st.success("✅ 推奨: **グループB**を選択することをお勧めします")
            else:  # 上級者
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        label="Expected Loss(A)",
                        value=f"{bayesian_result.expected_loss_a:.8f}",
                        delta=None,
                        help="Aを選択した場合の期待損失（Bを選んだ方が良かった場合の期待される機会損失）"
                    )

                with col2:
                    st.metric(
                        label="Expected Loss(B)",
                        value=f"{bayesian_result.expected_loss_b:.8f}",
                        delta=None,
                        help="Bを選択した場合の期待損失（Aを選んだ方が良かった場合の期待される機会損失）"
                    )

                # 損失比率の計算
                if bayesian_result.expected_loss_a > 0 and bayesian_result.expected_loss_b > 0:
                    loss_ratio = bayesian_result.expected_loss_b / bayesian_result.expected_loss_a if bayesian_result.expected_loss_a < bayesian_result.expected_loss_b else bayesian_result.expected_loss_a / bayesian_result.expected_loss_b
                    st.caption(f"損失比率: {loss_ratio:.2f}x")

                if bayesian_result.expected_loss_a < bayesian_result.expected_loss_b:
                    st.success("✅ 推奨: **グループA**を選択（期待損失が小さい）")
                else:
                    st.success("✅ 推奨: **グループB**を選択（期待損失が小さい）")

        # 可視化
        st.subheader("📊 可視化")
        fig = plot_bayesian_distributions(bayesian_test, bayesian_result)
        st.pyplot(fig)
        plt.close(fig)

    # 頻度主義アプローチ
    with tab2:
        st.header("頻度主義A/Bテスト")

        # 知識レベル別の説明
        if expertise_level == "初心者 (シンプル)":
            st.markdown("""
            <div style="background-color: #fff3cd; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="margin-top: 0;">📊 頻度主義アプローチとは？</h4>
                <p>「<b>AとBに本当に差があるのか</b>」を統計的に検証する方法です。</p>
                <p>p値が0.05より小さければ「差がある」と判定します（一般的な基準）。</p>
            </div>
            """, unsafe_allow_html=True)
        elif expertise_level == "中級者 (標準)":
            st.markdown("""
            頻度主義アプローチは、仮説検定を用いて統計的有意性を判定します。
            帰無仮説「AとBに差がない」を棄却できるかを検証します。
            """)
        else:  # 上級者
            st.markdown("""
            頻度主義アプローチは、帰無仮説統計検定 (NHST) を用いて統計的有意性を判定します。
            H₀: p_A = p_B（差がない）vs H₁: p_A ≠ p_B（差がある）を検証し、p値とα水準で判定します。
            """)

        with st.spinner("頻度主義分析を実行中..."):
            frequentist_test = FrequentistABTest(data=data, confidence_level=confidence_level)
            frequentist_result = frequentist_test.run(method=test_method)

        # 結果のサマリー（知識レベル別表示）
        st.subheader("📋 結果サマリー")

        if expertise_level == "初心者 (シンプル)":
            # 初心者向け：判定結果を前面に
            if frequentist_result.is_significant:
                if data.cvr_b > data.cvr_a:
                    st.success("✅ 結論: **Bの方が優れています**（統計的に有意な差があります）")
                else:
                    st.success("✅ 結論: **Aの方が優れています**（統計的に有意な差があります）")
            else:
                st.warning("⚠️ 結論: **明確な差は見られません**（統計的に有意な差がありません）")

            # p値を簡単に表示
            st.metric(
                label="p値（小さいほど差がある）",
                value=f"{frequentist_result.p_value:.4f}",
                delta=None
            )

            # わかりやすい説明
            with st.expander("💡 p値とは？"):
                st.markdown(f"""
                **p値 = {frequentist_result.p_value:.4f}**

                p値は「AとBに差がない」と仮定した場合に、今回の結果が偶然起こる確率です。

                - **p値 < 0.05**: 差がある可能性が高い（偶然では説明しづらい）
                - **p値 ≥ 0.05**: 差があるとは言えない（偶然かもしれない）

                今回の結果: {'p < 0.05 なので、差があると判定できます' if frequentist_result.is_significant else 'p ≥ 0.05 なので、差があるとは言えません'}
                """)

        elif expertise_level == "中級者 (標準)":
            # 中級者向け：主要な統計量
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="p値",
                    value=f"{frequentist_result.p_value:.6f}",
                    delta=None
                )

            with col2:
                st.metric(
                    label="検定統計量",
                    value=f"{frequentist_result.test_statistic:.4f}",
                    delta=None
                )

            with col3:
                significance = "有意" if frequentist_result.is_significant else "非有意"
                st.metric(
                    label="統計的有意性",
                    value=significance,
                    delta=None
                )

            # 信頼区間
            st.info(
                f"**{frequentist_result.confidence_level:.0%} 信頼区間**: "
                f"[{frequentist_result.ci_lower:.4f}, {frequentist_result.ci_upper:.4f}]"
            )

            # 判定結果
            if frequentist_result.is_significant:
                if data.cvr_b > data.cvr_a:
                    st.success("✅ 判定: グループBはグループAよりも**統計的に有意に優れています**")
                else:
                    st.success("✅ 判定: グループAはグループBよりも**統計的に有意に優れています**")
            else:
                st.warning("⚠️ 判定: グループAとグループBの間に**統計的に有意な差は見られません**")

        else:  # 上級者
            # 上級者向け：詳細な統計量
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="p-value",
                    value=f"{frequentist_result.p_value:.8f}",
                    delta=None,
                    help="帰無仮説下での検定統計量の確率"
                )

            with col2:
                st.metric(
                    label="Test Statistic",
                    value=f"{frequentist_result.test_statistic:.6f}",
                    delta=None,
                    help=f"{test_method.value}の検定統計量"
                )

            with col3:
                significance = "Significant" if frequentist_result.is_significant else "Not Significant"
                st.metric(
                    label=f"Significance (α={1-confidence_level:.2f})",
                    value=significance,
                    delta=None
                )

            with col4:
                # 効果量（Cohen's h）を計算
                import numpy as np
                p1_transformed = 2 * np.arcsin(np.sqrt(data.cvr_a))
                p2_transformed = 2 * np.arcsin(np.sqrt(data.cvr_b))
                cohens_h = abs(p2_transformed - p1_transformed)
                st.metric(
                    label="Cohen's h",
                    value=f"{cohens_h:.4f}",
                    delta=None,
                    help="効果量（0.2=小, 0.5=中, 0.8=大）"
                )

            # 信頼区間
            st.info(
                f"**{frequentist_result.confidence_level:.0%} CI (差分)**: "
                f"[{frequentist_result.ci_lower:.6f}, {frequentist_result.ci_upper:.6f}]"
            )

            # 追加情報
            with st.expander("📊 詳細な検定情報"):
                st.markdown(f"""
                **検定方法**: {test_method.value}
                - 帰無仮説 (H₀): p_A = p_B
                - 対立仮説 (H₁): p_A ≠ p_B
                - 有意水準 (α): {1 - confidence_level:.3f}
                - 検定統計量: {frequentist_result.test_statistic:.6f}
                - p値: {frequentist_result.p_value:.8f}
                - 判定: {'H₀を棄却（有意差あり）' if frequentist_result.is_significant else 'H₀を棄却できない（有意差なし）'}
                """)

            # 判定結果
            if frequentist_result.is_significant:
                if data.cvr_b > data.cvr_a:
                    st.success("✅ 判定: グループB > グループA（統計的に有意, p < α）")
                else:
                    st.success("✅ 判定: グループA > グループB（統計的に有意, p < α）")
            else:
                st.warning("⚠️ 判定: 有意差なし（p ≥ α, H₀を棄却できない）")

        # 可視化
        st.subheader("📊 可視化")
        fig = plot_frequentist_results(data, frequentist_result)
        st.pyplot(fig)
        plt.close(fig)

    # 比較タブ
    with tab3:
        st.header("ベイジアン vs 頻度主義")

        # 知識レベル別の説明
        if expertise_level == "初心者 (シンプル)":
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h4 style="margin-top: 0;">⚖️ 2つの方法を比較してみよう</h4>
                <p>同じデータでも、分析方法によって結果の見え方が変わります。</p>
                <p>どちらも正しい方法ですが、使い分けることが大切です。</p>
            </div>
            """, unsafe_allow_html=True)
        elif expertise_level == "中級者 (標準)":
            st.markdown("""
            両方のアプローチを比較して、それぞれの特徴と結論を確認します。
            """)
        else:  # 上級者
            st.markdown("""
            ベイジアンアプローチと頻度主義アプローチの比較分析を行います。
            異なる推論パラダイムによる結果の違いと、実務での使い分けを理解します。
            """)

        # 分析が実行されていることを確認
        if 'bayesian_result' not in locals():
            with st.spinner("ベイジアン分析を実行中..."):
                bayesian_test = BayesianABTest(
                    data=data,
                    alpha_prior=alpha_prior,
                    beta_prior=beta_prior,
                    credible_level=credible_level,
                    n_samples=100000
                )
                bayesian_result = bayesian_test.run()

        if 'frequentist_result' not in locals():
            with st.spinner("頻度主義分析を実行中..."):
                frequentist_test = FrequentistABTest(data=data, confidence_level=confidence_level)
                frequentist_result = frequentist_test.run(method=test_method)

        # 比較サマリー（知識レベル別表示）
        st.subheader("📋 比較サマリー")

        if expertise_level == "初心者 (シンプル)":
            # 初心者向け：シンプルな結論の比較
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🎲 ベイジアン")
                st.metric(
                    label="BがAより良い確率",
                    value=f"{bayesian_result.prob_b_better:.1%}"
                )

                if bayesian_result.prob_b_better > 0.95:
                    st.success("✅ **Bを選ぶべき**")
                elif bayesian_result.prob_b_better < 0.05:
                    st.success("✅ **Aを選ぶべき**")
                else:
                    st.warning("⚠️ **判断が難しい**")

            with col2:
                st.markdown("### 📊 頻度主義")
                st.metric(
                    label="p値",
                    value=f"{frequentist_result.p_value:.4f}"
                )

                if frequentist_result.is_significant:
                    if data.cvr_b > data.cvr_a:
                        st.success("✅ **Bが優れている**")
                    else:
                        st.success("✅ **Aが優れている**")
                else:
                    st.warning("⚠️ **差は見られない**")

            # わかりやすい比較説明
            st.markdown("---")
            if bayesian_result.prob_b_better > 0.95 and frequentist_result.is_significant and data.cvr_b > data.cvr_a:
                st.success("🎉 **両方の方法でBが優れていると判定されました！** 自信を持ってBを選べます。")
            elif bayesian_result.prob_b_better < 0.05 and frequentist_result.is_significant and data.cvr_a > data.cvr_b:
                st.success("🎉 **両方の方法でAが優れていると判定されました！** 自信を持ってAを選べます。")
            elif not frequentist_result.is_significant and 0.2 < bayesian_result.prob_b_better < 0.8:
                st.info("📊 **両方の方法で判断が難しい結果です。** もっとデータを集めることをお勧めします。")
            else:
                st.warning("⚠️ **2つの方法で結論が異なります。** 詳しく見てみましょう。")

        elif expertise_level == "中級者 (標準)":
            # 中級者向け：標準的な比較
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ベイジアン")
                st.markdown(f"- **BがAより優れている確率**: {bayesian_result.prob_b_better:.1%}")
                st.markdown(f"- **差の期待値**: {bayesian_result.diff_mean:+.4f}")
                st.markdown(
                    f"- **{bayesian_result.credible_level:.0%} 確信区間**: "
                    f"[{bayesian_result.diff_ci_lower:.4f}, {bayesian_result.diff_ci_upper:.4f}]"
                )

                # ベイジアンの結論
                if bayesian_result.prob_b_better > 0.95:
                    st.success("✅ 結論: **Bが優れている**（95%以上の確率）")
                elif bayesian_result.prob_a_better > 0.95:
                    st.success("✅ 結論: **Aが優れている**（95%以上の確率）")
                else:
                    st.info("📊 結論: **判定不能**（どちらが優れているか明確ではない）")

            with col2:
                st.markdown("### 頻度主義")
                st.markdown(f"- **p値**: {frequentist_result.p_value:.6f}")
                st.markdown(f"- **検定統計量**: {frequentist_result.test_statistic:.4f}")
                st.markdown(
                    f"- **{frequentist_result.confidence_level:.0%} 信頼区間**: "
                    f"[{frequentist_result.ci_lower:.4f}, {frequentist_result.ci_upper:.4f}]"
                )

                # 頻度主義の結論
                if frequentist_result.is_significant:
                    if data.cvr_b > data.cvr_a:
                        st.success("✅ 結論: **Bが優れている**（統計的に有意）")
                    else:
                        st.success("✅ 結論: **Aが優れている**（統計的に有意）")
                else:
                    st.info("📊 結論: **有意差なし**（統計的に有意な差は見られない）")

        else:  # 上級者
            # 上級者向け：詳細な比較
            comparison_data = {
                "指標": ["P(B > A)", "点推定 (B-A)", "区間推定", "判定基準", "判定結果"],
                "ベイジアン": [
                    f"{bayesian_result.prob_b_better:.4f}",
                    f"{bayesian_result.diff_mean:+.6f}",
                    f"[{bayesian_result.diff_ci_lower:.6f}, {bayesian_result.diff_ci_upper:.6f}]",
                    f"P(B>A) > 0.95",
                    "B優位" if bayesian_result.prob_b_better > 0.95 else ("A優位" if bayesian_result.prob_b_better < 0.05 else "判定不能")
                ],
                "頻度主義": [
                    "N/A (確率は計算しない)",
                    f"{data.cvr_diff:+.6f}",
                    f"[{frequentist_result.ci_lower:.6f}, {frequentist_result.ci_upper:.6f}]",
                    f"p < {1-confidence_level:.3f}",
                    "B優位" if frequentist_result.is_significant and data.cvr_b > data.cvr_a else ("A優位" if frequentist_result.is_significant and data.cvr_a > data.cvr_b else "有意差なし")
                ]
            }

            import pandas as pd
            df_comparison = pd.DataFrame(comparison_data)
            st.table(df_comparison)

            # 詳細な統計量
            with st.expander("📊 詳細な統計的比較"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    **ベイジアン統計量**
                    - P(B > A) = {bayesian_result.prob_b_better:.6f}
                    - E[B - A] = {bayesian_result.diff_mean:+.8f}
                    - Std[B - A] = {bayesian_result.diff_std:.8f}
                    - {bayesian_result.credible_level:.0%} HDI: [{bayesian_result.diff_ci_lower:.6f}, {bayesian_result.diff_ci_upper:.6f}]
                    - Bayes Factor: {bayesian_result.bayes_factor:.4f if bayesian_result.bayes_factor is not None else 'N/A'}
                    """)
                with col2:
                    st.markdown(f"""
                    **頻度主義統計量**
                    - p-value = {frequentist_result.p_value:.8f}
                    - Test Statistic = {frequentist_result.test_statistic:.6f}
                    - {frequentist_result.confidence_level:.0%} CI: [{frequentist_result.ci_lower:.6f}, {frequentist_result.ci_upper:.6f}]
                    - Significance: {'Yes' if frequentist_result.is_significant else 'No'} (α={1-confidence_level:.3f})
                    """)

        # 比較の可視化
        st.subheader("📊 可視化")
        fig = plot_comparison(data, bayesian_result, frequentist_result)
        st.pyplot(fig)
        plt.close(fig)

        # 解釈のガイド（知識レベル別）
        st.subheader("📖 結果の解釈")

        if expertise_level == "初心者 (シンプル)":
            with st.expander("💡 どちらの方法を使えばいい？"):
                st.markdown("""
                ### 🎲 ベイジアンがおすすめの場合
                - 「どちらが良いか」を確率で知りたい
                - 少ないデータで判断したい
                - 直感的に理解したい

                ### 📊 頻度主義がおすすめの場合
                - 学術論文や報告書で使いたい（一般的によく使われている）
                - たくさんのデータがある
                - 標準的な方法を使いたい

                ### 🎯 迷ったら？
                両方の結果を見て、同じ結論なら自信を持って判断できます！
                """)
        elif expertise_level == "中級者 (標準)":
            st.markdown("""
            #### ベイジアンアプローチの特徴
            - ✅ 「BがAより優れている確率」を直接計算できる
            - ✅ 事前知識を取り込むことができる
            - ✅ 小サンプルでも安定した推論が可能
            - ⚠️ 事前分布の選択に依存する

            #### 頻度主義アプローチの特徴
            - ✅ 標準的な統計手法として広く使われている
            - ✅ 明確な判定基準（有意水準）がある
            - ⚠️ p値の解釈が難しい（「差がない」ことは証明できない）
            - ⚠️ サンプルサイズに敏感

            #### どちらを使うべきか？
            - **ベイジアン**: より直感的な確率解釈が欲しい場合、小サンプルの場合
            - **頻度主義**: 標準的な報告が必要な場合、大規模なサンプルがある場合
            - **両方**: 可能であれば両方の結果を見て、総合的に判断するのがベスト
            """)
        else:  # 上級者
            st.markdown("""
            #### ベイジアン推論の特徴
            **長所:**
            - 事後確率 P(B > A) を直接計算可能（確率的解釈が直感的）
            - 事前分布により専門知識やドメイン知識を組み込める
            - 小サンプルでも安定した推論（正則化効果）
            - 期待損失による意思決定理論と整合的
            - サンプリングの停止規則に依存しない

            **短所:**
            - 事前分布の選択が結果に影響（主観性の問題）
            - 計算コストが高い（MCMCサンプリング）
            - 解釈には確率論の理解が必要

            #### 頻度主義検定の特徴
            **長所:**
            - 広く普及しており、報告・解釈が標準化されている
            - 明確な判定基準（α水準、p値）
            - 事前分布不要（客観性）
            - 計算が高速（解析的な解が存在）

            **短所:**
            - p値の誤解釈が多い（「差がない確率」ではない）
            - H₀を棄却できないことは「差がない」ことを意味しない
            - サンプルサイズに敏感（大サンプルで微小な差も有意に）
            - 多重比較問題、停止規則への依存性

            #### 実務での使い分け
            | 状況 | 推奨アプローチ |
            |------|----------------|
            | 意思決定（ビジネス判断） | ベイジアン（期待損失を考慮） |
            | 学術論文・公的報告 | 頻度主義（標準的な手法） |
            | 小サンプル（n < 100） | ベイジアン（安定した推論） |
            | 大サンプル（n > 1000） | どちらでも（結果は収束） |
            | 事前知識がある | ベイジアン（事前分布を活用） |
            | 完全に中立的な分析 | 頻度主義 or 無情報事前分布 |
            """)

except ValueError as e:
    st.error(f"❌ データエラー: {str(e)}")
    st.info("左のサイドバーで正しいデータを入力してください。")

# フッター（知識レベル別）
st.markdown("---")

if expertise_level == "初心者 (シンプル)":
    st.markdown("""
    ### 💡 使い方のヒント

    1. **プリセットから始めよう**
       - サイドバーで「明確な差がある例」を選択して試してみましょう
       - 次に「差がない例」も試して、結果の違いを確認してみましょう

    2. **自分のデータで試そう**
       - プリセットを「カスタム」に変更
       - 実際のA/Bテストデータを入力してみましょう

    3. **両方の手法を比較しよう**
       - 「比較」タブで、ベイジアンと頻度主義の違いを確認
       - 同じ結論なら自信を持って判断できます！

    ### 📊 実践例

    **ECサイトのボタン色テスト**
    - A: 青ボタン（1000人訪問、100人購入）
    - B: 赤ボタン（1000人訪問、120人購入）

    上のデータをサイドバーに入力して試してみましょう！
    """)

elif expertise_level == "中級者 (標準)":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 💡 使い方のヒント

        1. **プリセットから始める**
           - まずは「明確な差がある例」で動作を確認
           - 次に「微妙な差がある例」や「差がない例」も試す

        2. **自分のデータで試す**
           - プリセットを「カスタム」に変更
           - 実際のA/Bテストデータを入力

        3. **詳細設定を活用**
           - ベイジアン：確信水準を調整
           - 頻度主義：検定方法を選択
        """)

    with col2:
        st.markdown("""
        ### 🎓 学習のポイント

        **ベイジアンアプローチ**
        - 「BがAより良い確率」が直接わかる
        - 小サンプルでも安定した推論
        - 事前知識を活用できる

        **頻度主義アプローチ**
        - 広く使われている標準的な手法
        - p値による明確な判定基準
        - 大サンプルで信頼性が高い
        """)

else:  # 上級者
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 💡 高度な機能

        **事前分布のカスタマイズ**
        - 仮想サンプル数で設定：直感的な入力
        - α・βで直接設定：完全なコントロール
        - 無情報事前分布：Beta(1, 1)

        **検定方法の選択**
        - Z検定：大サンプル、正規近似
        - t検定：Welchの方法、不等分散
        - χ²検定：カテゴリカルデータ
        """)

    with col2:
        st.markdown("""
        ### 🎓 統計的考察

        **ベイジアン vs 頻度主義**
        - 事後確率 vs p値の解釈
        - 確信区間(HDI) vs 信頼区間
        - 期待損失による意思決定
        - ベイズファクターによる証拠評価

        **実装の詳細**
        - MCMCサンプリング（10万サンプル）
        - Beta-Binomial共役事前分布
        - 解析的な事後分布計算
        """)

    with col3:
        st.markdown("""
        ### 📊 応用例

        **小サンプルの意思決定**
        - n_A = 50, conv_A = 10
        - n_B = 50, conv_B = 15
        - → ベイジアンで安定した推論

        **大サンプルの検証**
        - n_A = 10000, conv_A = 2000
        - n_B = 10000, conv_B = 2150
        - → 微小な差も検出可能

        **事前知識の活用**
        - 過去データをpriorとして設定
        - ドメイン知識の組み込み
        """)

st.markdown("---")

st.markdown("""
<div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
    <p style="margin: 0; color: #666;">
        このツールはオープンソースです。フィードバックや改善提案をお待ちしています！<br>
        <small>Powered by Streamlit | ベイジアンA/Bテスト & 頻度主義A/Bテスト</small>
    </p>
</div>
""", unsafe_allow_html=True)