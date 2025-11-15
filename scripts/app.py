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

st.sidebar.markdown("""
<div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
    <p style="margin: 0; font-size: 14px;">
        💡 <b>初めての方へ</b><br>
        まずは「明確な差がある例」を選択して、どんな分析ができるか試してみましょう！
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

# 詳細設定
with st.sidebar.expander("⚙️ 詳細設定"):
    # ベイジアン設定
    st.markdown("**ベイジアン設定**")

    # 事前分布の設定方法を選択
    prior_mode = st.selectbox(
        "事前分布の設定方法",
        ["無情報事前分布", "A/B個別設定"],
        help="""
        - 無情報事前分布: α=1, β=1 (何も知らない状態)
        - A/B個別設定: グループごとに異なる事前分布を設定
        """
    )

    if prior_mode == "無情報事前分布":
        alpha_prior_a = 1.0
        beta_prior_a = 1.0
        alpha_prior_b = 1.0
        beta_prior_b = 1.0
        st.info("グループA/B共通: α=1.0, β=1.0 (無情報事前分布)")

    else:  # A/B個別設定
        st.markdown("**グループA の事前分布**")

        prior_type_a = st.radio(
            "グループA 設定方法",
            ["パラメータ指定", "データ指定"],
            horizontal=True,
            key="prior_type_a"
        )

        if prior_type_a == "パラメータ指定":
            alpha_prior_a = st.number_input(
                "グループA α",
                min_value=0.1,
                value=1.0,
                step=0.1,
                help="グループAのBeta分布パラメータα",
                key="alpha_a"
            )
            beta_prior_a = st.number_input(
                "グループA β",
                min_value=0.1,
                value=1.0,
                step=0.1,
                help="グループAのBeta分布パラメータβ",
                key="beta_a"
            )
        else:  # データ指定
            prior_n_a = st.number_input(
                "グループA サンプル数",
                min_value=0,
                value=10,
                step=1,
                help="グループAの事前知識のサンプル数",
                key="n_a"
            )
            prior_conv_a = st.number_input(
                "グループA コンバージョン数",
                min_value=0,
                max_value=int(prior_n_a),
                value=min(1, int(prior_n_a)),
                step=1,
                help="グループAの事前知識のコンバージョン数",
                key="conv_a"
            )

            alpha_prior_a = prior_conv_a + 1.0
            beta_prior_a = (prior_n_a - prior_conv_a) + 1.0

            prior_mean_a = prior_conv_a / prior_n_a if prior_n_a > 0 else 0.5
            st.info(f"グループA: α={alpha_prior_a:.1f}, β={beta_prior_a:.1f} (事前平均CVR: {prior_mean_a:.2%})")

        st.markdown("---")
        st.markdown("**グループB の事前分布**")

        prior_type_b = st.radio(
            "グループB 設定方法",
            ["パラメータ指定", "データ指定"],
            horizontal=True,
            key="prior_type_b"
        )

        if prior_type_b == "パラメータ指定":
            alpha_prior_b = st.number_input(
                "グループB α",
                min_value=0.1,
                value=1.0,
                step=0.1,
                help="グループBのBeta分布パラメータα",
                key="alpha_b"
            )
            beta_prior_b = st.number_input(
                "グループB β",
                min_value=0.1,
                value=1.0,
                step=0.1,
                help="グループBのBeta分布パラメータβ",
                key="beta_b"
            )
        else:  # データ指定
            prior_n_b = st.number_input(
                "グループB サンプル数",
                min_value=0,
                value=10,
                step=1,
                help="グループBの事前知識のサンプル数",
                key="n_b"
            )
            prior_conv_b = st.number_input(
                "グループB コンバージョン数",
                min_value=0,
                max_value=int(prior_n_b),
                value=min(1, int(prior_n_b)),
                step=1,
                help="グループBの事前知識のコンバージョン数",
                key="conv_b"
            )

            alpha_prior_b = prior_conv_b + 1.0
            beta_prior_b = (prior_n_b - prior_conv_b) + 1.0

            prior_mean_b = prior_conv_b / prior_n_b if prior_n_b > 0 else 0.5
            st.info(f"グループB: α={alpha_prior_b:.1f}, β={beta_prior_b:.1f} (事前平均CVR: {prior_mean_b:.2%})")

    credible_level = st.slider(
        "確信水準",
        min_value=0.80,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="ベイジアンの確信区間の水準"
    )

    # 頻度主義設定
    st.markdown("**頻度主義設定**")
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
        st.markdown("""
        ベイジアンアプローチは、確率的な推論を行います。
        「BがAより優れている確率」を直接計算できるのが特徴です。
        """)

        with st.spinner("ベイジアン分析を実行中..."):
            bayesian_test = BayesianABTest(
                data=data,
                alpha_prior_a=alpha_prior_a,
                beta_prior_a=beta_prior_a,
                alpha_prior_b=alpha_prior_b,
                beta_prior_b=beta_prior_b,
                credible_level=credible_level,
                n_samples=100000
            )
            bayesian_result = bayesian_test.run()

        # 結果のサマリー
        st.subheader("📋 結果サマリー")

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

        # 期待損失
        if bayesian_result.expected_loss_a is not None and bayesian_result.expected_loss_b is not None:
            st.subheader("💰 期待損失")
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

        # 可視化
        st.subheader("📊 可視化")
        fig = plot_bayesian_distributions(bayesian_test, bayesian_result)
        st.pyplot(fig)
        plt.close(fig)

    # 頻度主義アプローチ
    with tab2:
        st.header("頻度主義A/Bテスト")
        st.markdown("""
        頻度主義アプローチは、仮説検定を用いて統計的有意性を判定します。
        帰無仮説「AとBに差がない」を棄却できるかを検証します。
        """)

        with st.spinner("頻度主義分析を実行中..."):
            frequentist_test = FrequentistABTest(data=data, confidence_level=confidence_level)
            frequentist_result = frequentist_test.run(method=test_method)

        # 結果のサマリー
        st.subheader("📋 結果サマリー")

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

        # 可視化
        st.subheader("📊 可視化")
        fig = plot_frequentist_results(data, frequentist_result)
        st.pyplot(fig)
        plt.close(fig)

    # 比較タブ
    with tab3:
        st.header("ベイジアン vs 頻度主義")
        st.markdown("""
        両方のアプローチを比較して、それぞれの特徴と結論を確認します。
        """)

        # 分析が実行されていることを確認
        if 'bayesian_result' not in locals():
            with st.spinner("ベイジアン分析を実行中..."):
                bayesian_test = BayesianABTest(
                    data=data,
                    alpha_prior_a=alpha_prior_a,
                    beta_prior_a=beta_prior_a,
                    alpha_prior_b=alpha_prior_b,
                    beta_prior_b=beta_prior_b,
                    credible_level=credible_level,
                    n_samples=100000
                )
                bayesian_result = bayesian_test.run()

        if 'frequentist_result' not in locals():
            with st.spinner("頻度主義分析を実行中..."):
                frequentist_test = FrequentistABTest(data=data, confidence_level=confidence_level)
                frequentist_result = frequentist_test.run(method=test_method)

        # 比較サマリー
        st.subheader("📋 比較サマリー")

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

        # 比較の可視化
        st.subheader("📊 可視化")
        fig = plot_comparison(data, bayesian_result, frequentist_result)
        st.pyplot(fig)
        plt.close(fig)

        # 解釈のガイド
        st.subheader("📖 結果の解釈")
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

except ValueError as e:
    st.error(f"❌ データエラー: {str(e)}")
    st.info("左のサイドバーで正しいデータを入力してください。")

# フッター
st.markdown("---")

st.markdown("""
<div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
    <p style="margin: 0; color: #666;">
        このツールはオープンソースです。フィードバックや改善提案をお待ちしています！<br>
        <small>Powered by Streamlit | ベイジアンA/Bテスト & 頻度主義A/Bテスト</small>
    </p>
</div>
""", unsafe_allow_html=True)