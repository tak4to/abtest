# A/Bテスト分析ツール

ベイジアンA/Bテストと頻度主義A/Bテストの両方を実行し、結果を可視化するPythonツールです。

## 特徴

- **ベイジアンA/Bテスト**: Beta分布を使用した確率的推論
- **頻度論的A/Bテスト**: z検定、t検定、カイ二乗検定をサポート
- **豊富な可視化**: 事後分布、信頼区間、確信区間を直感的に可視化
- **インタラクティブ分析**: Jupyter Notebookで簡単にデータを入力して分析
- **GitHub Codespaces対応**: ブラウザ上で即座に実行可能

## クイックスタート

### GitHub Codespacesで実行（推奨）

1. このリポジトリをGitHubで開く
2. 緑色の「Code」ボタンをクリック
3. 「Codespaces」タブを選択
4. 「Create codespace on main」をクリック

Codespacesが起動したら、以下のコマンドでJupyter Notebookを開きます:

```bash
cd notebook
uv run jupyter notebook
```

ブラウザで `interactive_ab_test.ipynb` を開いて、データを入力するだけで分析できます！

### ローカル環境でのセットアップ

#### 1. uvのインストール

[uvのドキュメント](https://docs.astral.sh/uv/)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. 依存関係のインストール

```bash
# プロジェクトディレクトリで実行
uv sync
```

#### 3. Jupyter Notebookの起動

```bash
cd notebook
uv run jupyter notebook
```

`interactive_ab_test.ipynb` を開いて分析を開始してください。

## 使い方

### インタラクティブノートブック（推奨）

`notebook/interactive_ab_test.ipynb` を使用すると、以下のことができます:

1. グループAとグループBのサンプル数と成功数を入力
2. ワンクリックでベイジアン・頻度論的分析を実行
3. 美しい可視化グラフを自動生成
4. 分布表で詳細な統計情報を確認

### Pythonコードでの使用

```python
from src.test_data import TestData, TestMethod
from src.bayesian import BayesianABTest
from src.frequentist import FrequentistABTest
from src.visualization import (
    plot_bayesian_distributions,
    plot_frequentist_results,
    plot_comparison,
    create_distribution_table
)

# データの作成
data = TestData(
    n_a=1000,      # グループAのサンプル数
    conv_a=120,    # グループAの成功数
    n_b=1000,      # グループBのサンプル数
    conv_b=150     # グループBの成功数
)

# ベイジアン分析
bayesian_test = BayesianABTest(data)
bayesian_result = bayesian_test.run()

# 頻度論的分析
frequentist_test = FrequentistABTest(data)
frequentist_result = frequentist_test.run(TestMethod.Z_TEST)

# 分布表の表示
table = create_distribution_table(data, bayesian_result, frequentist_result)
print(table)

# 可視化
import matplotlib.pyplot as plt

# ベイジアン分布の可視化
plot_bayesian_distributions(bayesian_test, bayesian_result)
plt.show()

# 頻度論的結果の可視化
plot_frequentist_results(data, frequentist_result)
plt.show()

# 両者の比較
plot_comparison(data, bayesian_result, frequentist_result)
plt.show()
```

## プロジェクト構造

```
abtest/
├── .devcontainer/          # GitHub Codespaces設定
│   └── devcontainer.json
├── .github/
│   └── workflows/          # GitHub Actions
│       └── test.yml
├── src/                    # ソースコード
│   ├── __init__.py
│   ├── test_data.py        # データモデル
│   ├── bayesian.py         # ベイジアンA/Bテスト
│   ├── frequentist.py      # 頻度論的A/Bテスト
│   ├── results.py          # 結果データクラス
│   ├── comparison.py       # 比較ツール
│   └── visualization.py    # 可視化機能 ✨NEW
├── notebook/               # Jupyter Notebook
│   ├── trial.ipynb
│   └── interactive_ab_test.ipynb  ✨NEW
├── scripts/                # スクリプト
│   ├── main.py
│   └── examples.py
├── pyproject.toml          # プロジェクト設定
└── README.md
```

## 可視化機能

### 1. ベイジアン分布の可視化

- **事後分布**: グループAとBのBeta分布
- **差の分布**: BとAの差の分布とヒストグラム
- **確率の比較**: どちらが優れているかの確率
- **統計サマリー**: 事後分布のパラメータ、確信区間、ベイズファクター

### 2. 頻度論的結果の可視化

- **コンバージョン率の比較**: エラーバー付きの棒グラフ
- **統計サマリー**: 検定統計量、p値、信頼区間

### 3. ベイジアンvs頻度論的の比較

- **区間の比較**: 確信区間と信頼区間の重ね合わせ
- **結論の比較**: 両アプローチの判断結果
- **詳細な比較表**: 主要メトリクスの並列比較

## 分析手法

### ベイジアンアプローチ

- **事前分布**: Beta(1, 1) = 無情報事前分布（デフォルト）
- **事後分布**: Beta分布を使用
- **推論**: モンテカルロサンプリング（100,000サンプル）
- **メトリクス**:
  - P(B > A): BがAより優れている確率
  - 確信区間: 95%確信区間（デフォルト）
  - 期待損失: 各選択肢の期待される損失
  - ベイズファクター: 証拠の強さ

### 頻度論的アプローチ

- **Z検定**: 大サンプル向け（n ≥ 30）
- **t検定**: Welchの方法、小〜中サンプル向け
- **カイ二乗検定**: カテゴリカルデータの独立性検定
- **メトリクス**:
  - p値: 帰無仮説の下でのデータの確率
  - 信頼区間: 95%信頼区間（デフォルト）
  - 検定統計量: z値、t値、χ²値

## 依存関係

- Python >= 3.12
- numpy >= 2.3.4
- scipy >= 1.16.3
- pandas >= 2.3.2
- matplotlib >= 3.10.7
- seaborn >= 0.13.2

開発用:
- ipykernel >= 6.30.1
- pytest >= 8.4.2

## 依存関係の管理

### パッケージの追加

```bash
uv add <package-name>
```

### 開発用パッケージの追加

```bash
uv add --dev <package-name>
```

### パッケージの削除

```bash
uv remove <package-name>
```

## テスト

```bash
uv run pytest
```

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します！バグ報告や機能リクエストは、Issueで報告してください。

## サンプルシナリオ

インタラクティブノートブックには、以下のサンプルシナリオが含まれています:

1. **明らかな差がある場合**: CVR 10% vs 15%
2. **差がほとんどない場合**: CVR 10% vs 10.5%
3. **サンプルサイズが小さい場合**: 100サンプルずつ

これらのシナリオを実行して、ベイジアンと頻度論的アプローチの違いを体験してください！

## よくある質問

### Q: ベイジアンと頻度論的、どちらを使うべきですか？

A: 両方とも有用ですが、使い分けのポイント:

- **ベイジアン**:
  - 「BがAより優れている確率」が知りたい場合
  - 小サンプルでも安定した推論が必要な場合
  - 継続的なモニタリングが必要な場合

- **頻度論的**:
  - 伝統的な統計的有意性検定が必要な場合
  - 学術的な文脈で報告する場合
  - 決められたサンプルサイズで1回だけ分析する場合

### Q: 必要なサンプルサイズは？

A: 検出したい効果量によります。一般的には:
- 大きな差（5%以上）: 各グループ数百サンプル
- 中程度の差（2-5%）: 各グループ数千サンプル
- 小さな差（<2%）: 各グループ数万サンプル

詳細なサンプルサイズ計算は、統計的検出力分析を使用してください。

### Q: GitHub Codespacesの料金は？

A: GitHub Freeプランでは月60時間まで無料で使用できます。詳細は[GitHubの料金ページ](https://github.com/pricing)をご確認ください。

## 参考文献

- Evan Miller. ["How Not To Run An A/B Test"](https://www.evanmiller.org/how-not-to-run-an-ab-test.html)
- John K. Kruschke. "Bayesian estimation supersedes the t test"
- VWO. ["Bayesian vs Frequentist A/B Testing"](https://vwo.com/ab-testing/)
