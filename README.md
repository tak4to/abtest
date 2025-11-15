# A/Bテスト分析ツール

ベイジアンA/Bテストと頻度主義A/Bテストの両方を体験できるインタラクティブなWebアプリケーションです。

## 特徴

- **ベイジアンアプローチ**: 確率的な推論により「BがAより優れている確率」を直接計算
- **頻度主義アプローチ**: 統計的仮説検定（Z検定、t検定、カイ二乗検定）による分析
- **インタラクティブな可視化**: 事後分布、差の分布、確信区間/信頼区間の比較
- **プリセットデータ**: 様々なシナリオを簡単に試せるサンプルデータ
- **詳細設定**: 事前分布、検定方法、信頼水準などをカスタマイズ可能

## デモ

URLを知っているだけで、誰でもA/Bテストを体験できます。

## ローカルでの実行方法

### 必要要件

- Python 3.12以上
- uv（Pythonパッケージマネージャー）

### uvのインストール

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### セットアップと実行

1. リポジトリをクローン

```bash
git clone <repository-url>
cd abtest
```

2. 依存関係をインストール

```bash
uv sync
```

3. Streamlitアプリを起動

```bash
uv run streamlit run scripts/app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

## Streamlit Cloudへのデプロイ方法

1. GitHubにリポジトリをプッシュ
2. [Streamlit Cloud](https://streamlit.io/cloud)にアクセスしてサインイン
3. "New app"をクリック
4. リポジトリを選択し、以下の設定を入力:
   - **Main file path**: `scripts/app.py`
   - **Python version**: 3.12
5. "Deploy"をクリック

数分後、アプリが公開されます。共有可能なURLが発行され、そのURLを知っている人なら誰でもアクセスできます。

## 使い方

### 1. データ入力

左のサイドバーで以下を設定できます:

- **プリセット**: 様々なシナリオのサンプルデータを選択
  - 明確な差がある例
  - 微妙な差がある例
  - 差がない例
  - 小サンプルの例

- **グループA/Bのデータ**:
  - サンプルサイズ（訪問者数）
  - コンバージョン数

- **詳細設定** (オプション):
  - ベイジアン分析の事前分布パラメータ
  - 確信水準/信頼水準
  - 検定方法（Z検定、t検定、カイ二乗検定）

### 2. 結果の確認

3つのタブで異なる視点から結果を確認できます:

#### ベイジアンアプローチ
- 事後分布の可視化
- BがAより優れている確率
- 期待損失（どちらを選ぶべきか）
- 確信区間

#### 頻度主義アプローチ
- p値と検定統計量
- 統計的有意性の判定
- 信頼区間
- コンバージョン率の比較

#### 比較
- 両アプローチの結果を並べて表示
- 確信区間と信頼区間の比較
- 結論の一致/不一致を確認

## プロジェクト構成

```
abtest/
├── scripts/
│   ├── app.py              # Streamlitアプリケーション
│   ├── main.py             # CLIツール
│   └── examples.py         # 使用例
├── src/
│   ├── bayesian.py         # ベイジアンA/Bテスト実装
│   ├── frequentist.py      # 頻度主義A/Bテスト実装
│   ├── test_data.py        # データクラス
│   ├── results.py          # 結果クラス
│   ├── visualization.py    # 可視化関数
│   └── comparison.py       # 比較分析
├── .streamlit/
│   └── config.toml         # Streamlit設定
├── requirements.txt        # 依存パッケージ
├── pyproject.toml          # プロジェクト設定
└── README.md               # このファイル
```

## 依存パッケージ

- pandas>=2.3.2
- scipy>=1.16.3
- numpy>=2.3.4
- matplotlib>=3.10.7
- seaborn>=0.13.2
- streamlit>=1.51.0

## 技術詳細

### ベイジアンA/Bテスト

- **事前分布**: Beta分布（デフォルト: Beta(1, 1) = 一様分布）
- **事後分布**: Beta-Binomial共役性を利用
- **サンプリング**: モンテカルロシミュレーション（100,000サンプル）
- **評価指標**:
  - P(B > A): BがAより優れている確率
  - 期待損失: 各選択肢を選んだ場合の期待される損失
  - ベイズファクター: 仮説の支持度

### 頻度主義A/Bテスト

- **Z検定**: 大サンプル向け（正規近似）
- **t検定**: Welchの方法（等分散性を仮定しない）
- **カイ二乗検定**: カテゴリカルデータの独立性検定
- **信頼区間**: Wilson score method

## ライセンス

MIT License

## 参考資料

- [ベイジアン統計学入門](https://www.example.com)
- [統計的仮説検定の基礎](https://www.example.com)
- [A/Bテストの実践ガイド](https://www.example.com)

## 開発

### テストの実行

```bash
uv run pytest
```

### 新しい依存パッケージの追加

```bash
uv add <package-name>
```

### 開発用パッケージの追加

```bash
uv add --dev <package-name>
```

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。
