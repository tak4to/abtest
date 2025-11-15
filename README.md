# A/Bテスト分析ツール

ベイジアンA/Bテストと頻度主義A/Bテストの両方を体験できるインタラクティブなツールです。

## 🚀 クイックスタート

### 🌐 オンラインで体験（Pythonのインストール不要）

**Streamlit Community Cloud**でホストされているアプリを直接体験できます！
URLを知っているだけで、誰でもブラウザから利用できます。

デプロイ方法は下記の「Streamlit Cloudへのデプロイ」セクションを参照してください。

### 💻 ローカルで起動

```bash
# 依存関係のインストール
uv sync

# Streamlitアプリを起動
uv run streamlit run scripts/app.py
```

ブラウザが自動的に開き、アプリケーションが表示されます。

## 📊 機能

- **ベイジアンA/Bテスト**: 確率的推論によるA/Bテスト
  - BがAより優れている確率を直接計算
  - 事後分布の可視化
  - 期待損失の計算
  - ベイズファクターの計算

- **頻度主義A/Bテスト**: 統計的仮説検定によるA/Bテスト
  - Z検定、t検定、カイ二乗検定
  - p値と信頼区間の計算
  - 統計的有意性の判定

- **比較機能**: 両方のアプローチを並べて比較
  - 区間推定の比較
  - 結論の比較
  - 解釈のガイド

## 📖 使い方

1. **サイドバー**でデータを入力
   - プリセットから選択するか、カスタムデータを入力
   - グループAとBのサンプルサイズとコンバージョン数を設定

2. **タブ**で結果を確認
   - **ベイジアンアプローチ**: 確率的な推論結果
   - **頻度主義アプローチ**: 統計的仮説検定の結果
   - **比較**: 両方のアプローチを並べて比較

3. **詳細設定**で分析パラメータを調整
   - ベイジアン: 事前分布、確信水準
   - 頻度主義: 検定方法、信頼水準

## 🛠️ 開発環境のセットアップ

### uvをインストール
uvを使ってPythonのパッケージの管理を行う。
[uvのドキュメント](https://docs.astral.sh/uv/)

#### uvの環境セットアップ
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Pythonのバージョン管理
```bash
# Pythonのインストール
uv python install 3.10 3.11 3.12

# インストール済みのPythonバージョンを確認
uv python list
```

### 環境構築
```bash
# pyproject.tomlを使って仮想環境を構築
uv sync

# 新規プロジェクトの初期化（必要な場合）
uv init プロジェクト名
```

### 依存関係の管理
```bash
# パッケージの追加
uv add pandas

# 開発用パッケージの追加
uv add --dev pytest

# パッケージの削除
uv remove pandas
```

## 📁 プロジェクト構造

```
.
├── scripts/
│   ├── app.py           # Streamlitアプリケーション
│   ├── main.py          # CLIツール
│   └── examples.py      # サンプルコード
├── src/
│   ├── bayesian.py       # ベイジアンA/Bテスト
│   ├── frequentist.py    # 頻度主義A/Bテスト
│   ├── visualization.py  # 可視化機能
│   ├── test_data.py      # データ構造定義
│   └── results.py        # 結果データ構造
├── .streamlit/
│   └── config.toml       # Streamlit設定
├── requirements.txt      # 依存関係（Streamlit Cloud用）
└── pyproject.toml        # プロジェクト設定
```

## 🎯 使用例

### プリセットデータで試す
1. サイドバーで「明確な差がある例」を選択
2. 各タブで結果を確認
3. ベイジアンと頻度主義の結論を比較

### カスタムデータで分析
1. サイドバーで「カスタム」を選択
2. 独自のデータを入力
3. 詳細設定で分析パラメータを調整

## 📚 技術スタック

- **Python 3.12+**
- **Streamlit**: Webアプリケーションフレームワーク
- **NumPy/SciPy**: 数値計算・統計計算
- **Matplotlib/Seaborn**: データ可視化
- **Pandas**: データ処理

## 🚀 Streamlit Cloudへのデプロイ

Pythonに触れないユーザーでもURLからアクセスして体験できるように、Streamlit Community Cloudにデプロイできます。

### デプロイ手順

1. **GitHubリポジトリを準備**
   - このリポジトリをGitHubにプッシュ済みであることを確認

2. **Streamlit Community Cloudにアクセス**
   - [share.streamlit.io](https://share.streamlit.io) にアクセス
   - GitHubアカウントでログイン

3. **新しいアプリをデプロイ**
   - "New app" をクリック
   - リポジトリを選択: `tak4to/abtest`
   - ブランチを選択: `main` または適切なブランチ
   - Main file path を入力: `scripts/app.py`
   - "Deploy" をクリック

4. **デプロイ完了**
   - 数分でデプロイが完了します
   - 自動生成されたURLが発行されます（例: `https://abtest-xxx.streamlit.app`）
   - このURLを共有することで、誰でもアプリを利用できます

### デプロイに必要なファイル

以下のファイルがリポジトリに含まれています：

- `requirements.txt`: 依存関係の定義
- `scripts/app.py`: Streamlitアプリケーション本体
- `.streamlit/config.toml`: Streamlit設定（オプション）

### 注意事項

- デプロイは無料ですが、リソース制限があります
- プライベートリポジトリの場合は、Streamlitに権限を付与する必要があります
- デプロイ後、アプリは24時間アクティブでない場合、スリープ状態になります

## 📄 ライセンス

MIT License
