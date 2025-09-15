# smaple_dev

# 開発環境をインストール
uvを使ってPythonのパッケージの管理を行う。<br>
[uvのドキュメント](https://docs.astral.sh/uv/)
## uvの環境セットアップ
```
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

uvでPythonのバージョンを管理する。以下のコマンドでPythonを任意のバージョンで管理する。
```
uv python install 3.10 3.11 3.12
```

インストール済みのPythonのバージョンは以下で確認することができる
```
uv python list
```