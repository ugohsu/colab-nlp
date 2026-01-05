# 大規模コーパス構築ガイド (CorpusDB)

本ドキュメントでは、`CorpusDB` クラスを使用して、大量のテキストファイルを Google Colaboratory 上で安全にデータベース化する方法を説明します。

## 概要とメリット

数万〜数百万ファイル規模のテキストデータを扱う場合、すべてのデータを一度にメモリ（Pandas DataFrame）に読み込むと、Colab のメモリ制限を超えてクラッシュする恐れがあります。

`CorpusDB` は以下の設計により、この問題を解決します。

* **省メモリ設計**: テキストの中身は処理直前まで読み込まず、パス情報のみを先に管理します。
* **安全な逐次処理**: 1ファイル処理するごとに DB 接続をコミット・切断するため、メモリリークやデータ損失を防ぎます。
* **中断・再開**: 処理状況を記録しているため、タイムアウトやエラーで止まっても「未処理のファイル」から即座に再開できます。

---

## 実行手順

### 1. ライブラリの準備

```python
# colab-nlp を clone 済みでパスが通っている前提
from colab_nlp import tokenize_df, CorpusDB
```

### 2. データベースの初期化

データベースファイル（SQLite）を指定して初期化します。 Google Drive 上のパスを指定すると、セッションが切れてもデータが残るため推奨されます。

```python
# Google Drive をマウント（必要に応じて）
from google.colab import drive
drive.mount("/content/drive")

# DBファイルのパスを指定（Google Drive 上のパスを推奨）
db_path = "/content/drive/MyDrive/nlp_data/my_corpus.db"
db = CorpusDB(db_path)
```

### 3. ファイルの登録

指定したフォルダ配下のファイルを再帰的に検索し、パス情報のみを DB に登録します。 この段階ではテキストの中身を読み込まないため、大量のファイルがあっても高速に完了します。

```python
# テキストデータが格納されているルートフォルダ
root_dir = "/content/drive/MyDrive/path/to/text_data"

# 拡張子を指定して登録（リストで複数指定可能）
db.register_files(root_dir, exts=["*.txt"])
```

### 4. トークナイザ関数の定義

CorpusDB は処理の内部で「DataFrame を受け取り、形態素解析済みの DataFrame を返す関数」を必要とします。 `tokenize_df` の設定（エンジンや辞書、フィルタ設定など）を固定したラッパー関数を定義します。`tokenize_df` の詳細 (engin にあわせてインポートしなければならないライブラリなど) は [`tokenization.md`](./tokenization.md) を参照してください。

```python
# engine にあわせた追加のライブラリのインポートが必要。ここでは、大規模データという要件にあわせて sudachi を採用する
!pip install sudachipy sudachidict_core

# ラッパー関数の定義 (id_col, text_col, token_id_col は省略してもよい)
def my_tokenizer(df):
    """
    CorpusDB 用のトークナイズ関数
    入力: df (columns: [doc_id, text])
    出力: df (columns: [doc_id, word, pos, token_info, ...])
    """
    return tokenize_df(
        df,
        engine="sudachi",     # または "janome"
        id_col="doc_id",      # DBのカラム名と合わせる
        text_col="text",
        token_id_col="token_id",
    )
```

### 5. 処理の実行 (構築開始)

未処理のファイルを順次処理します。 このセルは何度実行しても大丈夫です。途中で Colab がタイムアウトしたりエラーで止まったりしても、再度実行すれば「まだ終わっていないファイル」から処理を再開します。

```python
# 構築処理の実行
db.process_queue(my_tokenizer)
```

---

## 構築したデータの利用方法

作成された corpus.db は SQLite 形式です。pandas と sqlite3 を使って簡単にデータを抽出できます。

### データの読み出し例

`token_info` カラムは JSON 文字列として保存されているため、読み出し時に `json.loads` で辞書型に復元すると便利です。

```python
import sqlite3
import pandas as pd
import json

db_path = "/content/drive/MyDrive/nlp_data/my_corpus.db"

with sqlite3.connect(db_path) as con:
    # 例：特定の単語を含むトークン情報を取得
    df = pd.read_sql("""
        SELECT 
            d.rel_path, 
            t.word, 
            t.pos, 
            t.token_info
        FROM tokens t
        JOIN documents d ON t.doc_id = d.doc_id
        WHERE t.word = '猫'
        LIMIT 10
    """, con)

# JSON 文字列を辞書型に復元
if not df.empty:
    df["token_info"] = df["token_info"].apply(json.loads)

# 結果の確認
print(df.head())
# df.iloc[0]["token_info"]["reading"] などでアクセス可能
```
