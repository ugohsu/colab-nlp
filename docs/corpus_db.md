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

## 設定変更と再処理（高速化）

「辞書を変えたい」「分割モード（A/C単位）を変えたい」といった理由で、**テキスト読み込みはスキップして形態素解析だけをやり直したい**場合は、以下のメソッドを使用します。

ファイルアクセスが発生しないため、`process_queue` よりも高速に試行錯誤のサイクルを回せます。

### 1. 解析結果のリセット

まず、既存の解析結果（tokens）を消去し、ステータスを「未解析」に戻します。

```python
# tokens テーブルを全消去し、tokenize_ok フラグをリセット
db.reset_tokens()

# ※ 特定の doc_id だけリセットすることも可能です
# db.reset_tokens(doc_ids=[1, 2, 3])
```

### 2. 再処理の実行

DB 内に保存されているテキストデータを使って再解析を実行します。

以下は、「Sudachi の Full 辞書」 を使い、「表層形（surface）」 で解析し直す例です。

```python
# 1. Full辞書のインストール（未インストールの場合）
!pip install sudachidict_full

from sudachipy import dictionary
from colab_nlp import tokenize_text_sudachi

# 2. Full辞書を指定して Tokenizer を作成
full_tokenizer = dictionary.Dictionary(dict="full").create()

# 3. 再処理用の関数を定義
#    Sudachi 固有の調整（辞書・語形・分割モード）を行いたい場合は
#    tokenize_text_fn を使って詳細を指定します。
def new_tokenizer(df):
    return tokenize_df(
        df,
        tokenize_text_fn=lambda text: tokenize_text_sudachi(
            text,
            tokenizer=full_tokenizer,   # 作成したFull辞書Tokenizer
            split_mode="C",             # 分割モード (A/B/C)
            word_form="surface"         # 表層形を採用 (辞書形なら "dictionary")
        )
    )

# 4. DB内のテキストを使って再解析
db.reprocess_tokens(new_tokenizer)
```

このメソッドは バッチ処理（まとめて処理） に対応しており、大量データでも高速に動作します。

> Note: `process_queue` は「未処理のファイル読み込み」を行いますが、`reprocess_tokens` は「DB内のテキスト再利用」を行います。

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

## パフォーマンスと検索のベストプラクティス

`CorpusDB` は大規模データを扱えるように設計されていますが、データの取り出し方（SQL）によっては処理時間が長くなったり、メモリ不足でクラッシュしたりすることがあります。

以下のリストを参考に、効率的なクエリを記述してください。

### ✅ 推奨される検索（高速・安全）

インデックス（索引）が効くカラムを使った検索は、データ量が数億行になっても一瞬で終わります。

| 検索パターン | SQL例 | 理由 |
| :--- | :--- | :--- |
| **単語で検索** | `SELECT * FROM tokens WHERE word = '猫'` | `word` にインデックスがあるため爆速です。 |
| **品詞で検索** | `SELECT * FROM tokens WHERE pos = '名詞'` | `pos` にインデックスがあるため高速です。 |
| **文書IDで検索** | `SELECT * FROM tokens WHERE doc_id = 100` | `doc_id` は主キーの一部であり最速です。 |
| **複合条件** | `SELECT * FROM tokens WHERE word = '猫' AND pos = '名詞'` | 両方のインデックスを活用して効率よく絞り込みます。 |
| **範囲指定** | `SELECT * FROM tokens WHERE doc_id BETWEEN 1 AND 1000` | 文書ごとの一括取得は非常に高速です。Python側でのループ処理に適しています。 |

### ⚠️ 避けるべき検索（低速・メモリ枯渇の危険あり）

インデックスがないカラムでの検索や、全データを一度に読み込む処理は避けてください。

| 検索パターン | SQL例 | 理由・対策 |
| :--- | :--- | :--- |
| **全件取得** | `SELECT * FROM tokens` | **危険**。数千万〜数億行をメモリに展開しようとして Colab がクラッシュします。必ず `WHERE` か `LIMIT` をつけてください。 |
| **詳細情報の検索** | `WHERE token_info LIKE '%読み%'` | JSON 文字列の中身検索はインデックスが効きません。全行スキャンが発生し、非常に遅くなります。 |
| **関数の使用** | `WHERE length(word) > 5` | カラムに対して関数（`length`など）を使うと、基本的にインデックスが使われず全行計算になります。 |
| **あいまい検索** | `WHERE word LIKE '%猫%'` | `%` で始まるあいまい検索（中間一致・後方一致）はインデックスが効きにくいです。`word = '猫'`（完全一致）や `word LIKE '猫%'`（前方一致）を推奨します。 |

### 💡 Python で処理する場合のコツ

「特定の単語を含まない行」や「複雑な条件」で絞り込みたい場合は、SQL だけで頑張らず、**「doc_id で少しずつ取り出して Python で処理する」** のが最も安全で確実です。

```python
# 良い例：1000文書ずつ読み込んで、Python (Pandas) で複雑なフィルタをする
chunk_size = 1000
for start_id in range(1, max_id + 1, chunk_size):
    # doc_id の範囲指定はインデックスが効くので高速
    df_chunk = pd.read_sql(
        f"SELECT * FROM tokens WHERE doc_id >= {start_id} AND doc_id < {start_id + chunk_size}", 
        con
    )
    
    if df_chunk.empty:
        continue
        
    # Pythonのメモリ上で複雑な絞り込み（ここは自由自在！）
    # 例：token_info の中の文字列を検索するなど
    # target = df_chunk[df_chunk['token_info'].str.contains(...)]
```

---

## 付録: 詳細リファレンス

### 1. データベース・スキーマ仕様

`CorpusDB` が作成・管理する SQLite データベースのテーブル定義です。

#### `documents` テーブル
ファイルパスなどのメタデータを管理します。

| カラム名 | 型 | 制約 | 説明 |
| :--- | :--- | :--- | :--- |
| `doc_id` | `INTEGER` | `PK`, `AUTOINCREMENT` | 文書ID（他のテーブルでも主キーとして使用） |
| `abs_path` | `TEXT` | `UNIQUE` | ファイルの絶対パス |
| `rel_path` | `TEXT` | | `register_files` 実行時のルートディレクトリからの相対パス |
| `created_at` | `TEXT` | | 登録日時（ISO 8601形式） |

#### `status` テーブル
各文書の処理進捗とエラー状態を管理します。

| カラム名 | 型 | 制約 | 説明 |
| :--- | :--- | :--- | :--- |
| `doc_id` | `INTEGER` | `PK`, `FK` | `documents.doc_id` への外部キー |
| `fetched_at` | `TEXT` | | テキスト取得（ファイル読み込み）完了日時 |
| `fetch_ok` | `INTEGER` | `DEFAULT 0` | テキスト取得成功フラグ（0:未, 1:済） |
| `tokenize_ok` | `INTEGER` | `DEFAULT 0` | 形態素解析成功フラグ（0:未, 1:済） |
| `error_message` | `TEXT` | | エラー発生時のトレースバック情報 |
| `updated_at` | `TEXT` | | 最終更新日時 |

#### `text` テーブル
読み込んだ原文テキストを格納します。

| カラム名 | 型 | 制約 | 説明 |
| :--- | :--- | :--- | :--- |
| `doc_id` | `INTEGER` | `PK`, `FK` | `documents.doc_id` への外部キー |
| `char_count` | `INTEGER` | | 文字数 |
| `text` | `TEXT` | | 原文テキストデータ |

#### `tokens` テーブル
形態素解析の結果を格納します。`doc_id` と `token_id` の複合主キーにより一意性が保たれます。

| カラム名 | 型 | 制約 | 説明 |
| :--- | :--- | :--- | :--- |
| `doc_id` | `INTEGER` | `FK` | `documents.doc_id` への外部キー |
| `token_id` | `INTEGER` | | 文書内でのトークン通し番号（0始まり） |
| `word` | `TEXT` | | トークン（表層形または辞書形など） |
| `pos` | `TEXT` | | 品詞（大分類） |
| `token_info` | `TEXT` | | 詳細情報の JSON 文字列（読み、原形など） |

> **Primary Key**: `(doc_id, token_id)`
> **Index**: `doc_id`, `word`, `pos` にインデックスが作成されます。

---

### 2. API リファレンス (CorpusDB クラス)

ユーザーが利用する主要なメソッドの一覧です。

#### `__init__(self, db_path="corpus.db")`
データベースに接続し、テーブルが存在しない場合は作成します。

- **Parameters**
  - `db_path` (str): SQLite データベースファイルのパス。Google Drive 上のパスを推奨。

#### `register_files(self, root_dir, exts=("*.txt",))`
指定ディレクトリ以下のファイルを走査し、パス情報を `documents` テーブルに登録します。テキストの中身は読み込みません。

- **Parameters**
  - `root_dir` (str | Path): 走査対象のルートディレクトリ。
  - `exts` (tuple | list): 対象とする拡張子のリスト（例: `["*.txt", "*.md"]`）。

#### `process_queue(self, tokenize_fn)`
未処理（`tokenize_ok=0`）のファイルを順次読み込み、解析して保存します。ファイルアクセスを伴います。

- **Parameters**
  - `tokenize_fn` (callable): `DataFrame(columns=[doc_id, text])` を受け取り、`DataFrame(columns=[doc_id, word, pos, ...])` を返す関数。

#### `reset_tokens(self, doc_ids=None, *, vacuum=False, reset_only_fetched=True)`
解析結果（`tokens`）を削除し、ステータス（`tokenize_ok`）を未完了（0）に戻します。

- **Parameters**
  - `doc_ids` (list | None): リセット対象の ID リスト。`None` の場合は全件リセット。
  - `vacuum` (bool): `True` の場合、削除後に `VACUUM` コマンドを実行して DB ファイルサイズを最適化する。
  - `reset_only_fetched` (bool): `True` の場合、テキスト取得済み（`fetch_ok=1`）のデータのみステータスを戻す。全件リセット時は通常 `True` でよい。

#### `reprocess_tokens(self, tokenize_fn, *, doc_ids=None, ...)`
DB 内のテキスト（`fetch_ok=1`）を使用して再解析を行います。ファイルアクセスが発生しないため高速です。

- **Parameters**
  - `tokenize_fn` (callable): 再解析に使用するトークナイズ関数。
  - `doc_ids` (list | None): 対象 ID のリスト。`None` の場合は「テキスト取得済みかつ未解析（`tokenize_ok=0`）」の全件を対象とする。
  - `max_chars` (int): 1バッチあたりの最大文字数（メモリ保護用。既定 `200,000`）。
  - `max_docs` (int): 1バッチあたりの最大文書数（既定 `500`）。
  - `progress_every_batches` (int): 進捗表示を行うバッチ間隔（既定 `10`）。
  - `fallback_to_single` (bool): バッチ処理でエラーが出た際、1件ずつの処理に切り替えて救済するかどうか（既定 `True`）。

---

### 共通ユーティリティ関数 (corpus_reader)

大規模データ（SQLite データベース）から、メモリを節約しつつトークン列を読み込むためのヘルパー関数（ジェネレータ）です。

```python
corpus_reader(
    db_path,
    *,
    table_name="tokens",
    id_col="doc_id",
    word_col="word",
    chunk_size=1000,
) -> Generator[list[str]]
```

#### 引数

* **`db_path`** (必須)
* 型: `str`
* SQLite データベースファイルのパス（例: `"corpus.db"`）。


* **`table_name`**
* 型: `str`
* 既定: `"tokens"`
* 読み込み対象のテーブル名。


* **`id_col`**
* 型: `str`
* 既定: `"doc_id"`
* 文書IDのカラム名。データの読み込み範囲を制御するために使用されます。


* **`word_col`**
* 型: `str`
* 既定: `"word"`
* 単語のカラム名。


* **`chunk_size`**
* 型: `int`
* 既定: `1000`
* 一度に DB から読み込む文書IDの範囲（幅）。
* 大きくすると DB アクセス回数は減りますが、メモリ使用量が増えます。

#### 戻り値 (Yields)

* **`list[str]`**
* 1文書に含まれるトークンのリストを、文書ごとに順番に返します（yield）。
