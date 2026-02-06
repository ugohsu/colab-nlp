# 大規模コーパス構築ガイド (CorpusDB)

`CorpusDB` は、大規模なテキストデータと、その形態素解析結果（トークン）を SQLite 上で効率的に管理するためのクラスです。

## 概要と責任範囲

自然言語処理のワークフローにおいて、以下の3つの役割を担います。

1. **Ingest (テキストの蓄積)**
    * 大量のテキストファイルや、外部DB・CSVからのデータを、安全に SQLite に格納します。
    * データの重複排除や、ファイル更新検知による差分取り込みを自動化します。
2. **Tokenize (形態素解析の管理)**
    * 保存されたテキストに対して任意のトークナイザ（Sudachi, Janome, MeCab 等）を適用し、結果を保存します。
    * 処理の中断・再開や、並列処理に向けたバッチ管理をサポートします。
3. **Filter (分析用データの抽出)**
    * 解析結果から「特定の品詞のみを含める」「特定の品詞を除外する」といったサブセットを抽出し、分析専用の軽量なデータベース（Export DB）を作成します。

---

## 利用パターン 1: ファイルシステムからの構築 (Basic)

指定したディレクトリ内のテキストファイル群を読み込み、一つのデータベースで管理する最もシンプルな構成です。

### 1. 初期化とファイル登録

まずデータベースを初期化し、処理対象となるディレクトリをスキャンします。

```python
from colab_nlp import CorpusDB

# データベースの初期化（存在しない場合は作成されます）
db = CorpusDB("my_corpus.db")

# 対象ディレクトリ以下の .txt ファイルを再帰的に登録
# ※ この時点ではファイルパスのみが登録され、中身はまだ読み込まれません
db.register_files("data/text_files")

```

### 2. テキストの取り込み (Ingest)

ファイルの中身を読み込み、データベースに格納します。
この処理はファイル更新日時をチェックするため、後でファイルが修正された場合は自動的に再取得の対象となります。

```python
# テキストデータの読み込みと保存
db.ingest_queue()

```

### 3. 形態素解析 (Tokenize)

保存されたテキストに対して、指定した関数を用いて形態素解析を行います。

```python
!pip install janome
from colab_nlp import tokenize_df

# DB内のテキストを読み出し、解析結果を tokens テーブルに保存
db.tokenize_stored_text(tokenize_df)

```

---

## 利用パターン 2: 既存データのインポートと分離管理

CSVやSQLなど既存のデータソースを利用する場合や、試行錯誤のために「原文データ（Master）」と「解析結果（Work）」を分けたい場合に適した構成です。

### 1. Master DB の構築（インポート）

分析対象のテキストを Master DB に格納し、それを形態素解析したデータ・品詞を絞り込んだデータを Work DB に格納するというデータベースの分離管理が可能です。1 つの Master DB に対して複数の Work DB を対応付ける (たとえば、Work1 は Janome、Work2 は Sudachi、Work3 は Work2 をベースに品詞を絞り込む) ことが可能です。

まず、原文データのみを管理する Master DB を作成します。`ingest_queue` はプレーンテキストで保存された複数のテキストファイルを DB に集約するメソッドですが、すでにスプレッドシートやリレーショナルデータベースの形式でテキストデータを管理している、という場合もあるでしょう。このような場合には `register_content_stream` を使うことで、イテレータからメモリ効率よくデータを取り込めます。

```python
import sqlite3
from colab_nlp import CorpusDB

# Master DB の初期化
master_db = CorpusDB("corpus_master.db")

# データソース（例: 別のSQLiteやCSV）から読み込むイテレータを準備
# yield (doc_id, text_content) の形式である必要があります
conn_src = sqlite3.connect("original_data.db")
cursor = conn_src.cursor()
cursor.execute("SELECT id, body FROM articles")

# ストリームインポート実行
master_db.register_content_stream(cursor)

conn_src.close()

```

### 2. Work DB の初期化（分離管理）

Master DB はそのままに、解析結果だけを保存する Work DB を作成します。これにより、解析辞書や設定を変えて形態素解析を何度でもやり直すことが容易になります。

```python
# master_db_path を指定して Work DB を初期化
work_db = CorpusDB(db_path="corpus_work_v1.db", master_db_path="corpus_master.db")

# Master DB の文書リストを Work DB に同期（解析待ちリストの作成）
work_db.sync_status_from_master()

```

### 3. 形態素解析の実行

Work DB に対して解析を実行します。テキストデータは Master から読み込まれ、トークンは Work に保存されます。

```python
from sudachipy import dictionary
from colab_nlp import tokenize_text_sudachi

# 1. Full辞書を指定して Tokenizer を作成
full_tokenizer = dictionary.Dictionary(dict="full").create()

# 2. 形態素解析用の関数を定義
#    Sudachi 固有の調整（辞書・語形・分割モード）を行いたい場合は
#    以下のように tokenize_text_fn を使って詳細を指定します。
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

# カスタムトークナイザで解析
work_db.tokenize_stored_text(new_tokenizer)

```

---

## 分析用データの抽出 (Filter)

解析が完了したデータベース（Full DB）はサイズが大きくなりがちです。Word2Vec や LDA などの学習に必要な情報だけを抽出し、軽量なデータベースを作成します。

```python
## ノイズになる品詞を除外
def my_filter(df_tokens):
    return filter_tokens_df(df_tok_all, pos_exclude={"補助記号", "空白"})

# フィルタリングされた新しいDBを作成
work_db.export_filtered_tokens_db(
    dst_db_path="analysis_ready.db",
    transform_fn=my_filter
)

```

作成された `analysis_ready.db` はサイズが小さく、分析フェーズでの読み込みが高速になります。

---

`corpus_db.py` の内容を詳細に解析しました。
ご要望の通り、データベースの内部構造（スキーマ）と、クラス・関数の完全なリファレンスを作成しました。

この内容は、先ほどのドキュメントの後半（「利用パターン」の後）に **「付録: 詳細リファレンス」** として追記する形で使用できます。

---

## 付録: 詳細リファレンス

### 1. データベース・スキーマ仕様

#### 1. 原文管理テーブル (Master / Single)

ファイルパスと原文データを管理します。
分離管理（Split Mode）の場合、これらのテーブルは Master DB 側に作成されます。

| テーブル名 | カラム名 | 型 | 備考・役割 |
| --- | --- | --- | --- |
| **documents** | `doc_id` | `INTEGER` | **主キー (Auto Increment)**。システム全体で文書を一意に識別するID。 |
|  | `abs_path` | `TEXT` | **Unique制約**。ファイルの絶対パス、または `imported://` で始まる仮想パス。重複登録を防ぐために使用。 |
|  | `rel_path` | `TEXT` | 指定ルートディレクトリからの相対パス。表示用。 |
|  | `created_at` | `TEXT` | レコードの初回登録日時（ISO 8601形式）。 |
| **status_fetch** | `doc_id` | `INTEGER` | **主キー**。documentsテーブルと1対1で対応。 |
|  | `fetch_ok` | `INTEGER` | **Indexあり**。`0`: 未取得/失敗、`1`: 取得成功。`ingest_queue` の処理対象判定に使用。 |
|  | `fetched_at` | `TEXT` | テキスト取得完了日時。ファイル更新検知（mtimeとの比較）に使用される重要なタイムスタンプ。 |
|  | `updated_at` | `TEXT` | ステータスの最終更新日時。 |
|  | `error_message` | `TEXT` | 読み込みエラー時のスタックトレース等を記録。 |
| **text** | `doc_id` | `INTEGER` | **主キー**。documentsテーブルと1対1で対応。 |
|  | `text` | `TEXT` | テキスト本文データ。 |
|  | `char_count` | `INTEGER` | 文字数。**バッチ処理のメモリ制御**（`max_chars`）に使用され、処理落ちを防ぐ重要な指標。 |

#### 2. 解析管理テーブル (Work / Single)

形態素解析の結果を管理します。
分離管理（Split Mode）の場合、これらのテーブルは Work DB 側に作成され、Master DB の原文を参照しながら解析を行います。

| テーブル名 | カラム名 | 型 | 備考・役割 |
| --- | --- | --- | --- |
| **status_tokenize** | `doc_id` | `INTEGER` | **主キー**。解析ステータスを管理。 |
|  | `tokenize_ok` | `INTEGER` | **Indexあり**。`0`: 未解析/失敗、`1`: 解析完了。`tokenize_stored_text` の処理対象判定に使用。 |
|  | `updated_at` | `TEXT` | ステータスの最終更新日時。 |
|  | `error_message` | `TEXT` | 解析エラー時のログを記録。 |
| **tokens** | `doc_id` | `INTEGER` | **複合主キーの一部**。文書ID。**Indexあり**（文書ごとのトークン取得を高速化）。 |
|  | `token_id` | `INTEGER` | **複合主キーの一部**。文書内でのトークンの出現順序（連番）。文脈復元（N-gram等）に必須。 |
|  | `word` | `TEXT` | **Indexあり**。単語の表層形（Surface）。特定の単語を含む文書の検索に使用。 |
|  | `pos` | `TEXT` | **Indexあり**。品詞情報（例: `名詞-一般`）。`export_filtered_tokens_db` でのフィルタリング条件として多用される。 |
|  | `token_info` | `TEXT` | その他の詳細情報（原形、読み、正規化形など）を格納する **JSON文字列**。 |

> **Note:** `tokens` テーブルは `PRIMARY KEY (doc_id, token_id)` の複合主キーを持ちます。これにより、同一文書内でのトークンの重複を防ぎつつ、高速な挿入を実現しています。

---

### 2. CorpusDB クラスリファレンス

#### 初期化

```python
CorpusDB(db_path="corpus.db", master_db_path=None)

```

* **`db_path`** (str):
    * 作業用（Work）データベースのパス。解析結果（`tokens`）や解析ステータス（`status_tokenize`）が保存されます。
    * `master_db_path` が未指定の場合は、原文データ（`text` 等）もここに保存されます（一元管理モード）。
* **`master_db_path`** (str, optional):
    * 原文データ（Master）を持つデータベースのパスを指定します（分離管理モード）。
    * 指定した場合、`db_path` 側からは Master DB が参照専用として扱われ、解析結果のみが `db_path` に書き込まれます。

---

#### データ登録・取り込み (Ingest Phase)

##### `register_files(root_dir, exts=("*.txt",))`

指定ディレクトリ以下のファイルを走査し、パス情報を `documents` テーブルに登録します。中身の読み込みは行いません。

* **`root_dir`** (str): 走査対象のルートディレクトリ。
* **`exts`** (tuple/list): 対象とする拡張子（例: `("*.txt", "*.csv")`）。

##### `register_content_stream(iterator, batch_size=10000, source_label="imported")`

Python のイテレータから直接データを読み込み、DBに登録します。CSVや他DBからの移行に使用します。

* **`iterator`** (iterable): `(original_id, text_body)` のタプルを順次返すイテレータ。
* **`batch_size`** (int): コミットを行う頻度。
* **`source_label`** (str): 仮想パス生成用のプレフィックス（`imported://<id>` となります）。

##### `ingest_queue(check_updates=True, batch_size=100)`

未処理または更新されたファイルを読み込み、テキストデータを `text` テーブルに保存します。

* **`check_updates`** (bool):
    * `True` (既定): すでに取り込み済み（`fetch_ok=1`）のファイルでも、ファイル更新日時（mtime）がDB保存日時より新しい場合は再取得します。
    * `False`: 未取得（`fetch_ok=0`）のファイルのみを対象とします。
* **`batch_size`** (int): DBコミットを行う頻度（ファイル数単位）。

---

#### 形態素解析 (Tokenize Phase)

##### `tokenize_stored_text(tokenize_fn, doc_ids=None, ...)`

DBに保存されたテキストを読み出し、指定された関数でトークン化して `tokens` テーブルに保存します。

* **`tokenize_fn`** (callable):
    * **入力**: `pandas.DataFrame` (columns: `doc_id`, `text`)
    * **出力**: `pandas.DataFrame` (columns: `doc_id`, `word`, `pos`, `token_info` 等)
    * 実際の解析ロジックを持つ関数を渡します（例: `colab_nlp.tokenize_df`）。
* **`doc_ids`** (list, optional):
    * 特定の文書IDのみを強制的に再解析したい場合に指定します。
    * `None` (既定) の場合は、「未解析（`tokenize_ok=0`）」の文書のみが対象となります。
* **`max_chars`** (int): 1バッチあたりの最大処理文字数（メモリ不足防止用）。
* **`fallback_to_single`** (bool): バッチ処理でエラーが発生した際、1件ずつの処理に切り替えて救済するかどうか（既定 `True`）。

##### `reprocess_tokens(...)`

`tokenize_stored_text` のエイリアス（別名）です。過去のコードとの互換性のために残されています。

##### `process_queue(tokenize_fn, fetch_only=False, check_updates=True)`

`ingest_queue` と `tokenize_stored_text` を連続して実行するラッパーメソッドです。

---

#### 管理・ユーティリティ

##### `export_filtered_tokens_db(dst_db_path, transform_fn, vacuum=False, ...)`

現在の `tokens` テーブルに対してフィルタや変換を適用し、結果を新しいデータベースに保存します。

* **`dst_db_path`** (str): 出力先のDBパス。既存ファイルは上書きされます。
* **`transform_fn`** (callable): `DataFrame` を受け取り、フィルタ後の `DataFrame` を返す関数。
    * **制約**: `token_id` は変更せず、そのまま維持してください（文脈維持のため）。


* **`vacuum`** (bool): 作成後に `VACUUM` コマンドを実行し、ファイルサイズを最小化するかどうか。

##### `sync_status_from_master()`

【分離管理用】Master DB に登録されている文書リストを、Work DB の解析待ちリスト（`status_tokenize`）に同期します。Work DB を新規作成した直後に実行します。

##### `reset_tokens(doc_ids=None, reset_only_fetched=True, vacuum=False)`

解析結果（`tokens`）を削除し、ステータスを「未解析」に戻します。

* **`doc_ids`** (list): 指定したIDのみリセットします。`None` なら全件リセット。
* **`reset_only_fetched`** (bool): 全件リセット時、テキスト取得済みの文書のみを対象とするか（通常は `True` でOK）。

---

### 3. ヘルパー関数

#### `corpus_reader`

大規模な `CorpusDB` から、メモリを節約しつつトークン列を読み込むためのジェネレータです。Gensim (Word2Vec) 等の入力としてそのまま使用できます。

```python
corpus_reader(
    db_path, 
    table_name="tokens", 
    id_col="doc_id",
    word_col="word",
    token_id_col="token_id",
    chunk_size=1000,
    return_mode="tokens" 
)

```

* `db_path` : str
    * SQLiteデータベースへのパス。
* `table_name` : str
    * 読み込み対象のテーブル名（既定: "tokens"）。
* `id_col` : str
    * 文書IDのカラム名（既定: "doc_id"）。
* `word_col` : str
    * 単語のカラム名（既定: "word"）。
* `token_id_col` : str
    * トークン順序を表すカラム名（既定: "token_id"）。
* `chunk_size` : int
    * 一度に読み込む文書IDの範囲（既定: 1000）。
* `return_mode` : str
    * `"tokens"` (既定): `['私', 'は', ...]` のリストを yield します。
    * `"id_tokens"`: `(doc_id, ['私', 'は', ...])` のタプルを yield します。

---

## Tips: 文字コードエラー（Shift_JIS など）への対処

`CorpusDB` は、テキストファイルを **UTF-8** として読み込みます。そのため、Shift_JIS (CP932) や EUC-JP など、異なる文字コードで保存されたファイルが含まれている場合、`ingest_queue` の実行中に読み込みエラーが発生し、処理がスキップされます（`fetch_ok=0` のままになります）。

このような場合、読み込みに失敗したファイルだけを特定し、ツール（`nkf`）を使って UTF-8 に一括変換・上書きすることで解決できます。

### 手順

#### 1. nkf のインストール

Google Colab 上で `nkf` コマンドを使えるようにします。

```python
!sudo apt-get install -y nkf


```

#### 2. エラーファイルの特定と変換

データベースの `status` テーブルから「読み込みに失敗したファイル（`fetch_ok=0` かつ エラーログあり）」のパスを取得し、`nkf -w --overwrite` コマンドで UTF-8 に変換します。

> **⚠️ 注意**: この操作は元のファイルを **上書き** します。必ず元データのバックアップをとった上で実行してください。
> `fetch_ok = 0` は文字コード以外でもさまざまな原因で生じ得ます。この操作ですべての問題が解決するわけではありません。

```python
import sqlite3
import subprocess
import pandas as pd

# DBパス（ご自身の環境に合わせて変更してください）
db_path = "/content/drive/MyDrive/nlp_data/my_corpus.db"

# 1. 読み込みに失敗したファイルのパスを取得
with sqlite3.connect(db_path) as con:
    df_err = pd.read_sql("""
        SELECT d.abs_path
        FROM status_fetch s
        JOIN documents d ON s.doc_id = d.doc_id
        WHERE s.fetch_ok = 0 
          AND s.error_message IS NOT NULL
    """, con)

print(f"変換対象: {len(df_err)} 件")

# 2. nkf で UTF-8 に変換（上書き）
#    -w: UTF-8を出力
#    --overwrite: 元ファイルを上書き
for path in df_err["abs_path"]:
    subprocess.run(["nkf", "-w", "--overwrite", path])

print("変換が完了しました。")


```

#### 3. 処理の再開

変換後、再度 `ingest_queue` を実行すると、先ほどエラーになったファイルが正しく読み込まれます（未処理分として自動的にピックアップされます）。

```python
# エラーだったファイルのみ再処理されます
db.ingest_queue()

```
