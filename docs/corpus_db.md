## 使い方

```python
# 1. ライブラリの準備
import sys
# colab-nlp を clone 済みでパスが通っている前提
# sys.path.append("/content/colab-nlp") 
from colab_nlp import tokenize_df

# 作成した corpus_db.py を import
from colab_nlp import CorpusDB

# 2. DBの初期化
# DBファイル名はお好みで変更してください
db = CorpusDB("my_corpus.db")

# 3. ファイルの登録
# ここでは Google Drive 内のフォルダを指定する例です
# 再帰的に検索されるので、最上位のフォルダを指定すればOKです
root_dir = "/content/drive/MyDrive/path/to/text_data"
db.register_files(root_dir, exts=["*.txt"])

# 4. トークナイザ関数の定義
# CorpusDB は (df -> df) という変換関数を求めているので、
# tokenize_df の設定（エンジンや辞書など）をここで固定してラップします。
def my_tokenizer(df):
    return tokenize_df(
        df,
        engine="janome",      # または "sudachi"
        id_col="doc_id",      # DBのカラム名と合わせる
        text_col="text",
        token_id_col="token_id",
        # 必要に応じてフィルタリング
        pos_exclude=["空白", "補助記号", "記号"], 
    )

# 5. 処理の実行
# ループ処理が始まります。
# 途中で停止しても、再度このセルを実行すれば「未処理」のものから再開します。
db.process_queue(my_tokenizer)
```

## データの確認方法

```python
import sqlite3
import pandas as pd

with sqlite3.connect("my_corpus.db") as con:
    # 処理状況の確認
    df_status = pd.read_sql("SELECT * FROM status LIMIT 10", con)
    
    # 形態素解析結果の確認
    df_tokens = pd.read_sql("""
        SELECT t.doc_id, t.word, t.pos, d.rel_path 
        FROM tokens t
        JOIN documents d ON t.doc_id = d.doc_id
        LIMIT 20
    """, con)

print(df_status)
print(df_tokens)
```
