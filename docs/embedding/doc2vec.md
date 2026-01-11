# Doc2Vec による文書のベクトル化

本ドキュメントでは、**Doc2Vec** を用いて「文書全体」をベクトル化する方法を解説します。
記事の類似度判定や、レコメンドシステム（「この記事を読んだ人はこちらも...」）に応用できます。

---

## 1. 準備

Gensim は Google Colab にプリインストールされていますが、最新版を利用するためにアップグレードが必要な場合があります。

```python
!pip install --upgrade gensim

```

必要なライブラリをインポートします。

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from colab_nlp import corpus_reader

```

---

## 2. データの準備（イテレータの定義）

Doc2Vec の学習には、「単語リスト」と「文書ID」を持つ `TaggedDocument` が必要です。
また、学習プロセスでデータを複数回読み込むため、**再利用可能なイテレータクラス**を定義します。

```python
class TaggedCorpus:
    """
    CorpusDB から (doc_id, tokens) を読み込み、
    Gensim 用の TaggedDocument に変換して返すイテレータ
    """
    def __init__(self, db_path):
        self.db_path = db_path

    def __iter__(self):
        # corpus_reader を呼び出す
        # return_mode="id_tokens" で ID付きのデータを取得
        reader = corpus_reader(self.db_path, return_mode="id_tokens", chunk_size=1000)
        
        for doc_id, tokens in reader:
            yield TaggedDocument(words=tokens, tags=[doc_id])

# インスタンス化（データはまだ読み込まれません）
db_path = "corpus.db"
documents = TaggedCorpus(db_path)

```

---

## 3. モデルの学習

大規模データに対応するため、辞書作成と学習を分けて実行します。これにより、正確な進捗バーが表示されます。

```python
# 1. モデルの初期設定
model = Doc2Vec(
    vector_size=100,        # ベクトルの次元数
    window=5,               # 文脈の広さ
    min_count=5,            # 低頻度語のカット
    workers=4,              # 並列処理数
    epochs=10               # 学習回数（Word2Vecより多めが推奨されます）
)

# 2. 辞書（Vocabulary）の構築
print("辞書を作成中...")
model.build_vocab(documents)

print(f"学習対象の文書数: {model.corpus_count}")

# 3. 学習の実行
print("学習開始...")
model.train(
    documents,
    total_examples=model.corpus_count,
    epochs=model.epochs
)

print("学習完了")

```

---

## 4. モデルの保存と読み込み

```python
# 保存
model.save("doc2vec.model")

# 読み込み
# loaded_model = Doc2Vec.load("doc2vec.model")

```

---

## 5. モデルの活用

### 5-1. 類似文書の検索

ある文書（`doc_id`）に内容が似ている文書を探します。

```python
# ターゲットとなる文書ID（例: 100番の記事）
# ※ corpus.db に存在する doc_id を指定してください
target_doc_id = 100

try:
    # 似ている文書トップ10を取得
    # docvecs は古いため dv を使用（Gensim 4.0以降）
    sims = model.dv.most_similar(target_doc_id, topn=10)

    print(f"ID: {target_doc_id} に似ている記事:")
    for doc_id, score in sims:
        print(f"- ID: {doc_id} (類似度: {score:.3f})")

except KeyError:
    print(f"ID: {target_doc_id} は学習データに含まれていません。")

```

### 5-2. 未知の文書のベクトル化

学習に使っていない新しいテキストをベクトル化し、それに似ている過去記事を探すこともできます。
**注意:** トークン化には学習時と同じ辞書・設定を使う必要があります。

```python
from colab_nlp import tokenize_text_janome, tokenize_text_sudachi

# 1. 新しいテキストをトークン化（tokenize_df と同じエンジンを使うこと！）
text = "最近の人工知能技術の進歩は目覚ましいものがあります。"

# Janomeの場合
tokens = [w for w, p, i in tokenize_text_janome(text)]
# Sudachiの場合
# tokens = [w for w, p, i in tokenize_text_sudachi(text, mode="C")]

# 2. ベクトルを推論（infer_vector）
# 安定させるため、epochs を指定して推論します
new_vector = model.infer_vector(tokens, epochs=20)

# 3. 似ている記事を探す
sims = model.dv.most_similar([new_vector], topn=5)

print("入力テキストに似ている記事:")
for doc_id, score in sims:
    print(f"- ID: {doc_id} (類似度: {score:.3f})")

```
