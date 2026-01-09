# Word2Vec による単語の分散表現

本ドキュメントでは、**Gensim** ライブラリと本リポジトリの `corpus_reader` を組み合わせて、大規模なテキストデータから **Word2Vec** モデルを学習する方法を解説します。

---

## 1. 準備

Gensim は Google Colab にプリインストールされていますが、最新版を利用するためにアップグレードが必要な場合があります。

```python
!pip install --upgrade gensim
```

必要なライブラリをインポートします。

```python
from gensim.models import Word2Vec
from colab_nlp import corpus_reader
```

---

## 2. データの準備（ストリーム読み込み）

`Word2Vec` の学習には、大量のテキストデータが必要です。
本リポジトリの `CorpusDB` で構築したデータベース (`corpus.db`) を使用します。

`corpus_reader` を使うことで、データを全てメモリに読み込むことなく、**ディスクから少しずつ読み出しながら（ストリーミング）** 学習を行うことができます。

```python
# データベースのパス
db_path = "corpus.db"  # または Google Drive 上のパス

# ジェネレータの作成
# メモリには常に chunk_size (1000文書) 分しか展開されません
sentences = corpus_reader(db_path, chunk_size=1000)
```

---

## 3. モデルの学習

`sentences`（ジェネレータ）をそのまま `Word2Vec` に渡すだけで学習が始まります。

```python
# Word2Vec モデルの学習
model = Word2Vec(
    sentences=sentences,  # ジェネレータを直接渡す
    vector_size=100,      # ベクトルの次元数
    window=5,             # 前後の単語を見る範囲
    min_count=5,          # 5回未満の低頻度語は無視
    workers=4,            # 並列処理数
    epochs=5              # 繰り返し回数
)

print("学習完了")
```

> **Note:** データ量が多い場合、学習には時間がかかります。Colab のランタイムが切れないように注意してください。

---

## 4. モデルの保存と読み込み

学習したモデルはファイルとして保存し、後で再利用できます。

```python
# 保存
model.save("word2vec.model")

# 読み込み（新しいセッションで使う場合）
# loaded_model = Word2Vec.load("word2vec.model")
```

---

## 5. モデルの活用（類似語検索）

学習したモデルを使って、単語の意味的な近さを計算できます。

### 類似語の検索

ある単語と意味が似ている（同じような文脈で使われる）単語を探します。

```python
# 「猫」に似ている単語トップ10
similar_words = model.wv.most_similar("猫", topn=10)

for word, score in similar_words:
    print(f"{word}: {score:.3f}")
```

### 単語の演算

単語ベクトルの足し算・引き算を行います（例：王様 - 男 + 女 = 女王）。

```python
# ポジティブな要素 - ネガティブな要素
result = model.wv.most_similar(positive=["王", "女"], negative=["男"], topn=5)

for word, score in result:
    print(f"{word}: {score:.3f}")
```

### ベクトルの取得

機械学習の入力特徴量として使うために、単語ベクトルそのものを取得します。

```python
# "猫" の 100次元ベクトルを取得
vector = model.wv["猫"]
print(vector.shape)  # (100,)
```

---

## まとめ

* `corpus_reader` を使うことで、メモリ制限を気にせず大規模データの学習が可能になります。
* 学習したモデルは保存しておくことで、後の分析やアプリケーションで再利用できます。