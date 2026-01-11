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

## 2. データの準備（イテレータの準備）

Word2Vec の学習には、大量のテキストデータが必要です。
本リポジトリの `CorpusDB` で構築したデータベース (`corpus.db`) を使用します。

`corpus_reader` を使うことで、データを全てメモリに読み込むことなく、**ディスクから少しずつ読み出しながら（ストリーミング）** 学習を行うことができます。

```python
# データベースのパス
db_path = "corpus.db"  # または Google Drive 上のパス

# イテレータ（反復可能オブジェクト）の準備
# メモリには常に chunk_size (1000文書) 分しか展開されません
sentences = corpus_reader(db_path, chunk_size=1000)

# 【重要】
# corpus_reader は「1回ループしたら終わりのジェネレータ」ではなく、
# 何度でも先頭からループできる「イテレータ（Iterable）」を返します。
# Gensim は学習時にデータを複数回スキャンするため、この性質が必要です。

```

---

## 3. モデルの学習

大規模データを扱う場合、**「ボキャブラリ構築」と「学習」を分けて実行**するのがベストプラクティスです。
これにより、正確な進捗バーを表示させたり、学習前にデータ規模を確認したりできます。

### 3-1. 学習の実行コード

```python
# 1. モデルの初期設定（データはまだ渡さない）
model = Word2Vec(
    vector_size=100,      # ベクトルの次元数
    window=5,             # 前後の単語を見る範囲
    min_count=5,          # 5回未満の低頻度語は無視
    workers=4,            # 並列処理数
    epochs=5              # 繰り返し回数
)

# 2. ボキャブラリ（辞書）の構築
# データを一度スキャンして、単語の種類と頻度をカウントします
print("辞書を作成中...")
model.build_vocab(sentences)

print(f"学習対象の単語数: {len(model.wv.index_to_key)}")
print(f"学習対象の総文書数: {model.corpus_count}")

# 3. 学習の実行
print("学習開始...")
model.train(
    sentences,
    total_examples=model.corpus_count,
    epochs=model.epochs
)

print("学習完了")

```

> **Note:** データ量が多い場合、学習には時間がかかります。Colab のランタイムが切れないように注意してください。

### 3-2. `model.train` の引数解説

なぜ `model.train` にこれらの引数を渡す必要があるのか、詳しく解説します。

* **`sentences`**
* 学習に使用するデータ（イテレータ）です。
* Gensim はこのイテレータを通じてデータをディスクから少しずつ読み込み、学習を行います。


* **`total_examples=model.corpus_count`**
* **ここが重要です。** イテレータ形式のデータは、リストと違って「全部で何件あるか（`len()`）」を瞬時に知ることができません。
* そのため、この引数を省略すると、Gensim は「全体の進捗率」や「残り時間」を計算できず、プログレスバーが正しく表示されません。
* 直前の `model.build_vocab(sentences)` で全データを一度スキャンした際、Gensim は総文書数をカウントし、自動的に `model.corpus_count` に保存してくれています。それをここで渡すことで、正確な進捗表示が可能になります。


* **`epochs=model.epochs`**
* 学習データを何周するか（エポック数）を指定します。
* モデルの初期化時（`Word2Vec(...)`）に `epochs=5` と指定しましたが、`train` メソッドを呼ぶ際にも明示的に渡すのが推奨されています。
* `model.epochs` には初期化時に設定した値（ここでは 5）が入っています。



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
* **ボキャブラリ構築 (`build_vocab`)** と **学習 (`train`)** を分けることで、正確な進捗管理が可能になります。
* イテレータを使う際は、`total_examples` を明示的に指定することで、学習の残り時間が把握しやすくなります。

