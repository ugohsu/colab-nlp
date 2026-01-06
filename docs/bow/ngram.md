# N-gram（Nグラム）

本ドキュメントでは、形態素解析済みの **tokens DataFrame** から、**N-gram（隣り合う N 個の単語の連なり）** を抽出し、その出現頻度を集計する方法を解説します。

---

## N-gram とは

[語頻度（Term Frequency）](./term_frequency.md) では、文章を「単語の袋（Bag of Words）」として扱い、語順を無視して集計しました。

しかし、言葉の意味は **単語の並び** に宿ることがあります。

- 「人工」+「知能」 → **「人工知能」**（複合語）
- 「機械」+「学習」 → **「機械学習」**（複合語）
- 「て」+「いる」 → **「ている」**（補助動詞としての用法）

このように、隣り合う $N$ 個の単語をひとまとめにして扱う手法を **N-gram** と呼びます。

- **Unigram (N=1)**: 通常の単語（Term Frequency と同じ）
- **Bigram (N=2)**: 隣り合う2語のペア
- **Trigram (N=3)**: 隣り合う3語の連なり

---

## 1. Python (リスト操作) で仕組みを学ぶ

ライブラリの便利な関数を使う前に、Python の標準機能だけで N-gram を作る仕組みを見てみましょう。これが理解できれば、ブラックボックスにならずに済みます。

N-gram を作る基本的なアイデアは、**「元のリスト」と「1つずらしたリスト」を同時にループする** ことです。

### 1-A. リスト操作による Bigram 生成

```python
# サンプル文（分かち書き済みリスト）
words = ["今日", "は", "良い", "天気", "です"]

# zipを使って、元のリストと「1つずらしたリスト」を組み合わせる
# words[0:] : ['今日', 'は', '良い', '天気', 'です']
# words[1:] : ['は', '良い', '天気', 'です']

bigrams = list(zip(words, words[1:]))

print(bigrams)
```

出力結果:

```
[('今日', 'は'), ('は', '良い'), ('良い', '天気'), ('天気', 'です')]
```

このように、`zip` とスライス（`[1:]`）を使うだけで、簡単に隣り合うペアを作ることができます。

### 1-B. Pandas DataFrame への適用

これを tokens DataFrame 全体に適用する場合、「文書ごと」にこの処理を行います。

```python
import pandas as pd

# 1. 文書ごとに単語リスト化
docs_list = tokens.groupby("doc_id")["word"].apply(list)

# 2. 各文書に対して N-gram 生成処理を適用
def make_bigrams(words):
    return list(zip(words, words[1:]))

# 実際に適用してみる（最初の5文書）
docs_bigrams = docs_list.apply(make_bigrams)
docs_bigrams.head()
```

---

## 2. scikit-learn を直接使って集計する

実務では、速度と効率のために **scikit-learn** の `CountVectorizer` を使うのが一般的です。

ただし、通常の `CountVectorizer` は「スペース区切りの文字列」を入力として想定しています。我々のデータは「トークンのリスト（形態素解析済み）」なので、**リストをそのまま渡せるように設定を少し工夫** して使います。

### 実装コード

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# 1. 前処理：文書ごとにトークンをリストにまとめる
#    欠損（NaN）があるとエラーになるため、事前に除去します
tokens_clean = tokens.dropna(subset=["word"])
docs_list = tokens_clean.groupby("doc_id")["word"].apply(list)

# 2. CountVectorizer の設定（ここがポイント）
#    リストをそのまま受け取り、N-gram生成機能だけを利用するための設定です
def identity(x):
    return x

cv = CountVectorizer(
    preprocessor=identity,  # 前処理（小文字化など）をしない
    tokenizer=identity,     # トークナイズ（分割）をしない
    token_pattern=None,     # デフォルトの正規表現フィルタを無効化
    ngram_range=(2, 2),     # (2, 2) = Bigramのみ
)

# 3. 集計実行
X = cv.fit_transform(docs_list)

# 4. 結果の整形
vocab = cv.get_feature_names_out()
counts = X.sum(axis=0).A1  # 列ごとの合計

df_ngram = pd.DataFrame({"ngram": vocab, "count": counts})
df_ngram = df_ngram.sort_values("count", ascending=False).reset_index(drop=True)

df_ngram.head()
```

**出力イメージ**

| ngram | count |
| --- | --- |
| 人工 知能 | 150 |
| 機械 学習 | 120 |
| 自然 言語 | 90 |
| ... | ... |

このように、「何もしない関数（identity）」を挟むことで、scikit-learn の強力な集計機能をリストデータに対して適用できます。

※ 語頻度 (`term_frequency`) では `analyzer=lambda x: x` を使いましたが、`analyzer` を指定すると `N-gram` 生成機能までスキップされてしまう ため、ここでは `tokenizer` 等を個別に無効化する手法をとっています。

---

## 3. ラッパー関数を使う（compute_ngram）

前節の処理（前処理、設定、集計、整形）を毎回書くのは大変です。 本リポジトリでは、これらをまとめた関数 `compute_ngram` を提供しています。

#### 基本的な使い方

DataFrame を渡すだけで、N-gram 頻度を集計できます。

```python
from colab_nlp import compute_ngram

# Bigram (2-gram) を集計
df_bigram = compute_ngram(tokens, n=2, min_count=2)
```

* `n=3` にすれば Trigram が抽出されます。
* `min_count` で足切り（低頻度語の除外）ができます。
* `top_k=100` を指定すると、上位100件だけを返します。

---

## 4. 【応用】大規模データへの対応（CorpusDB）

数万〜数十万件の文書がある場合、すべてのデータを一度に DataFrame に読み込むとメモリ不足になる恐れがあります。

`compute_ngram` 関数は、DataFrame の代わりに **「データを少しずつ読み出すジェネレータ」** を受け取ることができます。

本リポジトリでは、`CorpusDB` で生成したデータ (SQLite データベース) からデータを少しずつ読み出すための補助関数 **`corpus_reader`** を提供しています。これらを組み合わせることで、省メモリな N-gram 分析が可能です。

### 実装例

```python
from colab_nlp import compute_ngram, corpus_reader

# データベースのパス
db_path = "corpus.db"

# 1. ジェネレータの作成
#    DBから 1000件ずつ 読み込み、トークンリストを小出しにする準備をします
reader = corpus_reader(db_path, chunk_size=1000)

# 2. N-gram 集計の実行
#    DataFrame の代わりに reader を渡します
df_bigram_large = compute_ngram(
    reader, 
    n=2, 
    min_count=5
)
```

この方法を使えば、Colab のメモリに載りきらないような大量のテキストデータに対しても、安全に分析を行うことができます。



先生、ありがとうございます！質が高いと評価いただき、大変光栄です。

「付録: 詳細リファレンス」の追加ですね。承知いたしました。
`tokenization.md` や `corpus_db.md` と同様の形式で、引数の意味やデフォルト値を網羅的に記述しましょう。

以下に追記用の Markdown コンテンツを提示します。これを `ngram.md` の末尾に追加してください。

---

## 付録: 詳細リファレンス

### compute_ngram

N-gram 頻度を集計するメイン関数です。入力として「DataFrame」と「ジェネレータ（イテレータ）」の両方を受け付ける点が特徴です。

```python
compute_ngram(
    data,
    *,
    n=2,
    min_count=1,
    col_doc="doc_id",
    col_word="word",
    top_k=None,
) -> pandas.DataFrame

```

#### 引数

* **`data`** (必須)
* 型: `pandas.DataFrame` または `Iterable[list[str]]`
* **DataFrame の場合**: 形態素解析済みの縦持ちデータ（`col_doc` 列と `col_word` 列が必要）。
* **Iterable の場合**: 文書ごとのトークンリストを返すジェネレータ（`corpus_reader` の戻り値など）。


* **`n`**
* 型: `int`
* 既定: `2`
* N-gram の N を指定します（`1`=Unigram, `2`=Bigram, `3`=Trigram）。


* **`min_count`**
* 型: `int`
* 既定: `1`
* 出現回数がこの値未満の N-gram を除外します。ノイズ除去に有用です。


* **`col_doc`**
* 型: `str`
* 既定: `"doc_id"`
* `data` が DataFrame の場合の「文書ID列名」を指定します。


* **`col_word`**
* 型: `str`
* 既定: `"word"`
* `data` が DataFrame の場合の「単語列名」を指定します。


* **`top_k`**
* 型: `int` または `None`
* 既定: `None` (全件)
* 指定した場合、頻度上位 `k` 件のみを返します。

#### 戻り値

* **`pandas.DataFrame`**
* `ngram`: N-gram 文字列（スペース区切り。例: `"人工 知能"`）
* `count`: 出現回数
* 頻度の降順でソート済みです。
