# 語頻度（Term Frequency）

本ドキュメントでは、形態素解析済みの **tokens DataFrame**  
（「1行 = 1トークン」の縦持ち形式）から、  
**語頻度（単語 × 出現回数）** を求める「最小構成」を示します。

---

## 前提：入力（tokens DataFrame）

入力は、すでに形態素解析が完了した DataFrame です。  
最低限、次の列を含んでいる必要があります。

- `doc_id`：文書ID（同じ文書に属するトークンをまとめるために使う）
- `word`：トークン（頻度を数える対象）
- `pos`：品詞（大分類。名詞だけ数える、などのフィルタに使う）

形態素解析（tokens の作成方法）は、以下を参照してください。

- 形態素解析（Janome / SudachiPy）：[`../tokenization.md`](../tokenization.md)  
- BoW 章トップ：[`README.md`](README.md)

---

## 0. 準備（最小）

```python
import pandas as pd
```

### 説明
- `pandas` は DataFrame 操作の標準ライブラリです。
- このドキュメントでは、語頻度集計を **pandas / scikit-learn** の両方で行いますが、  
  どちらの結果も DataFrame / Series で扱うのが自然なので、最初に import しておきます。

---

## 1. pandas で語頻度（最小）

### 1-A. 全体の語頻度（単語 × 回数）

```python
freq = tokens["word"].value_counts()
freq.head()
```

#### 行ごとの説明

**`tokens["word"]`**
- `tokens` の `word` 列だけを取り出しています（Series になります）。
- 語頻度は「単語ごとの回数」なので、まずは “数える対象” の列に絞ります。

**`.value_counts()`（必須）**
- Series に含まれる値を「値ごと」に数え上げ、頻度順（多い順）に並べた結果を返します。
- ここでは “単語ごとの出現回数” を作る最小の一手です。

**`freq.head()`（表示用）**
- 上位だけを表示するためのメソッドです。
- 集計そのものには不要なので、**省略しても動きます**。
- ただし Colab で結果確認をするなら、`head()` を付けておくと見やすいです。

#### 出力の型（重要）
- `freq` は **pandas.Series** です。
  - index：単語（word）
  - value：出現回数

---

### 1-B. 品詞で絞った語頻度（最小）

例：名詞のみ

```python
freq_noun = tokens.loc[tokens["pos"] == "名詞", "word"].value_counts()
freq_noun.head()
```

#### 行ごとの説明

**`tokens["pos"] == "名詞"`**
- `pos` 列が `"名詞"` の行だけを True にする条件です（真偽値の配列になります）。

**`tokens.loc[条件, "word"]`（必須）**
- `loc` は “行と列” を指定して取り出すための基本機能です。
- ここでは
  - 行：`pos == "名詞"` の行だけ
  - 列：`"word"` だけ
  を取り出しています。

**`.value_counts()`**
- 絞り込み後の `word` だけに対して頻度を数えます。

---

### 1-C. DataFrame として保持（最小）

後続処理（保存、結合、可視化など）を考える場合、DataFrame にします。

```python
freq_df = tokens["word"].value_counts().reset_index()
freq_df.columns = ["word", "count"]
freq_df.head()
```

#### 行ごとの説明

**`tokens["word"].value_counts()`**
- 1-A と同じく、語頻度の Series を作っています。

**`.reset_index()`（よく使う）**
- `value_counts()` の結果は Series（index に単語が入る）です。
- `reset_index()` をすると、index を通常の列として戻し、**2列の DataFrame** になります。
  - 1列目：単語
  - 2列目：回数

**`freq_df.columns = ["word", "count"]`（最小限の整形）**
- `reset_index()` 後の列名は分かりにくい場合があるため、ここで明示します。
- これは見栄えのためというより、**後で参照しやすくするための最小限**です。

---

## 2. scikit-learn で語頻度（最小・推奨）

ここでは、**sklearn に形態素解析はさせません**。  
tokens DataFrame から、

- 文書ごとにトークンの **list**
- それをそのまま `CountVectorizer` に渡す

という、最も素直な方法を用います。

> 重要：`CountVectorizer` は本来 “文字列（文書）” を受け取りますが、  
> 設定を工夫すると「事前トークン化済み入力（list）」をそのまま扱えます。

---

### 2-A. 文書ごとに「トークンの list」を作る

```python
docs_tokens = (
    tokens
    .groupby("doc_id")["word"]
    .apply(list)
)
```

#### 行ごとの説明

**`tokens.groupby("doc_id")`（必須）**
- `doc_id` が同じ行（＝同じ文書）をひとまとめにする操作です。
- これにより、「文書ごとのトークン列」を作れます。

**`["word"]`**
- グループ化したうえで、`word` 列だけを対象にします（数える対象は単語なので）。

**`.apply(list)`（必須）**
- 各文書の `word` を、Python の `list` に変換します。
- 結果は **「文書ID → トークン列（list）」** という対応になります。

#### 出力の型（重要）
- `docs_tokens` は **pandas.Series** です。
  - index：`doc_id`
  - value：`list[str]`（トークンのリスト）

---

### 2-B. CountVectorizer に直接渡す

```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(
    tokenizer=lambda x: x,
    preprocessor=lambda x: x,
    token_pattern=None,
)

X = cv.fit_transform(docs_tokens)
vocab = cv.get_feature_names_out()
```

#### 行ごとの説明

**`from ... import CountVectorizer`**
- BoW（文書×単語のカウント行列）を作るための標準クラスです。

---

#### `CountVectorizer(...)` の3つの必須設定

**`tokenizer=lambda x: x`（必須）**
- `tokenizer` は本来 “文字列を単語に分割する関数” を指定する場所です。
- しかし今回はすでに `x` が「トークンの list」なので、  
  **何もせずそのまま返す**（恒等関数）にします。

**`preprocessor=lambda x: x`（必須）**
- `preprocessor` は本来 “文字列の前処理（小文字化など）” をする場所です。
- 今回は `x` が list なので、ここも **何もせずそのまま返す**にします。

**`token_pattern=None`（必須）**
- `token_pattern` は “文字列からトークンを拾う正規表現” の設定です。
- 既定値のままだと、sklearn が「文字列処理モード」だと解釈しようとして混乱します。
- そこで `None` にして、**正規表現トークナイズを無効化**します。

---

#### 行列の作成

**`X = cv.fit_transform(docs_tokens)`（必須）**
- `fit_transform` は
  1. 語彙（vocabulary）を学習（fit）
  2. 文書×単語のカウント行列を作成（transform）
  を同時に行います。
- `X` は巨大なことが多いので、効率の良い **疎行列（sparse matrix）** になります。

**`vocab = cv.get_feature_names_out()`**
- 学習した語彙（単語の一覧）を取り出します。
- `X` の列（特徴量）がどの単語に対応するかを知るために必要です。

---

### 2-C. 全体の語頻度を求める

```python
word_counts = X.sum(axis=0)
freq_sklearn = pd.Series(word_counts.A1, index=vocab)
freq_sklearn = freq_sklearn.sort_values(ascending=False)
freq_sklearn.head()
```

#### 行ごとの説明

**`X.sum(axis=0)`（必須）**
- `X` は「文書×単語」の行列です。
- `axis=0` は “列方向に合計” を意味します。
- つまり、全ての文書について合計し、**単語ごとの総出現回数**を得ます。

**`word_counts.A1`**
- `word_counts` は行列型（numpy.matrix のような形）で返ることがあります。
- `.A1` は “1次元の numpy 配列に変換” するための簡便な属性です。
  - ここで 1次元配列にしておくと、pandas に渡しやすくなります。

**`pd.Series(..., index=vocab)`**
- 「回数の配列」と「単語の一覧」を結びつけて Series にします。
- これで
  - index：単語
  - value：回数
  という、pandas の `value_counts()` と同じ見た目の結果になります。

**`.sort_values(ascending=False)`**
- 合計の結果は必ずしも頻度順ではないので、降順に並べ替えます。
- これは “結果確認” として重要なので入れています（最小限）。

**`.head()`**
- 表示用です（省略可）。

---

## 注意（最小）

### tokens に欠損がある場合

```python
tokens = tokens.dropna(subset=["word"])
```

#### 説明
- `word` が欠損（NaN）だと、`value_counts()` や join が不安定になります。
- `dropna(subset=[...])` は、指定した列だけ欠損の行を落とす最小の対処です。

---

## まとめ：pandas と sklearn の使い分け

- **pandas** は “全体の語頻度を素早く見る” のに向いています  
  - `value_counts()` が非常に簡潔

- **scikit-learn** は “文書×単語行列（BoW 行列）を作って次の分析へ進む” のに向いています  
  - ただし本教材では、**tokens を出発点**にして sklearn の前処理を最小化します
