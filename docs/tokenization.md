# 形態素解析を行う関数（Janome / SudachiPy）

本ドキュメントでは、Pandas DataFrame に格納された日本語テキストを対象として、  
**Janome** または **SudachiPy** を用いた形態素解析を行い、  
「**1行 = 1トークン**」の縦持ち DataFrame に変換する関数群について説明します。

これらの関数は **ライブラリ化**されており、  
Google Colaboratory 上では本リポジトリを clone したうえで **import して使用**します。

いずれも **文書ID列を必須**とする設計であり、  
LDA / NMF / TF-IDF / WordCloud / 語頻度集計などの分析処理に  
そのまま接続できることを重視しています。

---

## Google Colab での使い方（重要）

### 0. どちらを使うか決める

本リポジトリでは、以下の **2 種類の形態素解析エンジン**を提供しています。

- **Janome 版**：軽量・導入が簡単（授業・演習向け）
- **SudachiPy 版**：高精度・高機能（研究用途向け）

👉 **通常はどちらか一方だけを使えば十分です。**  
Janome と SudachiPy を **同時にインストールする必要はありません。**

---

### 1-A. Janome を使う場合（おすすめ：授業・演習）

```python
!pip install janome
from libs import tokenize_df
```

---

### 1-B. SudachiPy を使う場合（研究用途・精度重視）

```python
!pip install sudachipy sudachidict_core
from libs import tokenize_df
```

---

※ **Google Colab では、ライブラリのインストールは必ず `!pip install ...` の形式で実行してください。**  
※ `pip install ...`（!なし）は Colab では動作しません。

---

# 1. まず全体像を理解する

## 1.1 なぜ「形態素解析」が必要なのか

日本語の文章は、英語のように単語ごとに空白で区切られていません。

例：
> 今日は良い天気でした。

この文をそのままコンピュータに渡しても、
- 「今日は」
- 「良い」
- 「天気」
- 「でした」

といった **意味のある単位（語）** を自動で切り出すことができません。

そこで必要になるのが **形態素解析** です。

形態素解析とは、
> 文を「これ以上分解すると意味を失う最小単位（形態素）」に分割し、  
> それぞれに品詞などの情報を付与する処理

です。

---

## 1.2 「表層（ひょうそう）」とは何か

形態素解析では、1つの語（トークン）に対して、複数の「語の表し方」があります。

例：
> 食べた

このとき、

- **表層形（surface）**：文章に実際に現れている文字列（この例では「食べた」）
- **基本形（原形 / 辞書形）**：辞書に載っている形（この例では「食べる」）

という2つを区別します。

- 「表層」は **文章の表面に見えている形**（そのままの文字列）
- 「基本形」は **活用などを元に戻した形**

です。

---

## 1.3 なぜ「縦持ち DataFrame」にするのか

本ライブラリでは、形態素解析の結果を次のような形で扱います。

| doc_id | word | pos |
|-----------|------|-----|
| 1 | 今日 | 名詞 |
| 1 | 良い | 形容詞 |
| 1 | 天気 | 名詞 |
| 1 | だ | 助動詞 |

このように  
**「1行 = 1トークン（語）」**  
という形式を「縦持ち」と呼びます。

### 縦持ちの利点

- 語の出現頻度（count）を数えやすい
- 品詞フィルタが簡単（`df[df["pos"]=="名詞"]` のように書ける）
- LDA / TF-IDF / WordCloud につながる前処理が作りやすい
- pandas の `groupby`, `value_counts` がそのまま使える

---

# 2. すすめ方：まず tokenize、あとでフィルタ

いきなり「名詞だけ」などに限定してしまうと、  
何が消えたのか／何が残ったのかが分かりにくくなります。

そこで、次の順序をおすすめします。

## Step 1：まずは品詞を限定せず tokenize する（全トークン）

```python
df_tok_all = tokenize_df(df)  # デフォルトは Janome
```

## Step 2：結果を観察する（頻出語・品詞）

```python
df_tok_all["pos"].value_counts().head(20)
df_tok_all["word"].value_counts().head(20)
```

## Step 3：あとから filter していく（段階的に落とす）

```python
from libs.preprocess import filter_tokens_df

# まずは記号や空白だけ落とす
df_tok_1 = filter_tokens_df(df_tok_all, pos_exclude={"補助記号", "空白"})

# 次に助詞も落とす（目的により）
df_tok_2 = filter_tokens_df(df_tok_1, pos_exclude={"助詞"})

# stopword も落とせる
top10words = df_tok_2["word"].value_count().head(10) # 出現頻度上位10単語をストップワードとして認定
df_tok_3 = filter_tokens_df(df_tok_2, stopwords=(top10words, "いる", "ある"))
```

---

# 3. 関数リファレンス（丁寧版）

ここからは、以下の4関数について

- **完全な引数一覧（網羅）**
- **各引数の意味**
- **よくある使い方のレシピ**
- **つまずきポイント**

をまとめます。

対象関数：
- `tokenize_df`
- `filter_tokens_df`
- `tokenize_text_janome`
- `tokenize_text_sudachi`

（※本説明は `preprocess.py` 清書版の仕様に合わせています。）

---

# 4. tokenize_df（入口関数）

## 4.1 何をする関数か

`tokenize_df` は **DataFrame（文書単位）** を受け取り、  
形態素解析して **DataFrame（トークン単位）** に変換します。

- 入力：1行=1文書（または1記事）の DataFrame
- 出力：1行=1トークンの DataFrame

---

## 4.2 シグネチャ（概念）

```python
tokenize_df(
    df,
    *,
    id_col="doc_id",
    text_col="text",
    engine="janome",
    tokenizer=None,
    tokenize_text_fn=None,
    use_base_form=True,
    pos_keep=None,
    pos_exclude=None,
    stopwords=None,
    extra_col="token_info",
) -> pandas.DataFrame
```

ポイント：
- **デフォルトは Janome**（導入が簡単で軽量）
- Sudachi 固有の `split_mode` は `tokenize_df` では扱いません  
  → 変更したい場合は `tokenize_text_fn` を使います（後述）

---

## 4.3 引数（完全網羅）

### df（必須）
- 型：`pandas.DataFrame`
- 意味：入力（文書単位）の表
- 必要な列：`id_col` と `text_col` が必ず存在すること

---

### id_col
- 型：`str`
- 既定：`"doc_id"`
- 意味：文書ID列名（例：記事ID、ファイルID、行ID）
- 例：`id_col="id"`

---

### text_col
- 型：`str`
- 既定：`"text"`
- 意味：解析対象の本文列名
- 例：`text_col="text"`

---

### engine
- 型：`str`
- 既定：`"janome"`
- 取りうる値：`"janome"` / `"sudachi"`
- 意味：どちらの形態素解析エンジンを使うか

---

### tokenizer
- 型：エンジン依存（Janome Tokenizer / Sudachi Tokenizer）
- 既定：`None`（関数内で自動生成）
- 意味：外で生成した tokenizer を渡したいときに指定

#### いつ使う？
- tokenize を何回も繰り返す（速度改善）
- `tokenize_text_fn` を作るときに、tokenizer をクロージャに閉じ込めたい

---

### tokenize_text_fn（拡張ポイント）
- 型：`callable` または `None`
- 既定：`None`

意味：
- 1テキスト（1文書）をトークナイズする処理を「丸ごと差し替える」ための引数。

**指定した場合は、原則として tokenize_text_fn が最優先です。**

#### tokenize_text_fn の仕様（必ず守る）

`tokenize_text_fn(text)` は次の形式を返します：

- 戻り値：`list[tuple[word, pos, token_info]]`

1トークンを表すタプル：
- `word`：`str`（下流の分析で使う語）
- `pos`：`str`（品詞の大分類）
- `token_info`：`dict` または `None`（追加情報。要らなければ None でOK）

例：
```python
[
  ("今日", "名詞", {"surface": "今日", "base_form": "今日"}),
  ("良い", "形容詞", {"surface": "良い", "base_form": "良い"}),
]
```

---

### use_base_form（共通オプション）
- 型：`bool`
- 既定：`True`
- 意味：「word として基本形を使うか／表層形を使うか」

| エンジン | True（既定） | False |
|---|---|---|
| Janome | 原形（base_form） | 表層形（surface） |
| Sudachi | 辞書形（dictionary_form） | 表層形（surface） |

---

### pos_keep
- 型：`iterable[str]` または `None`
- 既定：`None`（限定しない）
- 意味：指定した品詞（大分類）のトークンだけ残す
- 例：`pos_keep={"名詞","動詞","形容詞"}`

---

### pos_exclude
- 型：`iterable[str]` または `None`
- 既定：`None`（除外しない）
- 意味：指定した品詞（大分類）を除外
- 例：`pos_exclude={"助詞","補助記号","空白"}`

---

### stopwords
- 型：`iterable[str]` または `None`
- 既定：`None`
- 意味：word が stopwords に含まれるトークンを除外
- 例：`stopwords={"する","ある","なる"}`

---

### extra_col
- 型：`str` または `None`
- 既定：`"token_info"`
- 意味：
  - `None`：追加情報列を作らない（軽量化）
  - 文字列：その列名で token_info を格納

---

## 4.4 出力 DataFrame の仕様

tokenize_df の出力には次の列が含まれます。

- `id_col`（例：`doc_id`）
- `word`
- `pos`
- `token_info`（extra_col が None のときは作られない／または全て None）

---

# 5. filter_tokens_df（品詞フィルタ専用）

## 5.1 何をする関数か

`filter_tokens_df` は、`tokenize_df` が返す **トークン単位 DataFrame** を受け取り、  
主に次の3種類の条件でトークンを除外・抽出するための関数です。

- 品詞によるフィルタ（`pos_keep`, `pos_exclude`）
- ストップワードによる除外（`stopwords`）
- 指定の矛盾を検出するための安全チェック（`strict`）

この関数は **tokenize 後に段階的に適用すること**を前提に設計されています。

---

## 5.2 シグネチャ（概念）

```python
filter_tokens_df(
    df,
    *,
    pos_keep=None,
    pos_exclude=None,
    stopwords=None,
    strict=True,
) -> pandas.DataFrame
```

---

## 5.3 入力 DataFrame の前提

`df`（入力 DataFrame）は、最低限次の列を含んでいる必要があります。

- `word`：トークン文字列
- `pos`：品詞（大分類）

これらは `tokenize_df` の標準出力に含まれます。

---

## 5.4 pos_keep（残す品詞を指定）

- 型：`iterable[str]` または `None`
- 既定：`None`（すべて残す）

指定した場合、`df["pos"]` が **pos_keep に含まれる行だけ** が残ります。

```python
# 名詞だけ残す
df_noun = filter_tokens_df(df_tok, pos_keep={"名詞"})

# 名詞と形容詞を残す
df_noun_adj = filter_tokens_df(df_tok, pos_keep={"名詞", "形容詞"})
```

---

## 5.5 pos_exclude（除外する品詞を指定）

- 型：`iterable[str]` または `None`
- 既定：`None`（除外しない）

指定した場合、`df["pos"]` が **pos_exclude に含まれる行は除外**されます。

```python
# 助詞・助動詞を除外
df_no_particles = filter_tokens_df(
    df_tok,
    pos_exclude={"助詞", "助動詞"}
)

# 記号類を除外
df_no_symbols = filter_tokens_df(
    df_tok,
    pos_exclude={"補助記号", "記号"}
)
```

---

## 5.6 pos_keep と pos_exclude を同時に使う場合（strict）

- 型：`bool`
- 既定：`True`

`pos_keep` と `pos_exclude` を **同時に指定した場合の安全装置**です。

`strict=True` のとき：

- 両者の集合が **完全に無関係（共通要素がゼロ）** の場合、  
  意図しない指定の可能性が高いため `ValueError` を出します。

```python
# strict=True（デフォルト）
df_f = filter_tokens_df(
    df_tok,
    pos_keep={"名詞", "動詞"},
    pos_exclude={"助詞", "記号"},
)
```

このチェックを無効にしたい場合は `strict=False` を指定します。

```python
df_f = filter_tokens_df(
    df_tok,
    pos_keep={"名詞", "動詞"},
    pos_exclude={"助詞", "記号"},
    strict=False,
)
```

---

## 5.7 stopwords（ストップワードによる除外）

- 型：多様（下記参照）または `None`
- 既定：`None`

`stopwords` は **除外したい語の集合**として扱われます。  
`df["word"]` が stopwords に含まれる行は除外されます。

### 指定できる形式

`stopwords` には、次のような形式を **そのまま渡せます**。

- 単一の文字列  
  ```python
  stopwords="ある"
  ```
- 文字列のリスト / タプル / 集合  
  ```python
  stopwords=["ある", "いる"]
  ```
- pandas.Series（`value_counts()` の結果など）  
  → **index 部分**が stopwords として使われます
- pandas.Index
- 上記を入れ子にした構造  
  ```python
  stopwords=(["ある", "いる"], vc.head(10))
  ```

内部では `_normalize_stopwords` により自動的に正規化されます。

### 例

```python
# 頻出語上位をストップワードとして除外
vc_top10 = df_tok["word"].value_counts().head(10)

df_f = filter_tokens_df(
    df_tok,
    stopwords=vc_top10
)

# 手動指定 + 頻出語指定
df_f = filter_tokens_df(
    df_tok,
    stopwords=(vc_top10, ["ある", "いる"])
)
```

※ `"ある"` をそのまま渡しても、文字単位に分解されることはありません。

---

## 5.8 段階的にフィルタする例（推奨）

```python
# 1) まず全部 tokenize
df_tok_all = tokenize_df(df)

# 2) 記号・助詞を除外
df_tok_1 = filter_tokens_df(
    df_tok_all,
    pos_exclude={"補助記号", "空白", "助詞"}
)

# 3) 頻出語 + 手動 stopwords を除外
vc_top10 = df_tok_1["word"].value_counts().head(10)

df_tok_2 = filter_tokens_df(
    df_tok_1,
    stopwords=(vc_top10, ["ある", "いる"])
)
```

---

# 6. tokenize_text_janome（1テキスト用・Janome）

## 6.1 何をする関数か

Janome で 1つの文字列をトークナイズし、  
`(word, pos, token_info)` の配列を返します。

---

## 6.2 シグネチャ（概念）

```python
tokenize_text_janome(
    text,
    *,
    tokenizer,
    use_base_form=True,
    extra_col="token_info",
) -> list[tuple[word, pos, token_info]]
```

---

## 6.3 引数（完全網羅）

### text
- 型：`str`（None/空文字の場合もありえる）
- 挙動：None/空は `[]` を返す

### tokenizer（必須）
- 型：`janome.tokenizer.Tokenizer`

### use_base_form
- tokenize_df と同じ

### extra_col
- tokenize_df と同じ（None の場合 token_info を作らない）

---

# 7. tokenize_text_sudachi（1テキスト用・Sudachi）

## 7.1 何をする関数か

Sudachi で 1つの文字列をトークナイズし、  
`(word, pos, token_info)` の配列を返します。

---

## 7.2 シグネチャ（概念）

```python
tokenize_text_sudachi(
    text,
    *,
    tokenizer,
    split_mode="C",
    word_form=None,
    use_base_form=True,
    extra_col="token_info",
) -> list[tuple[word, pos, token_info]]
```

---

## 7.3 引数（完全網羅）

### tokenizer（必須）
- 型：Sudachi tokenizer（`dictionary.Dictionary().create()` の戻り値）

### split_mode（Sudachi 固有）
- 型：`str`
- 既定：`"C"`
- 取りうる値：`"A"`, `"B"`, `"C"`
- 意味：分割粒度（Aが細かい、Cが粗い…というイメージでOK）

### word_form（Sudachi 固有：語形指定）
- 型：`str` または `None`
- 既定：`None`
- 意味：最終的に word として採用する語形を指定
- 取りうる値：
  - None：use_base_form に従う（True→dictionary_form / False→surface）
  - `"dictionary"`：辞書形
  - `"surface"`：表層形
  - `"normalized"`：正規化形（表記ゆれ吸収に有用）

---

## 7.4 レシピ：Sudachi の正規化形を使う（word_form="normalized"）

```python
from sudachipy import dictionary
from libs import tokenize_df
from libs.preprocess import tokenize_text_sudachi

tok = dictionary.Dictionary().create()

df_tok = tokenize_df(
    df,
    tokenize_text_fn=lambda t: tokenize_text_sudachi(
        t,
        tokenizer=tok,
        split_mode="C",
        word_form="normalized",
        extra_col=None,
    ),
    extra_col=None,
)
```

---

## 7.5 レシピ：Sudachi の split_mode を変える（tokenize_text_fn を使う）

`split_mode` は `tokenize_df` では受け取りません。  
変えたい場合は、次のように `tokenize_text_fn` 経由で指定します。

```python
from sudachipy import dictionary
from libs import tokenize_df
from libs.preprocess import tokenize_text_sudachi

tok = dictionary.Dictionary().create()

df_tok_A = tokenize_df(
    df,
    tokenize_text_fn=lambda t: tokenize_text_sudachi(
        t, tokenizer=tok, split_mode="A", word_form=None, extra_col=None
    ),
    extra_col=None,
)

df_tok_C = tokenize_df(
    df,
    tokenize_text_fn=lambda t: tokenize_text_sudachi(
        t, tokenizer=tok, split_mode="C", word_form=None, extra_col=None
    ),
    extra_col=None,
)
```

---

# 8. tokenize_text_fn を自作する（MeCab など）

## 8.1 tokenize_text_fn の入力仕様

- 入力：`text`（1文書分の文字列）
- 可能性：None / 空文字が来ることがある
- 推奨：None/空なら `[]` を返す

---

## 8.2 tokenize_text_fn の出力仕様（最重要）

戻り値は **list** で、要素は **3要素タプル**です。

```python
[
  (word, pos, token_info),
  ...
]
```

- `word`：`str`
- `pos`：`str`（品詞の大分類）
- `token_info`：`dict` または `None`

---

## 8.3 MeCab の例（雛形）

```python
def tokenize_text_mecab(text):
    if text is None:
        return []
    s = str(text).strip()
    if s == "":
        return []

    records = []
    # ここで MeCab 解析し、各形態素について：
    #   word = ...
    #   pos  = ...
    #   token_info = {...} または None
    # を作って records.append((word, pos, token_info)) する
    return records

df_tok = tokenize_df(
    df,
    tokenize_text_fn=tokenize_text_mecab,
    extra_col=None,
)
```

---

# 9. ユーザー定義辞書

形態素解析では、既存の辞書に含まれていない専門用語・固有名詞・複合語などを  
**1 語として扱いたい** 場面がよくあります。そのために利用するのが  
**ユーザー定義辞書**です。

本ライブラリでは、ユーザー定義辞書の扱いについて次の方針を採っています。

- ユーザー定義辞書の指定や管理は **Janome / SudachiPy 側で行う**
- tokenize_df は **作成済みの tokenizer をそのまま使う**
- 追加設定が必要な場合は **tokenize_text_fn で明示的に制御する**

このため、ユーザー定義辞書を使う場合でも  
**preprocess.py を変更する必要はありません。**

---

## 9.1 基本的な考え方

tokenize_df は「入口関数」として設計されており、  
形態素解析エンジン固有の詳細設定は扱いません。

役割分担は次のとおりです。

- **辞書・詳細設定**  
  → Janome / SudachiPy で tokenizer を作成する段階で指定
- **DataFrame 全体のトークナイズ処理**  
  → tokenize_df が担当

したがって、

> tokenizer を自分で作って tokenize_df に渡せば、  
> その設定（ユーザー定義辞書を含む）がそのまま反映される

という構造になっています。

---

## 9.2 Janome でユーザー定義辞書を使う

Janome では、CSV 形式のユーザー辞書を指定して Tokenizer を作成できます。

```python
from janome.tokenizer import Tokenizer
from libs import tokenize_df

# 1. ユーザー辞書を指定して Tokenizer を作成
my_tokenizer = Tokenizer(
    udic_path="user_dictionary.csv",
    udic_enc="utf8"
)

# 2. 作成した tokenizer を tokenize_df に渡す
df_tokens = tokenize_df(
    df,
    engine="janome",
    tokenizer=my_tokenizer
)
```

---

## 9.3 SudachiPy でユーザー定義辞書を使う

SudachiPy では、ユーザー辞書や正規化設定を  
**設定ファイル（sudachi.json）でまとめて管理する**のが一般的です。

```python
from sudachipy import dictionary
from libs import tokenize_df

my_tokenizer = dictionary.Dictionary(
    config_path="path/to/sudachi.json"
).create()

df_tokens = tokenize_df(
    df,
    engine="sudachi",
    tokenizer=my_tokenizer
)
```

Sudachi のユーザー辞書の作成方法については、[公式ドキュメント（2025-12-26時点）](Sudachi_user_dict.md) を参照しましょう。

---

## 9.4 tokenize_text_fn を使って制御する（Sudachi）

```python
from sudachipy import dictionary
from libs import tokenize_df, tokenize_text_sudachi

tokenizer = dictionary.Dictionary(
    config_path="path/to/sudachi.json"
).create()

def my_tokenize_fn(text):
    return tokenize_text_sudachi(
        text,
        tokenizer=tokenizer,
        split_mode="B",
        word_form="normalized"
    )

df_tokens = tokenize_df(
    df,
    tokenize_text_fn=my_tokenize_fn
)
```

---

## 9.5 まとめ

- ユーザー定義辞書は tokenizer 作成時に指定する
- tokenize_df は tokenizer を差し替えるだけで対応できる
- split_mode や word_form を変えたい場合は tokenize_text_fn を使う
- preprocess.py を変更する必要はない

---

# 10. 全体のまとめ

- `tokenize_df` は入口（デフォルトは Janome）
- まず全部 tokenize → 結果を観察 → `filter_tokens_df` で段階的に落とす
- Sudachi 固有の調整（`split_mode`, `word_form`）は `tokenize_text_fn` 経由で行う
- 別エンジンを使う場合は、`tokenize_text_fn` が返す形式（list of 3-tuples）を守る
