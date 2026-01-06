# 共起ネットワーク分析

本ドキュメントでは、N-gram 分析の結果を **共起ネットワーク（Co-occurrence Network）** として可視化する方法を解説します。

単語の出現頻度（表）を見るだけでは分からない、**「単語と単語のつながり（文脈）」** を直感的に把握することができます。

---

## 共起ネットワークとは

文章中で **「一緒に使われることが多い単語のペア」** を線で結んだ図です。

- **ノード（丸）**: 単語
- **エッジ（線）**: 共起関係（一緒に使われたペア）
    - 線の **太さ**: 一緒に使われた回数（多いほど太い）
    - ノードの **大きさ**: つながっている相手の数（多いほど大きい＝ハブになる単語）

### 何が分かるか？

1.  **話題のまとまり（クラスタ）**:
    * 密につながっている単語のグループは、一つの「トピック」や「文脈」を表します。
    * 例：「人工」-「知能」-「技術」の塊、「美味しい」-「料理」-「店」の塊など。
2.  **中心的な単語**:
    * 多くの単語とつながっている（ノードが大きい）単語は、その文書集合における中心的なテーマや、一般的な動詞（「する」「ある」など）であることが多いです。

---

## 実装手順

本リポジトリで提供している `create_network_graph` 関数を使用します。
入力データは [`compute_ngram`](./ngram.md) で作成した DataFrame をそのまま使います。

### 1. 準備：日本語フォント

グラフ内の日本語を表示するために、フォントパスが必要です。
（`colab-common` のテンプレートで設定済みの `font_path` 変数を使用してください）

```python
# もし font_path が未定義なら、テンプレートを実行して取得してください
print(font_path)
# 出力例: /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

### 2. データ作成（N-gram）

```python
from colab_nlp import compute_ngram

# Bigram (2-gram) を作成
# ネットワーク図にする場合、あまりに低頻度なペアはノイズになるため min_count で足切りします
df_bigram = compute_ngram(tokens, n=2, min_count=5)
```

### 3. 可視化実行

```python
from colab_nlp import create_network_graph

create_network_graph(
    df_bigram,
    head=50,             # 上位50ペアを描画（多すぎると見づらいため）
    font_path=font_path, # 必須
    save_path="network.png"
)
```

---

## 分析のコツ

### 上手く描画されないときは？

1. **「真っ黒」になる**:
* `head` の値が大きすぎます（例: 500など）。
* まずは `head=30` 〜 `50` くらいから始めて、徐々に増やしてみましょう。


2. **「□□□」と文字化けする**:
* `font_path` が正しく渡されていません。
* パスが存在するか、`create_network_graph` の引数に指定しているか確認してください。

### 品詞でフィルタリングする

`tokens` の段階で品詞を絞り込むと、より意味のあるネットワークになります。

* **名詞のみ**: トピック（話題）の構造が見えやすくなります。
* **名詞 + 動詞**: 「誰が」「何をした」という動作の関係性が見えます。
* **形容詞**: 「面白い」「難しい」などが何にかかっているか（評価の対象）が見えます。

```python
# 例：名詞だけでネットワークを作る
from colab_nlp import filter_tokens_df

df_noun = filter_tokens_df(tokens, pos_keep={"名詞"})
df_noun_bigram = compute_ngram(df_noun, n=2, min_count=3)

create_network_graph(df_noun_bigram, head=40, font_path=font_path)
```
