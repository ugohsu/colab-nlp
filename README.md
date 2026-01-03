# Colab NLP Templates

このリポジトリは、**Google Colaboratory 上で自然言語処理（NLP）を学ぶための関数・解説ドキュメント**をまとめた教材用リポジトリです。

---

## 関数一覧（import して使う）

| 分類 | 内容 | 実装ファイル | 解説ドキュメント |
|---|---|---|---|
| I/O | フォルダ配下のテキスト→DataFrame（build_text_df） | [`colab_nlp/io_text.py`](./colab_nlp/io_text.py) | [`docs/io_text_basic.md`](./docs/io_text_basic.md) |
| I/O | Google Sheets 書き込み | [`colab_nlp/gsheet_io.py`](./colab_nlp/gsheet_io.py) | [`docs/write_google_spreadsheet.md`](./docs/write_google_spreadsheet.md) |
| 前処理 | 形態素解析（Janome / SudachiPy：tokenize_df） | [`colab_nlp/preprocess.py`](./colab_nlp/preprocess.py) | [`docs/tokenization.md`](./docs/tokenization.md) |
| BoW / 可視化 | 語頻度・WordCloud | [`colab_nlp/bow.py`](./colab_nlp/bow.py) | [`docs/bow/wordcloud.md`](./docs/bow/wordcloud.md) |

## ドキュメント一覧

BOW/README.md も忘れないように

---

## 関数群（colab_nlp）の基本的な使い方

以下のコードを実行することで、本リポジトリで提供されている関数をインポートできるようになります。

```python
!git clone https://github.com/ugohsu/colab-nlp-templates.git

import sys
sys.path.append("/content/colab-nlp-templates")
```

> ⚠️ **注意（Google Colab での git clone）**  
> 上記の `git clone` は **最初の1回だけ実行してください**。  
> 同じノートブックで 2 回以上実行すると、
>
> ```
> fatal: destination path 'colab-nlp-templates' already exists
> ```
>
> というエラーが出ます。当該セルで fatal を出した行以降のコードは実行されません。

具体的な関数をインポートする際には、以下のように指示します（例）。

```python
from colab_nlp import (
    tokenize_df,
    build_text_df,
    tokens_to_text,
    create_wordcloud,
    write_df_to_gsheet,
)

```

※ 各関数に必要な外部ライブラリの install 方法や  
　事前に実行すべきテンプレートについては、  
　**対応する docs を必ず参照してください。**

---

## このリポジトリで学ぶ NLP の全体像

本リポジトリでは、**データの読み込み方法**から始め、  
前処理 → Bag of Words → 可視化・出力という  
**日本語 NLP 分析パイプライン**を段階的に学びます。

### 1. テキストデータの取得

- スプレッドシートによる読み込み
    - Google スプレッドシートをテキストの入力データとして使用します ([`docs/load_google_spreadsheet.md`](./docs/load_google_spreadsheet.md))。
    - pandas データフレーム形式で読み込みます。基本的にはどのような表形式でも扱えますが、このプロジェクトでは、id 列と文書列を持つ表の読み込みを想定しています。
    - 少量の文書を手軽に扱え、前処理や分析結果の確認に向いています。
- プレーンテキストの読み込み
    - Google ドライブのマウント方法やファイルの読み込み方法について、[`docs/io_text_basic.md`](./docs/io_text_basic.md) にまとめています。
    - スプレッドシート読み書きラッパーよりも基礎的な方法であり、大規模データの処理などの発展的な内容に応用しやすいです。
    - 文書数が多い場合は、フォルダ配下の `.txt` をまとめて読み込み、(id 列 + 文書列) 形式の DataFrame を自動生成する `build_text_df` を利用できます。

---

### 2. 前処理（形態素解析）

- 内容
    - 日本語テキストの形態素解析
        - 日本語文を単語（トークン）に分割し、  
        「**1行 = 1トークン**」の縦持ち DataFrame に変換します。
        - Janome（軽量・授業向け）と SudachiPy（高精度・研究向け）の  
        2 種類の形態素解析器に対応しています。
        - 解析結果は、語・品詞・付加情報（token_info）を列として保持します。
    - 分析パイプラインの基盤
        - 以降の BoW、語頻度分析、WordCloud などは、  
        この形態素解析結果を前提として進みます。
        - 小規模データから研究用途まで、同じ形式で扱える設計です。
- 参考資料
    - [`docs/tokenization.md`](./docs/tokenization.md)

---

### 3. Bag of Words（BoW）

- 内容
    - BoW の考え方と位置づけ
        - 文書を「単語の集合」として表現し、  
        出現頻度や分布にもとづいて文章を数値化します。
        - 日本語 NLP の基礎となる表現方法です。
    - 基本的な分析と可視化
        - 単語の出現頻度を集計し、  
        文書全体やコーパス全体の特徴を把握します。
        - WordCloud による可視化で、  
        テキストの傾向を直感的に確認できます。
    - 教育・演習向けの最小構成
        - pandas ベースの実装で、  
        処理の流れを追いやすい構成になっています。
- 参考資料
    - BoW 総論・位置づけ
        - [`docs/bow/README.md`](./docs/bow/README.md)
    - 語頻度分析（最小構成
        - [`docs/bow/term_frequency.md`](./docs/bow/term_frequency.md)
    - WordCloud による可視化
        - [`docs/bow/wordcloud.md`](./docs/bow/wordcloud.md)

---

### 4. 出力・共有

- 内容
    - 分析結果の書き戻し
        - 形態素解析結果や BoW の集計結果を、  
        pandas DataFrame として扱います。
        - DataFrame を Google スプレッドシートへ  
        書き戻すための関数を用意しています。
    - 共有・配布を前提とした設計
        - スプレッドシート形式で出力することで、  
        結果の確認・共有・再利用が容易になります。
        - 授業での配布、レポート作成、  
        二次分析への接続を想定しています。
- 参考資料
    - [`docs/write_google_spreadsheet.md`](./docs/write_google_spreadsheet.md)

---

## ライセンス・利用について

- 教育・研究目的での利用を想定しています
- 講義資料・演習ノートへの組み込みも自由です
