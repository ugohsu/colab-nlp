# Colab NLP Utilities

このリポジトリは、**Google Colaboratory 上で自然言語処理（NLP）を学ぶための関数・解説ドキュメント**をまとめた教材用リポジトリです。

---

## ライブラリインポートの手順

本リポジトリで提供されている関数を使用するための基本的な手順は次のとおりです。

---

### 1. リポジトリを clone

```python
!git clone https://github.com/ugohsu/colab-nlp.git
```

### 2. import 用のパスを追加

```python
import sys
sys.path.append("/content/colab-nlp")
```

### 3. 関数を import (例)

```python
from colab_nlp import tokenize_df
```

---

データの入出力などに関する関数は、[colab-common](https://github.com/ugohsu/colab-common.git) リポジトリで提供されています。両リポジトリを導入する場合は、以下のような記述をおこない、セルを実行するとよいでしょう。

```python
## パス設定のために必要
import sys

## colab-common
!git clone https://github.com/ugohsu/colab-common.git
sys.path.append("/content/colab-common")

## colab-nlp
!git clone https://github.com/ugohsu/colab-nlp.git
sys.path.append("/content/colab-nlp")
```

---

### 注意（Google Colab での git clone）

同一ノートブック内で同一リポジトリに対する `git clone` を **2 回以上実行しないでください**。

```
fatal: destination path 'colab-common' already exists
```

というエラーが発生し、そのセルでは、当該行以降のコードが実行されなくなります。

---

## 関数一覧（import して使う）

| 分類 | 関数名 | 内容 | 実装ファイル | 解説ドキュメント |
| :--- | :--- | :--- | :--- | :--- |
| **I/O** | `CorpusDB` | 大規模テキストデータのDB化・管理（省メモリ・中断再開可） | [`colab_nlp/corpus_db.py`](./colab_nlp/corpus_db.py) | [`docs/corpus_db.md`](./docs/corpus_db.md) |
| **I/O** | `corpus_reader` | 大規模DBからトークンを逐次読み込み（ジェネレータ） | [`colab_nlp/corpus_db.py`](./colab_nlp/corpus_db.py) | [`docs/corpus_db.md`](./docs/corpus_db.md) |
| **前処理** | `tokenize_df` | 文書DFを形態素解析し、縦持ち（1行1語）形式に変換 | [`colab_nlp/preprocess.py`](./colab_nlp/preprocess.py) | [`docs/tokenization.md`](./docs/tokenization.md) |
| **前処理** | `tokenize_text_janome` | Janomeを用いて1つの文字列をトークン化（内部・単体用） | [`colab_nlp/preprocess.py`](./colab_nlp/preprocess.py) | [`docs/tokenization.md`](./docs/tokenization.md) |
| **前処理** | `tokenize_text_sudachi` | Sudachiを用いて1つの文字列をトークン化（内部・単体用） | [`colab_nlp/preprocess.py`](./colab_nlp/preprocess.py) | [`docs/tokenization.md`](./docs/tokenization.md) |
| **前処理** | `filter_tokens_df` | トークンDFから特定の品詞やストップワードを除外・抽出 | [`colab_nlp/preprocess.py`](./colab_nlp/preprocess.py) | [`docs/tokenization.md`](./docs/tokenization.md) |
| **前処理** | `tokens_to_text` | トークンDFを分かち書きテキスト（文字列）に再結合 | [`colab_nlp/preprocess.py`](./colab_nlp/preprocess.py) | [`docs/bow/wordcloud.md`](./docs/bow/wordcloud.md) |
| **BoW / 可視化** | `create_wordcloud` | 分かち書きテキストからWordCloud画像を生成・表示 | [`colab_nlp/bow.py`](./colab_nlp/bow.py) | [`docs/bow/wordcloud.md`](./docs/bow/wordcloud.md) |
| **BoW** | `compute_ngram` | トークン列からN-gram頻度を集計（DataFrame/大規模対応） | [`colab_nlp/ngram.py`](./colab_nlp/ngram.py) | [`docs/bow/ngram.md`](./docs/bow/ngram.md) |
| **BoW / 可視化** | `create_network_graph` | 共起ネットワーク図を作成・表示 | [`colab_nlp/network_graph.py`](./colab_nlp/network_graph.py) | [`docs/bow/network_graph.md`](./docs/bow/network_graph.md) |

---

## ドキュメント一覧

| 分類 | 内容 | ドキュメント |
| :--- | :--- | :--- |
| **I/O** | 大規模コーパス構築ガイド (CorpusDB) | [`docs/corpus_db.md`](./docs/corpus_db.md) |
| **前処理** | Janome / SudachiPy による形態素解析とDataFrame化 | [`docs/tokenization.md`](./docs/tokenization.md) |
| **前処理** | Sudachi ユーザー辞書の作成・設定方法 | [`docs/Sudachi_user_dict.md`](./docs/Sudachi_user_dict.md) |
| **BoW** | Bag of Words（BoW）の概念と位置づけ（総論） | [`docs/bow/README.md`](./docs/bow/README.md) |
| **BoW** | 語頻度（Term Frequency）の集計方法 | [`docs/bow/term_frequency.md`](./docs/bow/term_frequency.md) |
| **BoW / 可視化** | WordCloud による可視化 | [`docs/bow/wordcloud.md`](./docs/bow/wordcloud.md) |
| **BoW** | N-gram（Nグラム）の集計と大規模データ対応 | [`docs/bow/ngram.md`](./docs/bow/ngram.md) |
| **BoW / 可視化** | 共起ネットワーク分析による可視化 | [`docs/bow/network_graph.md`](./docs/bow/network_graph.md) |
| **Modeling** | モデリング・ベクトル化（総論） | [`docs/modeling/README.md`](./docs/modeling/README.md) |
| **Modeling** | Word2Vec による単語の分散表現 | [`docs/modeling/word2vec.md`](./docs/modeling/word2vec.md) |

---

## このリポジトリで学ぶ NLP の全体像

本リポジトリでは、データの準備から前処理、分析、可視化までの **日本語 NLP 分析パイプライン** を段階的に学びます。

### 1. テキストデータの取得

データの読み込みに関する基本的な機能は、共通ライブラリ **[colab-common](https://github.com/ugohsu/colab-common)** に集約しています。

- **基本的な読み込み（colab-common）**
    - テキストファイルの読み込み、Google スプレッドシートや SQLite との連携については、`colab-common` のドキュメント・関数を使用してください。
    - 分析の第一歩として、手持ちのデータを `pandas.DataFrame`（`doc_id` 列と `text` 列を持つ形式）に読み込むことを目指します。
- **大規模コーパスの構築（CorpusDB）**
    - 本リポジトリ（`colab-nlp`）では、大量のテキストファイルをメモリを節約しながら SQLite データベース化するクラス `CorpusDB` を提供しています。
    - 処理の中断・再開が可能で、Google Colab 上で数万〜数百万ファイル規模の文書集合を安全に処理するために使用します。
- 参考資料
    - [`docs/corpus_db.md`](./docs/corpus_db.md) 

---

### 2. 前処理（形態素解析）

- 内容
    - 日本語テキストの形態素解析
        - 日本語文を単語（トークン）に分割し、「**1行 = 1トークン**」の縦持ち DataFrame に変換します。
        - Janome（軽量・授業向け）と SudachiPy（高精度・研究向け）の 2 種類の形態素解析器に対応しています。
        - 解析結果は、語・品詞・付加情報（token_info）を列として保持します。
    - 分析パイプラインの基盤
        - 以降の BoW、語頻度分析、WordCloud などは、この形態素解析結果を前提として進みます。
        - 小規模データから研究用途まで、同じ形式で扱える設計です。
- 参考資料
    - [`docs/tokenization.md`](./docs/tokenization.md)

---

### 3. Bag of Words（BoW）

- 内容
    - BoW の考え方と位置づけ
        - 文書を「単語の集合」として表現し、出現頻度や分布にもとづいて文章を数値化します。
        - 日本語 NLP の基礎となる表現方法です。
    - 基本的な分析と可視化
        - 単語の出現頻度を集計し、文書全体やコーパス全体の特徴を把握します。
        - WordCloud による可視化で、テキストの傾向を直感的に確認できます。
    - 教育・演習向けの最小構成
        - pandas ベースの実装で、処理の流れを追いやすい構成になっています。
- 参考資料
    - BoW 総論・位置づけ
        - [`docs/bow/README.md`](./docs/bow/README.md)
    - 語頻度分析
        - [`docs/bow/term_frequency.md`](./docs/bow/term_frequency.md)
    - WordCloud による可視化
        - [`docs/bow/wordcloud.md`](./docs/bow/wordcloud.md)
    - N-gram
        - [`docs/bow/ngram.md`](./docs/bow/ngram.md)
    - 共起ネットワーク
        - [`docs/bow/network_graph.md`](./docs/bow/network_graph.md)

---

### 4. モデリング（ベクトル化）

- 内容
    - テキストデータを分散表現に変換し、機械学習モデル（Word2Vec, Doc2Vec, LDAなど）を適用します。
    - 単語の意味計算、類似文書検索、トピック分類などが可能になります。
- 参考資料
    - モデリング総論
        - [`docs/modeling/README.md`](./docs/modeling/README.md)
    - Word2Vec
        - [`docs/modeling/word2vec.md`](./docs/modeling/word2vec.md)

---

### 5. 出力・共有・保存

- 内容
    - 分析結果の出力
        - 形態素解析結果や BoW の集計結果を `colab-common` の機能を用いて出力・保存します。
        - **Google スプレッドシート** へ書き戻すことで、結果の共有や可視化が容易になります。
        - **SQLite データベース** へ保存することで、大規模なデータを効率的に管理し、SQL を用いた柔軟な検索や再利用が可能になります。
    - 二次利用を見据えた設計
        - 授業での配布やレポート作成だけでなく、SQL を活用した高度な二次分析への接続も想定しています。
- 参考資料
    - [colab-common: Google スプレッドシートの入出力](https://github.com/ugohsu/colab-common/blob/main/docs/gsheet_io.md)
    - [colab-common: データベース入出力（SQLite / SQL 基礎）](https://github.com/ugohsu/colab-common/blob/main/docs/io_sql_basic.md)

---

## ライセンス・利用について

- 教育・研究目的での利用を想定しています
- 講義資料・演習ノートへの組み込みも自由です
