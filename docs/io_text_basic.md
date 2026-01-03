# テキスト入出力（基本）

このドキュメントでは、Google Colab 上で  
**プレーンテキスト（.txt など）を読み込むための基本的な方法**をまとめます。

- まずは、Python の基本的な方法で **1つのテキストファイルを確実に読む**
- 次に、近年よく使われる **別の書き方（pathlib）** を紹介します
- そのうえで、文書数が増えた場合に役立つ **発展的な方法**へと進みます

---

## 1. Google Drive をマウントする（Colab）

Google Drive 上のファイルを読み込む場合は、  
最初に Drive をマウントします。

```python
from google.colab import drive
drive.mount("/content/drive")
```

マウント後、Google Drive 内のファイルは次のようなパスで参照できます。

- `/content/drive/MyDrive/...`

---

## 2. ファイルの場所を確認する

たとえば、Google Drive 上に次のようなファイルがあるとします。

```
MyDrive/
 └─ data/
     └─ sample.txt
```

このファイルは、次のパスで参照できます。

```python
file_path = "/content/drive/MyDrive/data/sample.txt"
```

---

## 3. with open を使ってテキストを読む（基本）

Python でファイルを読み込む最も基本的な方法は、  
`with open` を使う方法です。

```python
file_path = "/content/drive/MyDrive/data/sample.txt"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()
```

この方法は、

- 追加のライブラリが不要
- Python の基本構文だけで理解できる

という点で、**最初に学ぶ方法として適しています**。

### 文字コードについて

日本語テキストでは、文字コードの違いによって  
エラーや文字化けが起こることがあります。

その場合は、`errors="replace"` を指定すると、  
エラーで止まらずに読み込むことができます。

```python
with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()
```

---

## 4. pathlib を使ってテキストを読む（別の書き方）

`with open` による読み込みは、Python の基本的な方法です。  
一方で、近年の Python では **`pathlib`** を使って  
より簡潔にファイルを扱う書き方もよく使われます。

### pathlib とは

`pathlib` は、ファイルパスを **オブジェクトとして扱う**ための  
Python 標準ライブラリです。

```python
from pathlib import Path
```

### pathlib を使った読み込み例

```python
from pathlib import Path

file_path = Path("/content/drive/MyDrive/data/sample.txt")

text = file_path.read_text(encoding="utf-8")
```

このコードは、次の `with open` の処理とほぼ同じ意味です。

```python
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()
```

### どちらを使えばよいか

- **Python の基本を学ぶ目的**  
  → `with open` を使う
- **コードを簡潔に書きたい場合**  
  → `pathlib` を使う

本リポジトリの内部実装（関数）では、  
簡潔さと可読性の観点から `pathlib` を使うことがありますが、  
どちらを使っても問題ありません。

---

## 5. 読み込んだ内容を確認する

読み込んだテキストの中身は、`print` などで確認できます。

```python
print(text[:200])
```

長い文章の場合は、先頭の一部だけを表示すると確認しやすくなります。

---

## 6. （発展）フォルダ配下のテキストをまとめて DataFrame にする

前節までで、「1つのテキストファイルを確実に読む」ことができました。  
次のステップとして、スプレッドシート（id 列 + 文書列）を  
手で用意する代わりに、

**フォルダに置いた複数のテキストファイルから  
自動で DataFrame を作る**

方法もあります。

このリポジトリでは、そのための関数 `build_text_df` を用意しています。

- 指定ディレクトリ配下（必要ならサブディレクトリ含む）を走査
- 指定拡張子のファイルをすべて読み込み
- **1ファイル = 1文書**として、(id, 文書) 形式の DataFrame を作成

### 最小例（.txt をすべて読む）

```python
from libs import build_text_df

ROOT_DIR = "/content/drive/MyDrive/data"

df = build_text_df(
    ROOT_DIR,
    exts=(".txt",),
)
```

この DataFrame は、最低限次の列を持ちます。

- `article_id`（文書ID）
- `article`（本文）

加えて、確認用に次の列も作られます。

- `path`（フルパス）
- `relpath`（ROOT_DIR からの相対パス）


> 目安：授業・小規模データでは、  
> まずは「with open」で 1本読むところまで理解できれば十分です。  
> 文書数が増えてきたら、この `build_text_df` を使うとスムーズです。
