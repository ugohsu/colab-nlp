# Google スプレッドシートを Colab から読み込む（基本テンプレ）

本ドキュメントでは、本リポジトリ内の  
[`templates/load_google_spreadsheet.py`](../templates/load_google_spreadsheet.py)  
で提供している **Google スプレッドシートを Google Colab から読み込むための基本テンプレート**について説明します。

本テンプレートは、ゼミ・演習での利用を想定し、

- Google アカウントでの認証
- スプレッドシートの指定
- Pandas DataFrame への変換

という一連の流れを **最小構成・コピペ前提**でまとめたものです。

---

## 想定利用環境

- Google Colaboratory
- Google スプレッドシート
- Pandas を用いたデータ分析・前処理

---

## テンプレートの位置

- 実行用テンプレ  
  - [`templates/load_google_spreadsheet.py`](../templates/load_google_spreadsheet.py)

※ 本テンプレートは **import せず、Colab のコードセルに貼り付けて使用**します。

---

## テンプレートの構成と役割

### 1. 読み込み対象の指定（★要編集）

```python
sheet_url = ""
sheet_name = ""
```

- `sheet_url`  
  読み込みたい Google スプレッドシートの URL を指定します。  
  対象スプレッドシートには **「閲覧可」以上の権限**が必要です。

- `sheet_name`  
  スプレッドシート内の **ワークシート（タブ）の名前**を指定します。  
  表示されているタブ名と完全一致している必要があります。

---

### 2. Google アカウント認証

```python
from google.colab import auth
auth.authenticate_user()
```

- Google Colab から Google Drive / Spreadsheet にアクセスするために必要な処理です。
- **Colab セッションごとに 1 回だけ実行**すれば十分です。
- 実行時に、ブラウザ上で Google アカウントへのログインが求められます。

---

### 3. ライブラリの import

```python
import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe
from google.auth import default
```

- `gspread`  
  Google スプレッドシートを操作するためのライブラリ
- `gspread_dataframe.get_as_dataframe`  
  ワークシートの内容を Pandas DataFrame に変換するために使用します

※ Google Colab では、これらは **標準で利用可能**です（追加の pip install は不要）。

---

### 4. 認証情報の取得とクライアント作成

```python
creds, _ = default()
gc = gspread.authorize(creds)
```

- `google.auth.default()` により、Colab 環境に紐づいた認証情報を自動取得します。
- 取得した認証情報を使って、`gspread` のクライアントを作成します。

---

### 5. スプレッドシート → DataFrame

```python
worksheet = gc.open_by_url(sheet_url).worksheet(sheet_name)
df = get_as_dataframe(worksheet, header=0)
```

- `open_by_url()`  
  URL を指定してスプレッドシート全体を開きます。
- `worksheet()`  
  その中から、指定したワークシート（タブ）を取得します。
- `get_as_dataframe()`  
  ワークシートの内容を Pandas DataFrame に変換します。

`header=0` は  
**「1 行目を列名として扱う」**ことを意味します。

---

## よくあるエラーと注意点

### 権限エラーが出る場合

- スプレッドシートの共有設定を確認してください。
- **「閲覧可」以上の権限**が付与されていないと読み込めません。

### シート名エラーが出る場合

- `sheet_name` は **タブ名と完全一致**する必要があります。
- 全角・半角、スペースの違いにも注意してください。

---

## 想定ユースケース

- アンケート結果の読み込み
- 授業・演習用データの配布
- NLP 前処理前のデータ取得
- Google Forms と連携したデータ分析

