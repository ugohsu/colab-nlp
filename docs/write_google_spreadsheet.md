# Google スプレッドシートへの書き込み

このドキュメントでは、  
**pandas DataFrame / Series を Google スプレッドシートへ書き戻す方法**を説明します。

本リポジトリは **Google Colab 上での利用を前提**としています。

---

## 基本方針

- 書き込みはすべて `libs.gsheet_io.write_df_to_gsheet` を使用します
- Google Colab 前提のため、認証処理は内部で自動化されています
- 「とりあえず中身を確認したい」用途を重視しています

---

## クイックスタート（最小構成）

Google Colab 上では、`write_df_to_gsheet` を **そのまま呼ぶだけ**で  
Google スプレッドシートに書き込めます。

```python
from libs import write_df_to_gsheet

write_df_to_gsheet(df, SHEET_URL)
```

- `gc`（gspread client）は **省略可能**です  
  → Colab 前提で、自動的に認証・取得されます
- `sheet_name` を省略した場合は `"temporary"` が使われます
- `df` には `DataFrame` だけでなく  
  `value_counts()` などの **Series もそのまま渡せます**

---

## gc を明示的に使いたい場合（複数回書き込み）

同じノートブックで何度も書き込む場合は、  
`gc` を一度だけ用意して使い回すほうが効率的です。

```python
from libs import get_gspread_client_colab, write_df_to_gsheet

gc = get_gspread_client_colab()

write_df_to_gsheet(
    df,
    SHEET_URL,
    gc=gc,
)
```

- `get_gspread_client_colab()` は内部でキャッシュされるため、
  何度呼んでも再認証は行われません

---

## sheet_name（書き込み先シート名）

```python
write_df_to_gsheet(
    df,
    SHEET_URL,
    sheet_name="result",
)
```

- `sheet_name` を省略すると `"temporary"` が使われます
- シートが存在しない場合は自動的に作成されます
- 既存シートがある場合は、既定で **上書き**されます

---

## Series（value_counts など）を書き込む場合

```python
pos_counts = df_tok["pos"].value_counts()

write_df_to_gsheet(
    pos_counts,
    SHEET_URL,
)
```

`write_df_to_gsheet` は内部で自動的に整形を行うため、

- `Series` → `DataFrame` への変換
- `reset_index()` の自動適用
- 列名の補完・重複解消
- MultiIndex のフラット化

などを **ユーザーが意識せずに**書き込めます。

---

## まとめ

- 書き込み API は `libs` に集約されています
- 初学者は「最小構成」で、  
  中級者以上は `gc` 注入で柔軟に使えます

