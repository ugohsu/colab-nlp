# ============================================================
# Google スプレッドシート I/O（書き込み）
# ------------------------------------------------------------
# - Google Colab 前提（認証済み gspread client を受け取る）
# - gspread_dataframe.set_with_dataframe を使用
# - 「とりあえず df を出力して確認したい」用途を強く意識
#
# 改善点（2025-12）
# - sheet_name を省略できる（既定 "temporary"）
# - Series / value_counts() 由来のオブジェクトでも「良しなに」書き込む
#   （reset_index し忘れ、列名なし、MultiIndex などを自動で整形）
# - gc を省略できる（Colab 前提で自動認証・キャッシュ）
# ============================================================

from __future__ import annotations

from typing import Any, Iterable, Union

import pandas as pd
from gspread_dataframe import set_with_dataframe


# モジュール内キャッシュ（Colab の認証・初期化を毎回走らせない）
_GC = None


def get_gspread_client_colab(*, force: bool = False):
    """
    Google Colab 前提で、認証済み gspread client (gc) を返します。

    - 初回のみ認証・初期化を行い、以降はモジュール内にキャッシュします
    - force=True の場合はキャッシュを無視して再初期化します

    Notes
    -----
    Colab 以外（ローカル Python 等）では動作しない想定です。
    """
    global _GC
    if _GC is not None and not force:
        return _GC

    # Colab 認証
    from google.colab import auth

    auth.authenticate_user()

    import gspread
    from google.auth import default

    creds, _ = default()
    _GC = gspread.authorize(creds)
    return _GC


def _make_unique(names: Iterable[str]) -> list[str]:
    """列名を重複なしに整形する（例: a, a -> a, a_2）"""
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        base = str(n)
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def _flatten_columns(cols: Any) -> list[str]:
    """
    MultiIndex columns を 1段の文字列にする。
    例: ('a','b') -> 'a|b'
    """
    if isinstance(cols, pd.MultiIndex):
        flat: list[str] = []
        for tup in cols.to_list():
            parts = [str(x) for x in tup if x is not None and str(x) != ""]
            flat.append("|".join(parts) if parts else "")
        return flat
    return [str(c) if c is not None else "" for c in list(cols)]


def normalize_for_gsheet(
    obj: Union[pd.DataFrame, pd.Series],
    *,
    include_index: bool = False,
    index_name_default: str = "index",
) -> pd.DataFrame:
    """
    Google Sheets に書き込みやすい形に整形する。

    想定する「よくある困りごと」
    - value_counts() の結果（Series）をそのまま渡してしまう
    - reset_index() を忘れて index が情報を持っているのに列に出ない
    - columns が None / 空文字 / 重複 / MultiIndex
    """
    if isinstance(obj, pd.Series):
        # value_counts() など：Series -> DataFrame（列名が無い場合は count）
        name = obj.name if obj.name is not None and str(obj.name) != "" else "count"
        df = obj.to_frame(name=name)
    else:
        df = obj.copy()

    # index を列として残したい場合：
    # - include_index=True のときは set_with_dataframe 側で index を出すので reset_index はしない
    # - include_index=False でも、index が RangeIndex 以外なら情報が落ちるので reset_index する
    if not include_index:
        if not isinstance(df.index, pd.RangeIndex) or (df.index.name is not None):
            df = df.reset_index()
            if df.columns[0] in ("index", None, ""):
                df = df.rename(columns={df.columns[0]: index_name_default})

    # columns をフラット化 & 文字列化
    cols = _flatten_columns(df.columns)

    # 空列名は col_1, col_2... に置換
    fixed: list[str] = []
    for i, c in enumerate(cols, start=1):
        c2 = c.strip()
        fixed.append(c2 if c2 != "" else f"col_{i}")

    fixed = _make_unique(fixed)
    df.columns = fixed

    return df


def write_df_to_gsheet(
    df,
    sheet_url: str,
    *,
    gc: Any = None,
    sheet_name: str = "temporary",
    clear_sheet: bool = True,
    include_index: bool = False,
    normalize: bool = True,
) -> None:
    """
    Pandas DataFrame / Series を Google スプレッドシートに書き込む（上書き保存）。

    この関数は「とりあえず df の中身を確認したい」用途を強く意識しています。

    Parameters
    ----------
    df:
        書き込みたいデータ。DataFrame だけでなく Series（value_counts など）も可。
        Series の場合は内部で DataFrame に変換して書き込みます（normalize=True 既定）。
    sheet_url:
        書き込み先スプレッドシートの URL（書き込み権限が必要）。
    gc:
        認証済みの gspread クライアント。
        省略（None）の場合は、Colab 前提で自動的に認証して取得します（get_gspread_client_colab）。
    sheet_name:
        書き込み先のシート名。既定は "temporary"。
    clear_sheet:
        True の場合、書き込み前にシートを消去します（上書き用途向け）。
    include_index:
        True の場合、DataFrame の index をシートに含めます（set_with_dataframe の機能）。
    normalize:
        True の場合、Sheets に書き込みやすい形に自動整形します（推奨）。
    """
    if gc is None:
        gc = get_gspread_client_colab()

    # 1) Open spreadsheet
    sh = gc.open_by_url(sheet_url)

    # 2) Worksheet を取得（無ければ作る）
    try:
        ws = sh.worksheet(sheet_name)
    except Exception:
        ws = sh.add_worksheet(title=sheet_name, rows=100, cols=26)

    # 3) シート消去（必要なら）
    if clear_sheet:
        ws.clear()

    # 4) DataFrame 整形（必要なら）
    df_out = normalize_for_gsheet(df, include_index=include_index) if normalize else (
        df.to_frame(name=(df.name or "count")) if isinstance(df, pd.Series) else df
    )

    # 5) 書き込み
    set_with_dataframe(
        ws,
        df_out,
        include_index=include_index,
        include_column_header=True,
        resize=True,
    )

    print("✅ DataFrame written to Google Sheets:", sheet_url, "/", sheet_name)
