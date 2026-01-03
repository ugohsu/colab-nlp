# ============================================================
# テキスト入出力（基本）
# ------------------------------------------------------------
# Colab で「任意のフォルダ配下の .txt 等」をまとめて読み込み、
# 形態素解析（tokenize_df）にそのまま渡せる DataFrame を作るための関数です。
#
# 目的：
#   - スプレッドシート（id列 + 文書列）を手で用意するハードルを下げる
#   - 「フォルダにテキストを置く → すぐ分析」の入口を提供する
#
# 設計方針：
#   - 対象は「プレーンテキスト」を基本（.txt / .md など）
#   - PDF/Word/HTML などの抽出・整形は「応用」として本関数では扱わない
#   - 文字コードは utf-8 を既定にしつつ、失敗しても止まらない設定（errors="replace"）を既定にする
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd


PathLike = Union[str, Path]


def read_text_file(
    path: PathLike,
    *,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> str:
    """
    1つのテキストファイルを読み込みます。

    Parameters
    ----------
    path:
        読み込むファイルパス。
    encoding:
        文字コード。既定は "utf-8"。
    errors:
        デコード失敗時の挙動。既定は "replace"（置換して続行）。

    Returns
    -------
    str
        ファイル内容（文字列）。
    """
    p = Path(path)
    return p.read_text(encoding=encoding, errors=errors)


def build_text_df(
    root_dir: PathLike,
    *,
    exts: Union[str, Sequence[str]] = (".txt",),
    recursive: bool = True,
    encoding: str = "utf-8",
    errors: str = "replace",
    text_col: str = "text",
    id_col: str = "doc_id",
    path_col: str = "path",
    relpath_col: str = "relpath",
    include_empty: bool = False,
    sort_paths: bool = True,
) -> pd.DataFrame:
    """
    指定ディレクトリ配下のファイルを走査して、(id, 文書) 形式の DataFrame を作成します。

    - root_dir 配下（必要ならサブディレクトリ含む）を走査
    - 指定拡張子のファイルだけ対象
    - 1ファイル = 1文書として読み込み、pandas DataFrame にまとめる

    Parameters
    ----------
    root_dir:
        走査の起点ディレクトリ。
    exts:
        対象拡張子。".txt" のように 1つでも、(".txt", ".md") のように複数でも可。
        ドット付き（".txt"）を推奨します。
    recursive:
        True ならサブディレクトリも含めて走査します。
    encoding, errors:
        テキスト読み込み時の文字コード設定。
    text_col, id_col:
        出力 DataFrame の文書列名・ID列名。
    path_col, relpath_col:
        参考情報として、フルパスと root_dir からの相対パス列を付与します。
        不要なら None を指定してください。
    include_empty:
        空文字（strip して空）の文書も含めるか。既定 False（除外）。
    sort_paths:
        True ならパスでソートしてから ID を振ります（再現性が上がる）。

    Returns
    -------
    pd.DataFrame
        最低限、id_col と text_col を含む DataFrame。
        追加で path_col / relpath_col を含みます（None 指定なら作りません）。
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"root_dir not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir is not a directory: {root}")

    if isinstance(exts, str):
        exts = (exts,)

    # 先頭ドット無しが混ざっても動くように補正（".txt" 推奨だが親切に）
    exts_norm = tuple(e if e.startswith(".") else f".{e}" for e in exts)

    pattern = "**/*" if recursive else "*"
    paths = [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in exts_norm]
    if sort_paths:
        paths = sorted(paths, key=lambda p: str(p))

    rows = []
    for i, p in enumerate(paths, start=1):
        try:
            text = read_text_file(p, encoding=encoding, errors=errors)
        except Exception as e:
            # 1ファイルの失敗で全体を止めない（Colab/教育用途では特に重要）
            # ただし原因追跡のため、例外内容は列に残せるようにしたい場合は拡張してください。
            continue

        if (not include_empty) and (text.strip() == ""):
            continue

        row = {
            id_col: i,
            text_col: text,
        }
        if path_col is not None:
            row[path_col] = str(p)
        if relpath_col is not None:
            row[relpath_col] = str(p.relative_to(root))
        rows.append(row)

    return pd.DataFrame(rows)
