"""
前処理（形態素解析）ユーティリティ

設計方針
- tokenize_df は「Janome / Sudachi 共通のオプション」を中心に提供する
- Sudachi 固有の split_mode / word_form（dictionary/surface/normalized）などは tokenize_text_sudachi に残す
- split_mode を変更したい場合は、tokenize_text_fn を使って tokenize_text_sudachi を呼び出す（レシピは tokenization.md 参照）
- 品詞フィルタは、まず tokenize（全トークン）→ あとで filter_tokens_df で段階的に調整する運用を推奨
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple, List, Dict, Any
import pandas as pd


# ----------------------------------------------------------------------
# 内部ユーティリティ
# ----------------------------------------------------------------------

def _compile_pos_filter(
    pos_keep: Optional[Iterable[str]],
    pos_exclude: Optional[Iterable[str]],
    *,
    strict: bool = True,
) -> Tuple[Optional[set], Optional[set]]:
    """
    pos_keep / pos_exclude を set に正規化し、必要なら矛盾を検査する
    """
    keep_set = set(pos_keep) if pos_keep else None
    excl_set = set(pos_exclude) if pos_exclude else None

    # NOTE:
    # strict=True のとき、
    #   keep と exclude を同時指定して「交差がゼロ」だと、
    #   指定ミス（意図とズレ）の可能性が高いのでエラーにする。
    if strict and keep_set and excl_set:
        if keep_set.isdisjoint(excl_set):
            raise ValueError(
                "pos_keep と pos_exclude が同時指定されていますが、"
                "両者に交差がありません。指定ミスの可能性があります。"
            )

    return keep_set, excl_set


def _normalize_stopwords(stopwords) -> set:
    """
    stopwords を「除外語集合（set[str]）」に正規化する。

    受け付ける入力（単体/複合を問わず）：
    - None
    - str（単語1つ）
    - iterable[str]
    - pandas.Series（value_counts() などを想定：index を語として採用）
    - pandas.Index
    - iterable of the above（入れ子も可）

    注意：
    - str は iterable だが、文字単位に分解しない（単語1つとして扱う）
    """
    result: set = set()

    if stopwords is None:
        return result

    # str は最優先（iterable 扱いしない）
    if isinstance(stopwords, str):
        result.add(stopwords)
        return result

    # pandas Series / Index
    if isinstance(stopwords, pd.Series):
        # value_counts() の想定：index が語
        result |= set(stopwords.index.astype(str))
        return result

    if isinstance(stopwords, pd.Index):
        result |= set(stopwords.astype(str))
        return result

    # iterable（list/tuple/set など、入れ子も含む）
    try:
        for sw in stopwords:
            result |= _normalize_stopwords(sw)
        return result
    except TypeError:
        # iterable でない単体：文字列化して 1 語として扱う
        result.add(str(stopwords))
        return result


# ----------------------------------------------------------------------
# 公開ユーティリティ（tokenize 後の操作）
# ----------------------------------------------------------------------

def filter_tokens_df(
    df: pd.DataFrame,
    *,
    pos_keep: Optional[Iterable[str]] = None,
    pos_exclude: Optional[Iterable[str]] = None,
    stopwords=None,
    strict: bool = True,
    keep_original_index: bool = False,
) -> pd.DataFrame:
    """
    tokenize 後の DataFrame に対して品詞フィルタを適用する

    Parameters
    ----------
    df : pandas.DataFrame
        tokenize_df の出力（pos 列を含む）
    pos_keep : iterable[str], optional
        残す品詞（大分類）
    pos_exclude : iterable[str], optional
        除外する品詞（大分類）
    stopwords : optional
        除外語（word 列で判定）。str / list / tuple / set / pandas.Series / pandas.Index /
        それらの入れ子を受け付け、内部で良しなに集合化して除外する。
        例:
            stopwords="ある"
            stopwords=["ある", "いる"]
            stopwords=df["word"].value_counts().head(10)
            stopwords=[["ある", "いる"], df["word"].value_counts().head(10)]
    strict : bool, default True
        keep / exclude の同時指定時に矛盾を検出するか
    keep_original_index : bool, default False
        True の場合、元の index を保持する。False の場合、index を振り直す。

    Returns
    -------
    pandas.DataFrame
    """
    if "pos" not in df.columns:
        raise KeyError("filter_tokens_df: DataFrame に 'pos' 列がありません")

    keep_set, excl_set = _compile_pos_filter(
        pos_keep=pos_keep,
        pos_exclude=pos_exclude,
        strict=strict,
    )

    s = df
    if keep_set is not None:
        s = s[s["pos"].isin(keep_set)]
    if excl_set is not None:
        s = s[~s["pos"].isin(excl_set)]

    if stopwords is not None and not s.empty:
        if "word" not in s.columns:
            raise KeyError("filter_tokens_df: stopwords が指定されましたが DataFrame に 'word' 列がありません")
        sw_set = _normalize_stopwords(stopwords)
        if sw_set:
            s = s[~s["word"].isin(sw_set)]

    return s if keep_original_index else s.reset_index(drop=True)


# ----------------------------------------------------------------------
# 1 テキスト用 tokenizer（Janome）
# ----------------------------------------------------------------------

def tokenize_text_janome(
    text: str,
    *,
    tokenizer,
    use_base_form: bool = True,
    extra_col: Optional[str] = "token_info",
) -> List[Tuple[str, str, Optional[Dict[str, Any]]]]:
    """
    Janome による 1 テキスト分のトークナイズ

    Returns
    -------
    list of (word, pos, token_info)
    """
    if text is None:
        return []

    s = str(text).strip()
    if s == "":
        return []

    records: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []
    for token in tokenizer.tokenize(s):
        # word: 原形（base_form） or 表層形（surface）
        word = token.base_form if use_base_form else token.surface
        # 品詞（大分類）
        pos = token.part_of_speech.split(",")[0]

        token_info = None
        if extra_col is not None:
            token_info = {
                "surface": token.surface,
                "base_form": token.base_form,
                "reading": token.reading,
            }

        records.append((word, pos, token_info))

    return records


# ----------------------------------------------------------------------
# 1 テキスト用 tokenizer（Sudachi）
# ----------------------------------------------------------------------

def tokenize_text_sudachi(
    text: str,
    *,
    tokenizer,
    split_mode: str = "C",
    word_form: Optional[str] = None,
    use_base_form: bool = True,
    extra_col: Optional[str] = "token_info",
) -> List[Tuple[str, str, Optional[Dict[str, Any]]]]:
    """
    SudachiPy による 1 テキスト分のトークナイズ

    Parameters
    ----------
    split_mode : {"A","B","C"}
        Sudachi の分割モード（デフォルト "C"）
    word_form : str | None
        出力する word の形式を指定します。None の場合は use_base_form に従います。
        - "dictionary" : m.dictionary_form()
        - "surface"    : m.surface()
        - "normalized" : m.normalized_form()
        ※ 大文字小文字は無視します（"Norm" なども可）
    use_base_form : bool
        word_form が None の場合のみ参照されます。
        True なら dictionary_form、False なら surface を返します。
    extra_col : str | None
        None の場合 token_info を作りません（高速化・軽量化したい場合）

    Returns
    -------
    list of (word, pos, token_info)
    """
    if text is None:
        return []

    s = str(text).strip()
    if s == "":
        return []

    try:
        from sudachipy import SplitMode
    except ImportError as e:
        raise ImportError(
            "SudachiPy が見つかりません。Sudachi を使う場合は "
            "`pip install sudachipy sudachidict_core` を実行してください。"
        ) from e

    mode_map = {
        "A": SplitMode.A,
        "B": SplitMode.B,
        "C": SplitMode.C,
    }
    mode = mode_map.get(str(split_mode).upper(), SplitMode.C)

    records: List[Tuple[str, str, Optional[Dict[str, Any]]]] = []
    for m in tokenizer.tokenize(s, mode):
        if word_form is None:
            word = m.dictionary_form() if use_base_form else m.surface()
        else:
            wf = str(word_form).lower()
            if wf in {"dictionary", "dict", "base", "lemma"}:
                word = m.dictionary_form()
            elif wf in {"surface", "orig", "original"}:
                word = m.surface()
            elif wf in {"normalized", "normal", "norm"}:
                word = m.normalized_form()
            else:
                raise ValueError(
                    f"tokenize_text_sudachi: invalid word_form={word_form!r}. "
                    "Use 'dictionary'/'surface'/'normalized' (or omit)."
                )

        pos = m.part_of_speech()[0]

        token_info = None
        if extra_col is not None:
            token_info = {
                "surface": m.surface(),
                "dictionary_form": m.dictionary_form(),
                "normalized_form": m.normalized_form(),
            }

        records.append((word, pos, token_info))

    return records


# ----------------------------------------------------------------------
# DataFrame 用 tokenizer（入口）
# ----------------------------------------------------------------------

def tokenize_df(
    df: pd.DataFrame,
    *,
    id_col: str = "doc_id",
    text_col: str = "text",
    token_id_col: str = "token_id",
    engine: str = "janome",
    tokenizer=None,
    tokenize_text_fn: Optional[Callable[[str], List[Tuple[str, str, Any]]]] = None,
    use_base_form: bool = True,
    pos_keep: Optional[Iterable[str]] = None,
    pos_exclude: Optional[Iterable[str]] = None,
    stopwords: Optional[Iterable[str]] = None,
    extra_col: Optional[str] = "token_info",
) -> pd.DataFrame:
    """
    テキスト DataFrame を縦持ち token DataFrame に変換する（入口関数）

    方針
    - tokenize_df は Janome / Sudachi 共通の引数だけに寄せる
    - Sudachi の split_mode / word_form を変えたい場合は tokenize_text_fn を使用する

    Parameters
    ----------
    df : pandas.DataFrame
        入力 DataFrame（文書単位）
    id_col : str
        文書ID列名
    text_col : str
        テキスト列名
    token_id_col : str
        文書内通し番号の列名
    engine : {"janome","sudachi"}
        形態素解析エンジン（デフォルトは janome）
    tokenizer : optional
        エンジンの tokenizer を外部で生成して渡す（高速化・使い回し）
    tokenize_text_fn : callable, optional
        1テキストのトークナイズ処理を完全に差し替える
        返り値は list[(word, pos, token_info)] 形式にする
    use_base_form : bool
        基本形を使うか（Janome: base_form / Sudachi: dictionary_form）
    pos_keep, pos_exclude : iterable[str], optional
        品詞（大分類）フィルタ
        ※教育用途では、まず無指定で tokenize して、後で filter_tokens_df で段階的に調整するのがおすすめ
    stopwords : iterable[str], optional
        除外語リスト（word で判定）
    extra_col : str | None
        token_info 列名。None の場合は token_info を作らない（軽量化）

    Returns
    -------
    pandas.DataFrame
        columns: [id_col, token_id_col, "word", "pos", extra_col(optional)]
    """
    if id_col not in df.columns:
        raise KeyError(
            f"tokenize_df: DataFrame に id_col={id_col!r} 列がありません。"
            f"存在する列: {list(df.columns)}"
        )
    if text_col not in df.columns:
        raise KeyError(
            f"tokenize_df: DataFrame に text_col={text_col!r} 列がありません。"
            f"存在する列: {list(df.columns)}"
        )

    if tokenize_text_fn is None:
        eng = str(engine).lower()

        if eng == "janome":
            if tokenizer is None:
                from janome.tokenizer import Tokenizer
                tokenizer = Tokenizer()

            def _fn(t: str):
                return tokenize_text_janome(
                    t,
                    tokenizer=tokenizer,
                    use_base_form=use_base_form,
                    extra_col=extra_col,
                )
            tokenize_text_fn = _fn

        elif eng == "sudachi":
            if tokenizer is None:
                from sudachipy import dictionary
                tokenizer = dictionary.Dictionary().create()

            def _fn(t: str):
                return tokenize_text_sudachi(
                    t,
                    tokenizer=tokenizer,
                    split_mode="C",
                    word_form=None,
                    use_base_form=use_base_form,
                    extra_col=extra_col,
                )
            tokenize_text_fn = _fn

        else:
            raise ValueError(f"tokenize_df: Unknown engine={engine!r}")

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        doc_id = row[id_col]
        text = row[text_col]

        tokens = tokenize_text_fn(text)
        # enumerate でインデックス (i) を取得
        for i, (word, pos, token_info) in enumerate(tokens):
            rec = {
                id_col: doc_id,
                token_id_col: i,
                "word": word,
                "pos": pos
            }
            if extra_col:
                rec[extra_col] = token_info
            records.append(rec)

    out = pd.DataFrame(records)

    # stopwords はここでは除外しない（filter_tokens_df に一本化）

    if not out.empty:
        out = filter_tokens_df(
            out,
            pos_keep=pos_keep,
            pos_exclude=pos_exclude,
            stopwords=stopwords,
            strict=True,
        )

    return out.reset_index(drop=True)


# ----------------------------------------------------------------------
# token -> text（再結合）
# ----------------------------------------------------------------------

def tokens_to_text(
    df: pd.DataFrame,
    *,
    id_col: str = "doc_id",
    word_col: str = "word",
    sep: str = " ",
    pos_keep: Optional[Iterable[str]] = None,
    pos_exclude: Optional[Iterable[str]] = None,
    stopwords=None,
    per_doc: bool = False,
    strict: bool = True,
):
    """
    token DataFrame を text に戻す。

    デフォルトでは、全トークンを結合して 1 本の大きなテキスト（str）を返す。
    これは WordCloud など「コーパス全体を 1 つの入力文字列として扱う」用途を想定している。

    per_doc=True を指定した場合は、文書IDごとに結合した DataFrame を返す。
    （columns: [id_col, "text"]）

    Parameters
    ----------
    df : pandas.DataFrame
        tokenize_df の出力（1行=1トークン）。
    id_col : str
        文書ID列名（per_doc=True のときに使用）。
    word_col : str
        トークン列名（結合対象）。
    sep : str
        トークン間の区切り文字。
    pos_keep, pos_exclude : iterable[str], optional
        品詞（大分類）フィルタ。filter_tokens_df に委譲する。
    stopwords : optional
        除外語（word_col で判定）。filter_tokens_df に委譲する。
        str / list / tuple / set / pandas.Series / pandas.Index / 入れ子を受け付ける。
    per_doc : bool, default False
        True の場合、文書ごとの text を返す。False の場合、全体を 1 本に結合した str を返す。
    strict : bool, default True
        pos_keep / pos_exclude 同時指定時の矛盾チェックを行うか。

    Returns
    -------
    str or pandas.DataFrame
        per_doc=False: str（全体を 1 本に結合）
        per_doc=True : pandas.DataFrame（文書ごとに結合）
    """
    s = filter_tokens_df(
        df,
        pos_keep=pos_keep,
        pos_exclude=pos_exclude,
        stopwords=stopwords,
        strict=strict,
    )

    if per_doc:
        return (
            s.groupby(id_col)[word_col]
            .apply(lambda xs: sep.join(xs))
            .reset_index(name="text")
        )

    # 全トークンを 1 本に結合（入力順を尊重）
    return sep.join(s[word_col].astype(str).tolist())


__all__ = [
    "tokenize_df",
    "tokenize_text_janome",
    "tokenize_text_sudachi",
    "filter_tokens_df",
    "tokens_to_text",
]
