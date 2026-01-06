"""
N-gram 集計ユーティリティ

主な機能:
- compute_ngram: トークン列（DataFrame または ジェネレータ）からN-gram頻度を算出する
"""

from __future__ import annotations
from typing import Union, Iterable, Optional, Generator
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def compute_ngram(
    data: Union[pd.DataFrame, Iterable[list[str]]],
    *,
    n: int = 2,
    min_count: int = 1,
    col_doc: str = "doc_id",
    col_word: str = "word",
    top_k: Optional[int] = None
) -> pd.DataFrame:
    """
    トークン列から N-gram 頻度を集計する。

    入力 (data) は以下の2パターンに対応:
    1. pandas.DataFrame: 形態素解析済みの縦持ちデータ（col_doc, col_word 列が必要）
    2. Iterable[list[str]]: トークンリストのイテレータ（大規模データ/corpus_reader用）

    Parameters
    ----------
    data : pd.DataFrame or Iterable[list[str]]
        入力データ。
    n : int
        N-gramのN（2=Bigram, 3=Trigram...）。デフォルトは 2。
    min_count : int
        最小出現回数。これ未満のN-gramは除外する。
    col_doc : str
        文書ID列名（DataFrame入力時のみ使用）。
    col_word : str
        単語列名（DataFrame入力時のみ使用）。
    top_k : int, optional
        頻度上位 k 件のみを返す。Noneの場合は全件返す。

    Returns
    -------
    pandas.DataFrame
        columns: ["ngram", "count"]
        頻度の降順でソート済み。
    """
    
    # --- 1. 入力データの正規化 ---
    input_iterable = data
    
    if isinstance(data, pd.DataFrame):
        # 必要な列があるか確認
        if col_doc not in data.columns or col_word not in data.columns:
            raise KeyError(f"compute_ngram: DataFrame must contain columns: '{col_doc}', '{col_word}'")
            
        # 欠損除去
        df_clean = data.dropna(subset=[col_word])
        if df_clean.empty:
            return pd.DataFrame(columns=["ngram", "count"])
            
        # 文書ごとにリスト化
        input_iterable = df_clean.groupby(col_doc)[col_word].apply(list)

    # --- 2. Sklearn CountVectorizer の設定 ---
    # リストをそのまま受け取り、N-gram生成機能だけを利用するための設定
    def identity(x):
        return x

    cv = CountVectorizer(
        preprocessor=identity,
        tokenizer=identity,
        token_pattern=None,
        ngram_range=(n, n),  # 指定した N のみ抽出
    )

    # --- 3. 集計実行 ---
    try:
        # data がジェネレータの場合、ここで消費される
        X = cv.fit_transform(input_iterable)
    except ValueError:
        # 語彙が生成されなかった場合（空データや、単語数がN未満の文書のみの場合など）
        return pd.DataFrame(columns=["ngram", "count"])

    # --- 4. 結果の整形 ---
    # vocab は "単語1 単語2" のようなスペース区切りの文字列になる
    vocab = cv.get_feature_names_out()
    counts = X.sum(axis=0).A1

    df_ngram = pd.DataFrame({
        "ngram": vocab,
        "count": counts
    })

    # --- 5. フィルタリングとソート ---
    if min_count > 1:
        df_ngram = df_ngram[df_ngram["count"] >= min_count]

    df_ngram = df_ngram.sort_values("count", ascending=False).reset_index(drop=True)

    if top_k is not None:
        df_ngram = df_ngram.head(top_k)

    return df_ngram
