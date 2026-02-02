"""
共起ネットワーク可視化ユーティリティ（オーソドックス版）
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def create_network_graph(
    df: pd.DataFrame,
    *,
    head: int = 50,
    font_path: str = None,
    layout_seed: int = 42,
    figsize: tuple = (10, 8),
    save_path: str = None,
):
    """
    N-gram DataFrame から共起ネットワーク図を作成・表示・保存する。
    独自の装飾を控え、NetworkX の標準的なスタイルで描画する。

    Parameters
    ----------
    df : pandas.DataFrame
        compute_ngram の出力結果（"ngram", "count" 列を含むこと）。
    head : int
        上位何件のエッジ（ペア）を描画するか。
    font_path : str
        日本語フォントファイルのパス（必須）。
    layout_seed : int
        レイアウト（配置）の乱数シード。
    figsize : tuple
        描画領域のサイズ (width, height)。
    save_path : str
        保存先のファイルパス。
    """
    
    # --- 1. データ準備 ---
    df_sub = df.head(head).copy()
    
    if "ngram" not in df_sub.columns or "count" not in df_sub.columns:
        raise ValueError("DataFrame must contain 'ngram' and 'count' columns.")

    # "A B" -> "source", "target" に分割
    try:
        split_df = df_sub["ngram"].str.split(" ", n=1, expand=True)
        df_sub["source"] = split_df[0]
        df_sub["target"] = split_df[1]
    except Exception as e:
        raise ValueError(f"Failed to split 'ngram'. Error: {e}")

    # --- 2. グラフ構築 ---
    G = nx.from_pandas_edgelist(df_sub, source="source", target="target")

    # --- 3. レイアウト計算 ---
    # k: ノード間の反発力（標準的なヒューリスティック設定）
    k = 0.5  
    pos = nx.spring_layout(G, k=k, seed=layout_seed)

    # --- 4. フォント設定 ---
    font_family = "sans-serif"
    if font_path:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        font_family = prop.get_name()
    else:
        print("Warning: font_path is not specified. Japanese characters may not display correctly.")

    # --- 5. 描画 (オーソドックスな設定) ---
    plt.figure(figsize=figsize)
    
    nx.draw_networkx(
        G, pos,
        with_labels=True,
        # 色・サイズ設定（落ち着いた標準的なものに固定）
        node_color="lightblue",
        edge_color="#CCCCCC",  # 薄いグレー
        node_size=600,         # 固定サイズ
        width=1.0,             # 固定幅
        alpha=0.9,
        # フォント設定
        font_family=font_family,
        font_size=11,
        font_color="black"
    )

    plt.title(f"Co-occurrence Network (Top {head})")
    plt.axis("off")

    # --- 6. 保存と表示 ---
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Graph saved to {save_path}")

    plt.show()
