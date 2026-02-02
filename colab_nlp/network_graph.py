"""
共起ネットワーク可視化ユーティリティ（標準設定版）
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
    装飾指定を極力排除し、NetworkX のデフォルト設定で描画する。
    """
    
    # --- 1. データ準備 ---
    df_sub = df.head(head).copy()
    
    if "ngram" not in df_sub.columns or "count" not in df_sub.columns:
        raise ValueError("DataFrame must contain 'ngram' and 'count' columns.")

    try:
        split_df = df_sub["ngram"].str.split(" ", n=1, expand=True)
        df_sub["source"] = split_df[0]
        df_sub["target"] = split_df[1]
    except Exception as e:
        raise ValueError(f"Failed to split 'ngram'. Error: {e}")

    # --- 2. グラフ構築 ---
    # edge_attr="count" を追加して、グラフデータ内に頻度情報を含めます
    G = nx.from_pandas_edgelist(df_sub, source="source", target="target", edge_attr="count")

    # --- 3. レイアウト計算 ---
    # k=0.5 はノード間の距離感のバランスが良い標準的な値のため維持推奨ですが、
    # これも省いて pos = nx.spring_layout(G, seed=layout_seed) だけでも動作します。
    pos = nx.spring_layout(G, k=0.5, seed=layout_seed)

    # --- 4. フォント設定 (日本語表示に必須) ---
    font_family = "sans-serif"
    if font_path:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        font_family = prop.get_name()
    else:
        print("Warning: font_path is not specified. Japanese characters may not display correctly.")

    # --- 5. 描画設定 ---
    # エッジの太さを計算（そのままだと太すぎる場合があるため、最大5px程度に調整）
    # リスト内包表記で、エッジごとの count を取り出します
    weights = [d["count"] for u, v, d in G.edges(data=True)]
    
    if weights:
        max_weight = max(weights)
        # 最小1.0, 最大5.0 程度になるように計算 (あくまで一例です)
        widths = [(w / max_weight) * 4 + 1.0 for w in weights]
    else:
        widths = 1.0

    plt.figure(figsize=figsize)
    
    nx.draw_networkx(
        G, pos,
        with_labels=True,
        # 色・サイズ設定（落ち着いた標準的なものに固定）
        node_color="lightblue",
        edge_color="#CCCCCC",  # 薄いグレー
        alpha=0.9,
        width=widths,
        font_family=font_family
    )

    plt.title(f"Co-occurrence Network (Top {head})")
    plt.axis("off")

    # --- 6. 保存と表示 ---
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Graph saved to {save_path}")

    plt.show()
