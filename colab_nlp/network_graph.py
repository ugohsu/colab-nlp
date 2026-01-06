"""
共起ネットワーク可視化ユーティリティ
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
    figsize: tuple = (12, 12),
    node_color: str = "cyan",
    edge_color: str = "grey",
    save_path: str = None,
):
    """
    N-gram DataFrame から共起ネットワーク図を作成・表示・保存する。

    Parameters
    ----------
    df : pandas.DataFrame
        compute_ngram の出力結果（"ngram", "count" 列を含むこと）。
    head : int
        上位何件のエッジ（ペア）を描画するか。
        多すぎると図が真っ黒になるため、50〜100程度を推奨。
    font_path : str
        日本語フォントファイルのパス（必須）。
    layout_seed : int
        レイアウト（配置）の乱数シード。同じ値なら毎回同じ配置になる。
    figsize : tuple
        描画領域のサイズ (width, height)。
    node_color : str
        ノードの色。
    edge_color : str
        エッジの色。
    save_path : str
        保存先のファイルパス（例: "network.png"）。None の場合は保存しない。
    """
    
    # 1. 前処理: データの絞り込みと分割
    #    上位 head 件に絞る
    df_sub = df.head(head).copy()
    
    #    "ngram" 列 ("A B") を "source", "target" に分割
    if "ngram" not in df_sub.columns or "count" not in df_sub.columns:
        raise ValueError("DataFrame must contain 'ngram' and 'count' columns.")

    #    スペースで分割（n=1 で最初のスペースだけ分割、3-gram以上は "A" と "B C" のようになる）
    #    基本は Bigram (2-gram) を想定
    try:
        split_df = df_sub["ngram"].str.split(" ", n=1, expand=True)
        df_sub["source"] = split_df[0]
        df_sub["target"] = split_df[1]
    except Exception as e:
        raise ValueError(f"Failed to split 'ngram' column. Ensure it contains space-separated words. Error: {e}")

    # 2. グラフの構築 (NetworkX)
    G = nx.from_pandas_edgelist(
        df_sub,
        source="source",
        target="target",
        edge_attr="count"
    )

    # 3. レイアウト計算 (Spring Layout)
    #    k: ノード間の反発力（大きいほど広がる）。ノード数の平方根の逆数が目安とされる
    k = 1.0 / (len(G.nodes()) ** 0.5) if len(G.nodes()) > 0 else 0.5
    pos = nx.spring_layout(G, k=k, seed=layout_seed)

    # 4. 可視化設定 (Matplotlib)
    plt.figure(figsize=figsize)
    
    # フォント設定（日本語対応）
    prop = None
    if font_path:
        # フォントマネージャに追加＆プロパティ作成
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams["font.family"] = prop.get_name()
    else:
        raise ValueError("font_path must be specified for Japanese text.")

    # ノードサイズ: 次数（つながりの数）に比例させる
    # 次数が 1 のノードも最低限の大きさ (300) を確保し、次数倍する
    d = dict(G.degree)
    node_sizes = [v * 300 for v in d.values()]

    # エッジの太さ: count（共起回数）に比例させる
    # そのままだと太すぎる場合があるため、最大値で正規化するなどの調整を入れる
    # ここではシンプルに「logをとる」あるいは「定数倍」などが一般的だが、
    # わかりやすさ優先で count の値をスケーリングする（最大幅を 10 とする）
    max_count = df_sub["count"].max()
    edge_widths = [d["count"] / max_count * 10 for u, v, d in G.edges(data=True)]

    # 5. 描画
    # ノード
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        node_color=node_color, 
        alpha=0.8
    )
    
    # エッジ
    nx.draw_networkx_edges(
        G, pos, 
        width=edge_widths, 
        edge_color=edge_color, 
        alpha=0.6
    )
    
    # ラベル
    nx.draw_networkx_labels(
        G, pos, 
        font_family=prop.get_name() if prop else "sans-serif",
        font_size=12,
        font_color="black"
    )

    plt.title(f"Co-occurrence Network (Top {head} edges)", fontsize=16)
    plt.axis("off")  # 軸を消す

    # 6. 保存と表示
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Graph saved to {save_path}")

    plt.show()
