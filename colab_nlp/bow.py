# ============================================================
# Bag-of-Words utilities (WordCloud etc.)
# ============================================================

import matplotlib.pyplot as plt

# WordCloud は optional dependency とし、未インストール時は
# 実行時に分かりやすいエラーを出す
try:
    from wordcloud import WordCloud
except ImportError as e:
    WordCloud = None


def create_wordcloud(
    sentence,
    font_path=None,
    stopwords=None,
    outfile="img.png",
    figsize=(10, 6),
    background_color="white",
    width=1200,
    height=800,
    random_state=42,
):
    """
    分かち書き済みテキストから WordCloud を生成する。

    Notes
    -----
    - font_path は、日本語フォントファイルのパスを指定すること
      （例：matplotlib 日本語フォント設定テンプレで得たパスを渡す）
    """

    if WordCloud is None:
        raise ImportError(
            "wordcloud がインストールされていません。\n"
            "Google Colab では次を実行してください:\n"
            "  !pip -q install wordcloud"
        )

    if font_path is None:
        raise ValueError(
            "font_path が未指定です。日本語フォントファイルのパスを font_path に渡してください。\n"
            "例: create_wordcloud(sentence, font_path=font_path)"
        )

    if stopwords is None:
        stopwords = set()
    else:
        stopwords = set(stopwords)

    if not sentence or not sentence.strip():
        raise ValueError("sentence が空です。tokens_to_text の結果を確認してください。")

    wcloud = WordCloud(
        background_color=background_color,
        font_path=font_path,
        stopwords=stopwords,
        width=width,
        height=height,
        random_state=random_state,
        collocations=False,
    ).generate(sentence)

    plt.figure(figsize=figsize)
    plt.imshow(wcloud)
    plt.axis("off")

    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight")

    plt.show()

    return wcloud
