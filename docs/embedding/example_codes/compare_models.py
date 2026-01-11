import os
import logging
import pandas as pd
from gensim.models import Word2Vec

# =========================================================
# 1. 設定
# =========================================================
BASE_DIR = "."
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# 比較対象のモデル定義
TARGET_MODELS = [
    {"name": "Filtered (正規化)", "path": os.path.join(MODELS_DIR, "w2v_filtered.model")},
    {"name": "Surface (表層形)", "path": os.path.join(MODELS_DIR, "w2v_surface.model")},
    {"name": "Hybrid (ハイブリッド)", "path": os.path.join(MODELS_DIR, "w2v_hybrid.model")},
]

# 分析したい単語リスト（主要な概念）
TARGET_WORDS = ["戦略", "成長", "課題", "DX", "する", "ない", "[NUM]"]

# 類似度を測りたいペア（AとBの距離）
TARGET_PAIRS = [
    ("戦略", "実行"),
    ("DX", "デジタル"),
    ("課題", "解決"),
    ("成長", "投資")
]

# ログ設定
LOG_FILE = os.path.join(LOGS_DIR, "compare_models.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", # 比較結果を見やすくするためシンプルに
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# =========================================================
# 2. 分析ロジック
# =========================================================
def load_models():
    models = {}
    for entry in TARGET_MODELS:
        name = entry["name"]
        path = entry["path"]
        if os.path.exists(path):
            try:
                models[name] = Word2Vec.load(path)
                logger.info(f"Loaded: {name}")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
        else:
            logger.warning(f"File not found: {path}")
    return models

def compare_similar_words(models, target_words, topn=10):
    logger.info("\n" + "="*80)
    logger.info(f"【類似語ランキング比較 (Top {topn})】")
    logger.info("="*80)

    for word in target_words:
        logger.info(f"\nTarget Word: 【{word}】")
        
        # 結果を格納するDataFrame用の辞書
        result_dict = {}
        
        for name, model in models.items():
            if word in model.wv:
                sim_words = model.wv.most_similar(word, topn=topn)
                # "単語 (0.123)" の形式にする
                result_dict[name] = [f"{w} ({s:.3f})" for w, s in sim_words]
            else:
                result_dict[name] = ["(辞書になし)"] * topn
        
        # DataFrameにして見やすく表示
        # 行数が合わない（辞書なし等）場合のケア
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in result_dict.items()]))
        
        # Markdown形式のようなテーブルとしてログ出力
        logger.info(df.to_string(index=True))

def compare_pairs(models, target_pairs):
    logger.info("\n" + "="*80)
    logger.info("【単語ペア類似度比較 (Cosine Similarity)】")
    logger.info("="*80)
    
    # ヘッダー
    header = f"{'Pair':<20} | " + " | ".join([f"{m['name']:<15}" for m in TARGET_MODELS])
    logger.info(header)
    logger.info("-" * len(header))

    for w1, w2 in target_pairs:
        row_str = f"{w1} - {w2:<15} | "
        
        for entry in TARGET_MODELS:
            name = entry["name"]
            if name in models:
                model = models[name]
                if w1 in model.wv and w2 in model.wv:
                    score = model.wv.similarity(w1, w2)
                    row_str += f"{score:.4f}".center(15) + " | "
                else:
                    row_str += "Missing".center(15) + " | "
            else:
                row_str += "---".center(15) + " | "
        
        logger.info(row_str)

# =========================================================
# 3. 実行
# =========================================================
if __name__ == "__main__":
    logger.info("モデル比較分析を開始します...")
    
    # 1. モデル読み込み
    loaded_models = load_models()
    
    if not loaded_models:
        logger.error("有効なモデルが読み込めませんでした。終了します。")
        exit()

    # 2. 類似語ランキング比較
    compare_similar_words(loaded_models, TARGET_WORDS)

    # 3. ペア類似度比較
    compare_pairs(loaded_models, TARGET_PAIRS)

    logger.info("\n" + "="*80)
    logger.info(f"比較完了。詳細は {LOG_FILE} を確認してください。")