import os
import logging
from gensim.models import Word2Vec
from colab_nlp import corpus_reader

# =========================================================
# 1. パス・環境設定
# =========================================================
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ディレクトリが存在しない場合は作成（念のため）
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ログ設定（全体で1つのログファイルに出力）
LOG_FILE = os.path.join(LOGS_DIR, "train_all_models.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)

# =========================================================
# 2. 学習処理の共通化
# =========================================================
def train_and_save(config):
    """
    指定された設定でWord2Vecを学習し、モデルを保存する関数
    """
    db_name = config["db_name"]
    model_name = config["model_name"]
    window_size = config["window"]
    check_words = config["check_words"]

    db_path = os.path.join(DATA_DIR, db_name)
    model_save_path = os.path.join(MODELS_DIR, model_name)

    logger.info("=" * 60)
    logger.info(f"学習開始: {model_name}")
    logger.info(f"  Input DB : {db_path}")
    logger.info(f"  Window   : {window_size}")
    logger.info("-" * 60)

    # DB存在確認
    if not os.path.exists(db_path):
        logger.error(f"DBファイルが見つかりません: {db_path}")
        return

    # データ読み込み
    sentences = corpus_reader(db_path, chunk_size=1000)

    # モデル初期化
    model = Word2Vec(
        vector_size=100,
        window=window_size,
        min_count=5,
        workers=4,
        epochs=5
    )

    # 辞書構築
    logger.info("辞書構築中...")
    model.build_vocab(sentences)
    logger.info(f"  語彙数: {len(model.wv.index_to_key)}")

    # 学習実行
    logger.info("Training中...")
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )
    logger.info("学習完了")

    # 保存
    model.save(model_save_path)
    logger.info(f"モデル保存完了: {model_save_path}")

    # 類似語チェック
    logger.info(f"[{model_name}] 類似語チェック:")
    for w in check_words:
        if w in model.wv:
            try:
                sim_words = model.wv.most_similar(w, topn=3)
                sim_str = ", ".join([f"{sw}({score:.2f})" for sw, score in sim_words])
                logger.info(f"  【{w}】 -> {sim_str}")
            except Exception as e:
                logger.warning(f"  【{w}】 エラー: {e}")
        else:
            logger.warning(f"  【{w}】 語彙に存在しません")
    
    logger.info("\n") # 改行で見やすく

# =========================================================
# 3. 実行構成（コンフィグ）
# =========================================================
# 各モデルの設定リスト
configs = [
    {
        "name": "Filtered (正規化)",
        "db_name": "work_strategies_filtered.db",
        "model_name": "w2v_filtered.model",
        "window": 5, # 正規化済みなので狭くてOK
        "check_words": ["戦略", "成長", "課題", "DX", "[NUM]"]
    },
    {
        "name": "Surface (表層形)",
        "db_name": "work_strategies_surface.db",
        "model_name": "w2v_surface.model",
        "window": 10, # 文脈を拾うため広めに
        "check_words": ["戦略", "成長", "課題", "DX", "する", "した", "において"]
    },
    {
        "name": "Hybrid (いいとこ取り)",
        "db_name": "work_strategies_hybrid.db",
        "model_name": "w2v_hybrid.model",
        "window": 10, # Surfaceを含むため広めに
        "check_words": ["戦略", "成長", "課題", "DX", "する", "ない", "[NUM]"]
    }
]

# =========================================================
# 4. メイン処理
# =========================================================
if __name__ == "__main__":
    logger.info("統合学習プロセスを開始します...")
    
    for config in configs:
        try:
            train_and_save(config)
        except Exception as e:
            logger.exception(f"学習中に予期せぬエラーが発生しました: {config['name']}")

    logger.info("全ての学習プロセスが終了しました。")