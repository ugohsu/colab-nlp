import os
import logging
import pandas as pd
import json
import re
from colab_nlp import CorpusDB, filter_tokens_df

# =========================================================
# 1. パス・環境設定
# =========================================================
BASE_DIR = "."
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ディレクトリ作成
for d in [DATA_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ログ設定
LOG_FILE = os.path.join(LOGS_DIR, "create_all_dbs.log")

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

# パス定義
MASTER_DB_PATH = os.path.join(DATA_DIR, "master_strategies.db")
SRC_DB_PATH    = os.path.join(DATA_DIR, "work_strategies_full.db")

# 出力先定義
DST_FILTERED = os.path.join(DATA_DIR, "work_strategies_filtered.db")
DST_SURFACE  = os.path.join(DATA_DIR, "work_strategies_surface.db")
DST_HYBRID   = os.path.join(DATA_DIR, "work_strategies_hybrid.db")

# =========================================================
# 2. 変換ロジック定義
# =========================================================

# --- A. Filtered Logic (正規化・選別重視) ---
def logic_filtered(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 品詞フィルタ
    df = filter_tokens_df(
        df,
        pos_exclude=["空白", "補助記号", "記号"],
        strict=False
    )
    if df.empty: return df

    # 2. [NUM]化
    pattern_num = r'^\d+(\.\d+)?$'
    is_number = df["word"].astype(str).str.match(pattern_num)
    if is_number.any():
        df.loc[is_number, "word"] = "[NUM]"

    # 3. ノイズ除去
    pattern_junk = r'^([!-/:-@[-`{-~]+|[ぁ-ん])$'
    df = df[~df["word"].astype(str).str.match(pattern_junk)]
    
    return df

# --- B. Surface Logic (表層形・全残し) ---
def logic_surface(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Surfaceへの書き換え
    def extract_surface(row):
        try:
            info = json.loads(row["token_info"])
            return info.get("surface", row["word"])
        except:
            return row["word"]

    df["word"] = df.apply(extract_surface, axis=1)

    # 2. フィルタ（空白のみ削除）
    df = filter_tokens_df(
        df,
        pos_exclude=["空白"],
        strict=False
    )
    return df

# --- C. Hybrid Logic (いいとこ取り) ---
def logic_hybrid(df: pd.DataFrame) -> pd.DataFrame:
    POS_NORM = ["名詞", "形状詞", "接尾辞", "接頭辞"]
    POS_STOP = ["空白", "補助記号", "記号", "感動詞"]

    # 1. ストップ品詞除去
    df = filter_tokens_df(df, pos_exclude=POS_STOP, strict=False)
    if df.empty: return df

    # 2. Surface適用
    mask_surface = ~df["pos"].isin(POS_NORM)
    if mask_surface.any():
        def extract_surface(row):
            try:
                if isinstance(row["token_info"], str):
                    info = json.loads(row["token_info"])
                    return info.get("surface", row["word"])
            except: pass
            return row["word"]
        
        df.loc[mask_surface, "word"] = df[mask_surface].apply(extract_surface, axis=1)

    # 3. [NUM]化
    pattern_num = r'^\d+(\.\d+)?$'
    is_number = df["word"].astype(str).str.match(pattern_num)
    if is_number.any():
        df.loc[is_number, "word"] = "[NUM]"

    return df

# =========================================================
# 3. 実行関数
# =========================================================
def run_export(task_name, dst_path, logic_fn):
    logger.info("=" * 60)
    logger.info(f"DB作成開始: {task_name}")
    logger.info(f"  Output: {dst_path}")
    logger.info("-" * 60)
    
    # 既存ファイルのクリーンアップ（CorpusDB側でもやるが念のため）
    if os.path.exists(dst_path):
        logger.info(f"既存ファイルを削除します: {dst_path}")
        try:
            os.remove(dst_path)
        except OSError as e:
            logger.warning(f"削除失敗（上書きされます）: {e}")

    # DB接続
    try:
        src_db = CorpusDB(db_path=SRC_DB_PATH, master_db_path=MASTER_DB_PATH)
        
        # エクスポート実行
        src_db.export_filtered_tokens_db(
            dst_db_path=dst_path,
            transform_fn=logic_fn,
            chunk_char_limit=1_000_000,
            progress_every=10,
            vacuum=True
        )
        logger.info(f"完了: {task_name}")
        
    except Exception as e:
        logger.exception(f"エラー発生: {task_name}")
        raise e

# =========================================================
# 4. メイン処理
# =========================================================
if __name__ == "__main__":
    logger.info("統合DB作成プロセスを開始します...")
    
    if not os.path.exists(SRC_DB_PATH):
        logger.error(f"入力DBが見つかりません: {SRC_DB_PATH}")
        exit(1)

    try:
        # 1. Filtered
        run_export("Filtered (正規化)", DST_FILTERED, logic_filtered)
        
        # 2. Surface
        run_export("Surface (表層形)", DST_SURFACE, logic_surface)
        
        # 3. Hybrid
        run_export("Hybrid (ハイブリッド)", DST_HYBRID, logic_hybrid)
        
        logger.info("=" * 60)
        logger.info("全てのデータベース作成が正常に完了しました。")
        
    except Exception as e:
        logger.error("プロセスが中断されました。")