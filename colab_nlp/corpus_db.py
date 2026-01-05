import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback

class CorpusDB:
    def __init__(self, db_path="corpus.db"):
        """
        初期化: DBパスを指定し、テーブルが存在しなければ作成する。
        """
        self.db_path = db_path
        self._create_tables()

    def _connect(self):
        """DB接続を行い、外部キー制約を有効化して返す"""
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA foreign_keys = ON;")
        return con

    def _create_tables(self):
        """
        テーブル初期化
        - documents: ファイルパス管理
        - status: 進捗管理
        - text: 原文データ
        - tokens: 形態素解析結果
        """
        with self._connect() as con:
            # 1. documents
            con.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    abs_path TEXT UNIQUE,
                    rel_path TEXT,
                    created_at TEXT
                );
            """)

            # 2. status
            con.execute("""
                CREATE TABLE IF NOT EXISTS status (
                    doc_id INTEGER PRIMARY KEY,
                    fetched_at TEXT,
                    fetch_ok INTEGER DEFAULT 0,
                    tokenize_ok INTEGER DEFAULT 0,
                    error_message TEXT,
                    updated_at TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );
            """)

            # 3. text
            con.execute("""
                CREATE TABLE IF NOT EXISTS text (
                    doc_id INTEGER PRIMARY KEY,
                    char_count INTEGER,
                    text TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );
            """)

            # 4. tokens
            con.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    doc_id INTEGER,
                    token_id INTEGER,
                    word TEXT,
                    pos TEXT,
                    token_info TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );
            """)

            # インデックス作成（検索・結合の高速化用）
            con.execute("CREATE INDEX IF NOT EXISTS idx_tokens_doc_id ON tokens(doc_id);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_tokens_word ON tokens(word);")

    def register_files(self, root_dir, exts=("*.txt",)):
        """
        指定ディレクトリ以下のファイルを走査し、パス情報をDBに登録する。
        （中身は読み込まないため軽量）
        
        Parameters
        ----------
        root_dir : str or Path
            走査対象のルートディレクトリ
        exts : tuple or list
            対象拡張子（例: ("*.txt", "*.md")）
        """
        root = Path(root_dir)
        if isinstance(exts, str):
            exts = [exts]
        
        print(f"Scanning files in {root} ...")
        
        # パス一覧を取得
        paths = []
        for pattern in exts:
            paths.extend(root.rglob(pattern))
        
        # ソートして順序を固定（再現性のため）
        paths = sorted(paths, key=str)
        
        now = datetime.now().isoformat()
        added_count = 0

        with self._connect() as con:
            for p in paths:
                abs_p = str(p.resolve())
                rel_p = str(p.relative_to(root))
                
                # 重複チェック（abs_path が UNIQUE）
                exists = con.execute(
                    "SELECT 1 FROM documents WHERE abs_path = ?", (abs_p,)
                ).fetchone()
                
                if not exists:
                    try:
                        cur = con.execute(
                            "INSERT INTO documents (abs_path, rel_path, created_at) VALUES (?, ?, ?)",
                            (abs_p, rel_p, now)
                        )
                        doc_id = cur.lastrowid
                        
                        # status テーブルにも初期レコードを作成
                        con.execute(
                            "INSERT INTO status (doc_id, fetch_ok, tokenize_ok) VALUES (?, 0, 0)",
                            (doc_id,)
                        )
                        added_count += 1
                    except sqlite3.IntegrityError:
                        continue
            con.commit()
            
        print(f"Registered {added_count} new files. (Total found: {len(paths)})")

    def process_queue(self, tokenize_fn):
        """
        未処理のファイルを順次処理するメインループ。
        
        Parameters
        ----------
        tokenize_fn : function
            pandas.DataFrame (columns=[doc_id, text]) を受け取り、
            tokens DataFrame を返す関数。
        """
        # 未処理（tokenize_ok=0）の doc_id を取得
        with self._connect() as con:
            target_ids = pd.read_sql(
                "SELECT doc_id FROM status WHERE tokenize_ok = 0 ORDER BY doc_id", 
                con
            )["doc_id"].tolist()

        total = len(target_ids)
        print(f"Processing {total} documents...")

        for i, doc_id in enumerate(target_ids, start=1):
            try:
                self._process_one(doc_id, tokenize_fn)
                if i % 100 == 0:
                    print(f"Progress: {i}/{total} done.")
            except Exception as e:
                print(f"Error at doc_id={doc_id}: {e}")
                # エラー情報をDBに記録
                self._log_error(doc_id, e)

        print("Queue processing finished.")

    def _process_one(self, doc_id, tokenize_fn):
        """1文書に対する処理（読込 -> 解析 -> 保存）"""
        
        # 1. パスを取得（都度接続）
        with self._connect() as con:
            row = con.execute(
                "SELECT abs_path FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        
        if not row:
            return # IDが存在しない場合
            
        path_str = row[0]
        
        # 2. ファイル読み込み
        try:
            text = Path(path_str).read_text(encoding="utf-8", errors="replace")
            fetch_ok = 1
        except Exception as e:
            raise IOError(f"Failed to read file: {path_str}") from e

        # 3. 形態素解析用 DataFrame 作成
        # colab-nlp の tokenize_df は 'text' 列と ID列 を期待する
        df_input = pd.DataFrame([{
            "doc_id": doc_id,
            "text": text
        }])

        # 4. 解析実行（外部関数）
        df_tokens = tokenize_fn(df_input)
        
        # 5. 結果の保存（まとめてトランザクション）
        with self._connect() as con:
            now = datetime.now().isoformat()

            # text テーブルへ保存
            con.execute(
                "INSERT OR REPLACE INTO text (doc_id, char_count, text) VALUES (?, ?, ?)",
                (doc_id, len(text), text)
            )

            # tokens テーブルへ保存
            if not df_tokens.empty:
                # 必要なカラムが存在するか確認しつつ保存
                df_tokens.to_sql(
                    "tokens",
                    con,
                    if_exists="append",
                    index=False,
                    method="multi", # 複数行一括挿入で高速化
                    chunksize=5000  # メモリ溢れ防止
                )

            # status 更新（完了）
            con.execute("""
                UPDATE status 
                SET fetched_at = ?, fetch_ok = 1, tokenize_ok = 1, updated_at = ?, error_message = NULL
                WHERE doc_id = ?
            """, (now, now, doc_id))
            
            con.commit()

    def _log_error(self, doc_id, exception):
        """エラー発生時のログ記録"""
        msg = "".join(traceback.format_exception(None, exception, exception.__traceback__))
        now = datetime.now().isoformat()
        with self._connect() as con:
            con.execute(
                "UPDATE status SET error_message = ?, updated_at = ? WHERE doc_id = ?",
                (msg, now, doc_id)
            )
            con.commit()
