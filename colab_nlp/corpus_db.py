import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback
import json

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
            # PRIMARY KEY (doc_id, token_id) を追加して一意性を担保
            con.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    doc_id INTEGER,
                    token_id INTEGER,
                    word TEXT,
                    pos TEXT,
                    token_info TEXT,
                    PRIMARY KEY (doc_id, token_id),
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );
            """)

            # インデックス作成（検索・結合の高速化用）
            con.execute("CREATE INDEX IF NOT EXISTS idx_tokens_doc_id ON tokens(doc_id);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_tokens_word ON tokens(word);")
            
            # 【追加】品詞 (pos) 検索用インデックス
            con.execute("CREATE INDEX IF NOT EXISTS idx_tokens_pos ON tokens(pos);")

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

        # 【対策1】None返却時の安全策：Noneならエラーとして扱い、ログに残す
        if df_tokens is None:
            raise ValueError(f"tokenize_fn returned None for doc_id={doc_id}. Expected DataFrame.")

        # 【追加対策】型チェック：DataFrame以外が返ってきたらエラーにする
        if not isinstance(df_tokens, pd.DataFrame):
            raise TypeError(
                f"tokenize_fn must return pandas.DataFrame, got {type(df_tokens).__name__} for doc_id={doc_id}"
            )

        if not df_tokens.empty:
            # 【追加対策】doc_id の強制上書き（関数の実装ミスによるIDズレ・欠損を防止）
            df_tokens["doc_id"] = doc_id

            # 【対策2】必須カラムチェック：token_id, word が存在し、欠損していないこと
            if "token_id" not in df_tokens.columns:
                raise ValueError(f"tokenize_fn must return 'token_id' column for doc_id={doc_id}")
            if df_tokens["token_id"].isna().any():
                raise ValueError(f"token_id contains NA for doc_id={doc_id}")
            
            if "word" not in df_tokens.columns:
                raise ValueError(f"tokenize_fn must return 'word' column for doc_id={doc_id}")

            # 【対策3】二重エンコード防止：文字列でない場合のみ json.dumps する
            if "token_info" in df_tokens.columns:
                def safe_json_dumps(x):
                    if x is None:
                        return None
                    if isinstance(x, str):
                        return x  # すでに文字列ならそのまま
                    return json.dumps(x, ensure_ascii=False)

                df_tokens["token_info"] = df_tokens["token_info"].apply(safe_json_dumps)
            else:
                df_tokens["token_info"] = None

            # 【対策4】想定外カラムの除外：テーブル定義にあるカラムだけに絞る
            valid_cols = ["doc_id", "token_id", "word", "pos", "token_info"]
            # 存在しない列は None で埋める
            for col in valid_cols:
                if col not in df_tokens.columns:
                    df_tokens[col] = None
            # 必要な列だけ抽出
            df_tokens = df_tokens[valid_cols]

            # 【対策5】重複チェック：(doc_id, token_id) が重複していたらエラーにする
            if df_tokens.duplicated(subset=["doc_id", "token_id"]).any():
                 raise ValueError(f"Duplicate token_id detected for doc_id={doc_id}.")
        
        # 5. 結果の保存（まとめてトランザクション）
        with self._connect() as con:
            now = datetime.now().isoformat()

            # text テーブルへ保存
            con.execute(
                "INSERT OR REPLACE INTO text (doc_id, char_count, text) VALUES (?, ?, ?)",
                (doc_id, len(text), text)
            )

            # tokens テーブルへ保存
            # 再実行時の重複防止：先に既存のトークンを消す（空トークン時も残留させない）
            con.execute("DELETE FROM tokens WHERE doc_id = ?", (doc_id,))
            
            if not df_tokens.empty:
                df_tokens.to_sql(
                    "tokens",
                    con,
                    if_exists="append",
                    index=False,
                    method="multi", 
                    chunksize=5000
                )

            # status 更新（完了）
            # ここまで到達できた場合のみ fetch_ok=1, tokenize_ok=1 とする
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
