import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback
import json

# ===== 追加（新メソッド用。既存 import は変更しない） =====
import time
from typing import Callable, Optional, Sequence, Union, Generator, Iterable, Tuple
# ===========================================================

class CorpusDB:
    def __init__(self, db_path="corpus.db", master_db_path: Optional[str] = None):
        """
        初期化: DBパスを指定し、テーブルが存在しなければ作成する。

        Parameters
        ----------
        db_path : str
            作業用DB（work）。tokens/status_tokenize を保存する。
            master_db_path が None の場合は、documents/text/status_fetch もここに作成する（一元管理）。
        master_db_path : str, optional
            参照/本文管理用DB（master）。documents/text/status_fetch を保存する（分離管理）。
        """
        self.db_path = db_path
        self.master_db_path = master_db_path
        self.master_prefix = "master." if self.master_db_path else ""
        self._create_tables()

    def _connect(self):
        """
        DB接続を行う。
        - 常に work(db_path) に接続する
        - master_db_path が指定されている場合は ATTACH する
        """
        con = sqlite3.connect(self.db_path)
        
        # 外部キー制約を有効化
        # ※ 注意: Splitモード（DBが分かれている場合）、DBを跨ぐテーブル間（tokens -> documents等）には
        #   SQLiteの仕様上 FK制約を張れないため、スキーマ定義側で FK を除外して対応している。
        #   ただし、同一DB内のテーブル間（status_fetch -> documents等）の整合性を保つため常に ON にする。
        con.execute("PRAGMA foreign_keys = ON;")
        
        if self.master_db_path:
            # パス内のシングルクォートをエスケープ（SQLインジェクション対策）
            safe_path = self.master_db_path.replace("'", "''")
            # ATTACH 名は固定で master（SQLが読みやすい）
            con.execute(f"ATTACH DATABASE '{safe_path}' AS master")
        return con

    # --- テーブル名アクセサ（SQL内の直書きを避ける） ---
    @property
    def t_docs(self) -> str:
        return f"{self.master_prefix}documents"

    @property
    def t_text(self) -> str:
        return f"{self.master_prefix}text"

    @property
    def t_status_fetch(self) -> str:
        return f"{self.master_prefix}status_fetch"

    @property
    def t_tokens(self) -> str:
        return "tokens"

    @property
    def t_status_tokenize(self) -> str:
        return "status_tokenize"

    def _create_tables(self):
        """
        テーブル初期化
        - documents: ファイルパス管理
        - status_fetch: fetch 進捗管理
        - status_tokenize: tokenize 進捗管理
        - text: 原文データ
        - tokens: 形態素解析結果
        """
        with self._connect() as con:
            # ----------------------------
            # Master側（documents/text/status_fetch）
            # - master_db_path がある場合は master.* に作成
            # - 無い場合は main（work）に作成（一元管理）
            # ----------------------------
            # 1. documents
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.t_docs} (
                    doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    abs_path TEXT UNIQUE,
                    rel_path TEXT,
                    created_at TEXT
                );
            """)

            # 2. status_fetch（fetch のみ）
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.t_status_fetch} (
                    doc_id INTEGER PRIMARY KEY,
                    fetched_at TEXT,
                    fetch_ok INTEGER DEFAULT 0,
                    error_message TEXT,
                    updated_at TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );
            """)

            # 3. text
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.t_text} (
                    doc_id INTEGER PRIMARY KEY,
                    char_count INTEGER,
                    text TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );
            """)

            # ----------------------------
            # Work側（tokens/status_tokenize）
            # ----------------------------
            # 4. tokens（work固定）
            # PRIMARY KEY (doc_id, token_id) を追加して一意性を担保
            # 分離管理では documents が master 側のため、tokens には FK を張らない（DB跨ぎFK不可）。
            if self.master_db_path:
                con.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.t_tokens} (
                        doc_id INTEGER,
                        token_id INTEGER,
                        word TEXT,
                        pos TEXT,
                        token_info TEXT,
                        PRIMARY KEY (doc_id, token_id)
                    );
                """)
            else:
                con.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.t_tokens} (
                        doc_id INTEGER,
                        token_id INTEGER,
                        word TEXT,
                        pos TEXT,
                        token_info TEXT,
                        PRIMARY KEY (doc_id, token_id),
                        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                    );
                """)

            # 5. status_tokenize（tokenize のみ / work固定）
            if self.master_db_path:
                con.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.t_status_tokenize} (
                        doc_id INTEGER PRIMARY KEY,
                        tokenize_ok INTEGER DEFAULT 0,
                        error_message TEXT,
                        updated_at TEXT
                    );
                """)
            else:
                con.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.t_status_tokenize} (
                        doc_id INTEGER PRIMARY KEY,
                        tokenize_ok INTEGER DEFAULT 0,
                        error_message TEXT,
                        updated_at TEXT,
                        FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                    );
                """)

            # インデックス作成（検索・結合の高速化用）
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_tokens_doc_id ON {self.t_tokens}(doc_id);")
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_tokens_word ON {self.t_tokens}(word);")
            
            # 【追加】品詞 (pos) 検索用インデックス
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_tokens_pos ON {self.t_tokens}(pos);")

            # status_tokenize 参照高速化（JOIN/WHERE用）
            con.execute(f"CREATE INDEX IF NOT EXISTS idx_status_tokenize_ok ON {self.t_status_tokenize}(tokenize_ok);")
            
            # status_fetch 参照高速化（JOIN/WHERE用）
            # splitモードでは status_fetch は master 側にあるため、index も master 側に作る構文が必要
            if self.master_db_path:
                # CREATE INDEX master.<index_name> ON <table_name>(...)
                con.execute("CREATE INDEX IF NOT EXISTS master.idx_status_fetch_ok ON status_fetch(fetch_ok);")
            else:
                con.execute(f"CREATE INDEX IF NOT EXISTS idx_status_fetch_ok ON {self.t_status_fetch}(fetch_ok);")

    def register_files(self, root_dir, exts=("*.txt",)):
        """
        指定ディレクトリ以下のファイルを走査し、パス情報をDBに登録する。
        （中身は読み込まないため軽量）
        
        Note:
            master_db_path が指定されている場合、Master DB (documents 等) が更新されます。
        
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
                
                # 【修正】INSERT OR IGNORE で一発登録（高速化・簡素化）
                cur = con.execute(
                    f"INSERT OR IGNORE INTO {self.t_docs} (abs_path, rel_path, created_at) VALUES (?, ?, ?)",
                    (abs_p, rel_p, now)
                )
                
                # 新規登録された場合のみ (rowcount > 0) status を初期化
                if cur.rowcount > 0:
                    # lastrowid は ATTACH あり環境で安全でないことがあるため abs_path から引き直す
                    row = con.execute(
                        f"SELECT doc_id FROM {self.t_docs} WHERE abs_path = ?",
                        (abs_p,)
                    ).fetchone()
                    if not row:
                        continue
                    doc_id = row[0]

                    # status_fetch 初期化（master/単一のどちらでも self.t_status_fetch）
                    con.execute(
                        f"INSERT OR IGNORE INTO {self.t_status_fetch} (doc_id, fetch_ok) VALUES (?, 0)",
                        (doc_id,)
                    )
                    # status_tokenize 初期化（work固定）
                    con.execute(
                        f"INSERT OR IGNORE INTO {self.t_status_tokenize} (doc_id, tokenize_ok) VALUES (?, 0)",
                        (doc_id,)
                    )
                    added_count += 1
            
            con.commit()
            
        print(f"Registered {added_count} new files. (Total found: {len(paths)})")

    def register_content_stream(
        self,
        iterator: Iterable[Tuple[str, str]],
        *,
        batch_size: int = 10000,
        source_label: str = "imported"
    ):
        """
        イテレータからデータを順次読み込み、指定バッチサイズごとにDBへ登録する（省メモリ版）。
        ※ 重複データ（同じ abs_path）があっても、正しく既存の doc_id を再利用して上書きします。
        
        Note:
            master_db_path が指定されている場合、Master DB (documents 等) が更新されます。

        Parameters
        ----------
        iterator : iterable
            (original_id, text_body) のタプルを順次返すイテレータ。
            例: sqlite3 の cursor, CSV reader, またはジェネレータ関数。
        batch_size : int
            一度にコミットする件数。メモリ使用量と速度のバランスで調整。
        source_label : str
            abs_path 生成用のプレフィックス (imported://<original_id>)。
        """
        now = datetime.now().isoformat()
        
        # バッファ初期化
        batch_buffer = []
        
        print(f"Starting stream import (batch_size={batch_size})...")
        
        # コネクションは外側で1つ維持し、バッチごとに commit する
        with self._connect() as con:
            total_processed = 0
            
            for original_id, text_body in iterator:
                # バリデーション
                # 【修正】空文字 ("") も許容し、None だけ弾くように変更
                if text_body is None or not isinstance(text_body, str):
                    continue
                
                # バッファに追加 (ID解決は flush_batch で行う)
                batch_buffer.append((original_id, text_body))
                
                # バッチサイズに達したら書き込み
                if len(batch_buffer) >= batch_size:
                    # 【修正】実際に処理された件数を加算
                    added = self._flush_batch(con, batch_buffer, source_label, now)
                    total_processed += added
                    print(f"Imported {total_processed} records...")
                    
                    # バッファクリア
                    batch_buffer.clear()

            # 残りの端数を書き込み
            if batch_buffer:
                added = self._flush_batch(con, batch_buffer, source_label, now)
                total_processed += added
                
            print(f"Stream import finished. Total: {total_processed} records.")

    def _flush_batch(self, con, batch, source_label, now):
        """内部利用: バッチ書き込み実行（ID整合性対策済み）"""
        data_text = []
        data_status = []
        
        # バッチ内の各アイテムについて、正しい doc_id を解決しながらデータを準備
        for original_id, text_body in batch:
            abs_path = f"{source_label}://{original_id}"
            
            # 1. documents テーブルへ登録 (IDは自動採番におまかせ)
            #    すでに存在する場合は IGNORE で何もしない (既存IDが維持される)
            con.execute(
                f"INSERT OR IGNORE INTO {self.t_docs} (abs_path, rel_path, created_at) VALUES (?, ?, ?)",
                (abs_path, str(original_id), now)
            )
            
            # 2. 正しい doc_id を取得 (これが最も確実)
            #    abs_path は UNIQUE なので必ず1つ特定できる
            cur = con.execute(f"SELECT doc_id FROM {self.t_docs} WHERE abs_path = ?", (abs_path,))
            row = cur.fetchone()
            
            if not row:
                continue
                
            doc_id = row[0]

            # status_fetch / status_tokenize を確実に作成（ハイブリッド統一）
            con.execute(
                f"INSERT OR IGNORE INTO {self.t_status_fetch} (doc_id, fetch_ok) VALUES (?, 0)",
                (doc_id,)
            )
            con.execute(
                f"INSERT OR IGNORE INTO {self.t_status_tokenize} (doc_id, tokenize_ok) VALUES (?, 0)",
                (doc_id,)
            )
            
            # 3. 取得した doc_id を使ってコンテンツデータを準備
            data_text.append((
                doc_id,
                len(text_body),
                text_body
            ))
            data_status.append((
                doc_id, now, 1, now  # fetched_at, fetch_ok, updated_at
            ))

        # 4. コンテンツの一括書き込み
        if data_text:
            # text/status は上書き(REPLACE)して最新状態にする
            con.executemany(
                f"INSERT OR REPLACE INTO {self.t_text} (doc_id, char_count, text) VALUES (?, ?, ?)",
                data_text
            )
            con.executemany(
                f"INSERT OR REPLACE INTO {self.t_status_fetch} (doc_id, fetched_at, fetch_ok, updated_at) VALUES (?, ?, ?, ?)",
                data_status
            )
            
            # 【追加】本文更新に伴い、tokenize_ok をリセットする（再解析待ちにする）
            doc_ids = [row[0] for row in data_text]
            ph = ",".join(["?"] * len(doc_ids))
            con.execute(
                f"""
                UPDATE {self.t_status_tokenize}
                SET tokenize_ok = 0, updated_at = ?, error_message = NULL
                WHERE doc_id IN ({ph})
                """,
                [now, *doc_ids]
            )

        con.commit()
        
        # 【修正】実際に処理した件数を返す
        return len(data_text)

    def process_queue(self, tokenize_fn=None, *, fetch_only: bool = False):
        """
        未処理のファイルを順次処理するメインループ。
        
        Parameters
        ----------
        tokenize_fn : function, optional
            fetch_only=False の場合は必須。
            pandas.DataFrame (columns=[doc_id, text]) を受け取り、
            tokens DataFrame を返す関数。
        fetch_only : bool
            True の場合、テキスト読み込みとDB保存だけ行い、形態素解析はスキップする。
        """
        if not fetch_only and tokenize_fn is None:
            raise ValueError("tokenize_fn must be provided unless fetch_only=True")

        # 未処理の doc_id を取得
        # fetch_only=True なら「未取得(fetch_ok=0)」を対象にする
        # fetch_only=False なら「未解析(tokenize_ok=0)」を対象にする（取得済み未解析も含む）
        # status_fetch / status_tokenize に分割
        if fetch_only:
            target_condition = "sf.fetch_ok = 0"
        else:
            target_condition = "st.tokenize_ok = 0"

        with self._connect() as con:
            # 【修正】imported:// などの仮想パスを対象外にするフィルタを追加
            #  process_queue はファイル読み込みを伴うため、実ファイル以外は処理してはならない
            # 【A案対応】file:// も含め、スキーム(://)を持つパスは一律除外する（シンプル化）
            if fetch_only:
                query = f"""
                    SELECT sf.doc_id 
                    FROM {self.t_status_fetch} sf
                    JOIN {self.t_docs} d ON sf.doc_id = d.doc_id
                    WHERE {target_condition}
                      AND d.abs_path NOT LIKE '%://%'
                    ORDER BY sf.doc_id
                """
            else:
                query = f"""
                    SELECT st.doc_id 
                    FROM {self.t_status_tokenize} st
                    JOIN {self.t_docs} d ON st.doc_id = d.doc_id
                    WHERE {target_condition}
                      AND d.abs_path NOT LIKE '%://%'
                    ORDER BY st.doc_id
                """
            target_ids = pd.read_sql(query, con)["doc_id"].tolist()

        total = len(target_ids)
        print(f"Processing {total} documents (fetch_only={fetch_only})...")

        for i, doc_id in enumerate(target_ids, start=1):
            try:
                self._process_one(doc_id, tokenize_fn, fetch_only=fetch_only)
                if i % 100 == 0:
                    print(f"Progress: {i}/{total} done.")
            except Exception as e:
                print(f"Error at doc_id={doc_id}: {e}")
                # エラー情報をDBに記録（fetch_only or IOError は fetch 側、その他は tokenize 側）
                stage = "fetch" if fetch_only or isinstance(e, IOError) else "tokenize"
                self._log_error(doc_id, e, stage=stage)

        print("Queue processing finished.")

    def _process_one(self, doc_id, tokenize_fn, fetch_only=False):
        """1文書に対する処理（読込 -> [解析] -> 保存）"""
        
        # 1. パスを取得（都度接続）
        with self._connect() as con:
            row = con.execute(
                f"SELECT abs_path FROM {self.t_docs} WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        
        if not row:
            return # IDが存在しない場合
            
        path_str = row[0]
        
        # ============================================================
        # 【追加】誤って process_queue を呼んだ場合のガード (Safety Hook)
        # ============================================================
        # インポート機能で登録されたパス (例: imported://...) は実ファイルではないため、
        # process_queue (ファイル読み込み) ではなく reprocess_tokens (DB内テキスト利用) を使う必要があります。
        # ※ SQL側で除外しているため通常ここには来ないが、個別に呼んだ場合などの安全策として残す
        if "://" in path_str:
            raise ValueError(
                f"Virtual path detected: '{path_str}'\n"
                "This document was imported directly from DB/DataFrame and has no physical file.\n"
                "Please use `db.reprocess_tokens()` instead of `db.process_queue()`."
            )
        # ============================================================

        # 2. ファイル読み込み
        try:
            text = Path(path_str).read_text(encoding="utf-8", errors="strict")
        except Exception as e:
            raise IOError(f"Failed to read file: {path_str}") from e

        # ===== 追加: fetch_only の場合 =====
        if fetch_only:
            with self._connect() as con:
                now = datetime.now().isoformat()
                # text テーブルへ保存
                con.execute(
                    f"INSERT OR REPLACE INTO {self.t_text} (doc_id, char_count, text) VALUES (?, ?, ?)",
                    (doc_id, len(text), text)
                )
                # status_fetch 更新 (fetch_ok=1)
                con.execute(
                    f"""
                    UPDATE {self.t_status_fetch} 
                    SET fetched_at = ?, fetch_ok = 1, updated_at = ?, error_message = NULL
                    WHERE doc_id = ?
                    """,
                    (now, now, doc_id),
                )
                con.commit()
            return
        # ===================================

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
                f"INSERT OR REPLACE INTO {self.t_text} (doc_id, char_count, text) VALUES (?, ?, ?)",
                (doc_id, len(text), text)
            )

            # tokens テーブルへ保存
            # 再実行時の重複防止：先に既存のトークンを消す（空トークン時も残留させない）
            con.execute(f"DELETE FROM {self.t_tokens} WHERE doc_id = ?", (doc_id,))
            
            if not df_tokens.empty:
                df_tokens.to_sql(
                    self.t_tokens,
                    con,
                    if_exists="append",
                    index=False,
                    method="multi", 
                    chunksize=5000
                )

            # status_fetch（fetch 成功）を更新
            con.execute(
                f"""
                UPDATE {self.t_status_fetch}
                SET fetched_at = ?, fetch_ok = 1, updated_at = ?, error_message = NULL
                WHERE doc_id = ?
                """,
                (now, now, doc_id),
            )

            # status_tokenize（tokenize 成功）を更新
            con.execute(
                f"""
                UPDATE {self.t_status_tokenize}
                SET tokenize_ok = 1, updated_at = ?, error_message = NULL
                WHERE doc_id = ?
                """,
                (now, doc_id),
            )
            
            con.commit()

    def _log_error(self, doc_id, exception, *, stage: str = "tokenize"):
        """エラー発生時のログ記録（stage: 'fetch' or 'tokenize'）"""
        msg = "".join(traceback.format_exception(None, exception, exception.__traceback__))
        now = datetime.now().isoformat()
        table = self.t_status_fetch if stage == "fetch" else self.t_status_tokenize
        with self._connect() as con:
            con.execute(
                f"UPDATE {table} SET error_message = ?, updated_at = ? WHERE doc_id = ?",
                (msg, now, doc_id)
            )
            con.commit()

    # ==================================================================
    # ここから【新規追加メソッド】のみ
    # ==================================================================

    def sync_status_from_master(self):
        """
        [Splitモード用] Master DB の情報を基に Work DB の status_tokenize を初期化する。
        既存の Master DB に対して、新規に Work DB を作成した場合などに使用する。
        """
        if not self.master_db_path:
            print("This method is for split mode (master_db_path) only.")
            return

        print("Syncing status_tokenize from master...")
        with self._connect() as con:
            # master.status_fetch にある doc_id を work.status_tokenize に登録
            con.execute(f"""
                INSERT OR IGNORE INTO {self.t_status_tokenize} (doc_id, tokenize_ok)
                SELECT doc_id, 0 FROM {self.t_status_fetch}
            """)
            con.commit()
        print("Done.")

    def reset_tokens(
        self,
        doc_ids: Optional[Sequence[Union[int, str]]] = None,
        *,
        vacuum: bool = False,
        reset_only_fetched: bool = True,
    ) -> None:
        """
        tokens を削除し、対応する status.tokenize_ok を 0 に戻す。

        Parameters
        ----------
        doc_ids : list[int|str] | None
            None の場合は全消去。指定した場合はその doc_id のみ削除/リセット。
        vacuum : bool
            True の場合、削除後に VACUUM を実行してDBファイルサイズを最適化する。
            ※ VACUUM はトランザクション外で実行される必要がある。
        reset_only_fetched : bool
            doc_ids=None の全リセット時に、fetch_ok=1 の行だけ tokenize_ok を戻すかどうか。
            True 推奨（未fetchまで巻き戻す必要があるなら False）。
        """
        now = datetime.now().isoformat()

        def _placeholders(n: int) -> str:
            return ",".join(["?"] * n)

        print("Resetting tokens and status...")

        # --- 1) DELETE tokens + reset status ---
        with self._connect() as con:
            if doc_ids is None:
                # 全消去
                con.execute(f"DELETE FROM {self.t_tokens}")

                # tokenize_ok を戻す（status_tokenize のみ）
                if reset_only_fetched:
                    # fetch_ok=1 の文書のみ対象
                    con.execute(
                        f"""
                        UPDATE {self.t_status_tokenize}
                        SET tokenize_ok = 0, updated_at = ?, error_message = NULL
                        WHERE doc_id IN (
                            SELECT doc_id FROM {self.t_status_fetch} WHERE fetch_ok = 1
                        )
                        """,
                        (now,),
                    )
                else:
                    # 全件対象
                    con.execute(
                        f"""
                        UPDATE {self.t_status_tokenize}
                        SET tokenize_ok = 0, updated_at = ?, error_message = NULL
                        """,
                        (now,),
                    )
            else:
                # 部分消去
                doc_ids = list(doc_ids)
                if len(doc_ids) == 0:
                    print("doc_ids is empty; nothing to reset.")
                    return

                ph = _placeholders(len(doc_ids))

                con.execute(f"DELETE FROM {self.t_tokens} WHERE doc_id IN ({ph})", doc_ids)

                con.execute(
                    f"""
                    UPDATE {self.t_status_tokenize}
                    SET tokenize_ok = 0, updated_at = ?, error_message = NULL
                    WHERE doc_id IN ({ph})
                    """,
                    [now, *doc_ids],
                )

            con.commit()

        # --- 2) VACUUM (optional) ---
        if vacuum:
            # VACUUM は autocommit モードの専用接続で実行
            with sqlite3.connect(self.db_path, isolation_level=None) as con_vac:
                con_vac.execute("VACUUM")

        print("Reset complete. Ready to reprocess.")

    def reprocess_tokens(
        self,
        tokenize_fn: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        doc_ids: Optional[Sequence[Union[int, str]]] = None,
        max_chars: int = 200_000,
        max_docs: int = 500,
        progress_every_batches: int = 10,
        progress_every_seconds: int = 10,
        fallback_to_single: bool = True,
    ):
        """
        tokenize_ok=0 の文書を char_count ベースで batch 再解析する。
        """

        with self._connect() as con:
            if doc_ids is None:
                df = pd.read_sql(
                    f"""
                    SELECT st.doc_id,
                           t.text,
                           t.char_count
                    FROM {self.t_status_tokenize} st
                    JOIN {self.t_status_fetch} sf ON sf.doc_id = st.doc_id
                    JOIN {self.t_text} t ON t.doc_id = st.doc_id
                    WHERE sf.fetch_ok = 1
                      AND st.tokenize_ok = 0
                    ORDER BY st.doc_id
                    """,
                    con,
                )
            else:
                doc_ids = list(doc_ids)
                if not doc_ids:
                    print("doc_ids is empty; nothing to reprocess.")
                    return
                ph = ",".join(["?"] * len(doc_ids))
                df = pd.read_sql(
                    f"""
                    SELECT st.doc_id,
                           t.text,
                           t.char_count
                    FROM {self.t_status_tokenize} st
                    JOIN {self.t_status_fetch} sf ON sf.doc_id = st.doc_id
                    JOIN {self.t_text} t ON t.doc_id = st.doc_id
                    WHERE sf.fetch_ok = 1
                      AND st.tokenize_ok = 0
                      AND st.doc_id IN ({ph})
                    ORDER BY st.doc_id
                    """,
                    con,
                    params=doc_ids,
                )

        if df.empty:
            print("No documents to reprocess.")
            return

        df["char_count"] = df["char_count"].fillna(0).astype(int)

        total_docs = len(df)
        total_chars = int(df["char_count"].sum())

        batches = self._pack_batches_by_chars(
            df, max_chars=max_chars, max_docs=max_docs
        )

        print(
            f"Starting reprocess: {total_docs} docs / {total_chars:,} chars "
            f"in {len(batches)} batches"
        )

        t0 = time.time()
        last_print = t0
        done_docs = 0
        done_chars = 0

        for bi, batch_df in enumerate(batches, start=1):
            batch_docs = len(batch_df)
            batch_chars = int(batch_df["char_count"].sum())

            try:
                self._reprocess_batch(batch_df, tokenize_fn)

            except Exception as e:
                print(f"[Batch {bi}] Error: {e}")
                if not fallback_to_single:
                    raise

                # フォールバック：ファイルを読みに行かず、batch_df に載っている text を使って 1件ずつ再処理
                for row in batch_df.itertuples(index=False):
                    # row には doc_id / text / char_count がある想定（あなたの pack の出力に合わせる）
                    df_single = pd.DataFrame([{
                        "doc_id": row.doc_id,
                        "text": row.text,
                        "char_count": getattr(row, "char_count", None),
                    }])

                    try:
                        self._reprocess_batch(df_single, tokenize_fn)
                    except Exception as e2:
                        self._log_error(row.doc_id, e2, stage="tokenize")

            done_docs += batch_docs
            done_chars += batch_chars

            now = time.time()
            if (
                bi % progress_every_batches == 0
                or (now - last_print) >= progress_every_seconds
                or bi == len(batches)
            ):
                elapsed = now - t0
                cps = done_chars / max(elapsed, 1e-9)
                eta = (total_chars - done_chars) / max(cps, 1e-9)

                print(
                    f"[{bi}/{len(batches)}] "
                    f"docs {done_docs}/{total_docs} | "
                    f"chars {done_chars:,}/{total_chars:,} | "
                    f"{cps:,.0f} ch/s | ETA {self._fmt_eta(eta)}"
                )
                last_print = now

        print("Reprocessing finished.")

    def _reprocess_batch(self, batch_df: pd.DataFrame, tokenize_fn):
        """複数文書をまとめて tokenize して保存"""

        df_input = batch_df[["doc_id", "text"]]
        df_tokens = tokenize_fn(df_input)

        if df_tokens is None or not isinstance(df_tokens, pd.DataFrame):
            raise ValueError("tokenize_fn must return DataFrame.")

        if not df_tokens.empty:
            if "doc_id" not in df_tokens.columns:
                raise ValueError("batch tokenize requires doc_id column.")

            # 【B案対応】doc_id の NA チェック（デバッグ性向上）
            if df_tokens["doc_id"].isna().any():
                raise ValueError("tokenize_fn returned NaN in 'doc_id'. Please check tokenizer output.")

            # 【追加修正】word カラム必須チェック (process_queue と同等)
            if "word" not in df_tokens.columns:
                raise ValueError("tokenize_fn must return 'word' column.")

            # 【追加修正】doc_id 整合性チェック (バッチ外のID混入防止)
            # batch_df にある ID 以外が含まれていたらエラーにする
            valid_ids = set(batch_df["doc_id"])
            if not set(df_tokens["doc_id"]).issubset(valid_ids):
                 raise ValueError("tokenize_fn returned doc_ids not in current batch.")

            if "token_id" not in df_tokens.columns:
                df_tokens["token_id"] = (
                    df_tokens.groupby("doc_id").cumcount() + 1
                )

            # 【追加修正】token_id NAチェック (明示的なエラー)
            if df_tokens["token_id"].isna().any():
                raise ValueError("token_id contains NA.")

            df_tokens["token_id"] = pd.to_numeric(
                df_tokens["token_id"], errors="raise"
            ).astype(int)

            if "token_info" in df_tokens.columns:
                df_tokens["token_info"] = df_tokens["token_info"].apply(
                    lambda x: json.dumps(x, ensure_ascii=False)
                    if isinstance(x, (dict, list))
                    else x
                )
            else:
                df_tokens["token_info"] = None

            valid_cols = ["doc_id", "token_id", "word", "pos", "token_info"]
            for c in valid_cols:
                if c not in df_tokens.columns:
                    df_tokens[c] = None
            df_tokens = df_tokens[valid_cols]

            if df_tokens.duplicated(subset=["doc_id", "token_id"]).any():
                raise ValueError("Duplicate (doc_id, token_id) detected in batch.")

        with self._connect() as con:
            ids = batch_df["doc_id"].tolist()
            ph = ",".join(["?"] * len(ids))

            con.execute(f"DELETE FROM {self.t_tokens} WHERE doc_id IN ({ph})", ids)

            if not df_tokens.empty:
                df_tokens.to_sql(
                    self.t_tokens,
                    con,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=5000,
                )

            now = datetime.now().isoformat()
            con.execute(
                f"""
                UPDATE {self.t_status_tokenize}
                SET tokenize_ok = 1, updated_at = ?, error_message = NULL
                WHERE doc_id IN ({ph})
                """,
                [now, *ids],
            )
            con.commit()

    def _pack_batches_by_chars(self, df, *, max_chars: int, max_docs: int):
        batches = []
        cur = []
        cur_chars = 0

        for row in df.itertuples(index=False):
            doc_chars = int(row.char_count)

            if cur and (
                cur_chars + doc_chars > max_chars
                or len(cur) >= max_docs
            ):
                batches.append(pd.DataFrame(cur))
                cur = []
                cur_chars = 0

            cur.append(
                {
                    "doc_id": row.doc_id,
                    "text": row.text,
                    "char_count": doc_chars,
                }
            )
            cur_chars += doc_chars

        if cur:
            batches.append(pd.DataFrame(cur))

        return batches

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h}h{m:02d}m"
        if m > 0:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def export_filtered_tokens_db(
        self,
        dst_db_path: str,
        transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        chunk_char_limit: int = 2_000_000,
        doc_batch_limit: int = 5_000,
        progress_every: int = 20,
        vacuum: bool = False,
    ) -> None:
        """
        既存の tokens に対して transform_fn（フィルタ/変換）をチャンク単位で適用し、
        派生DB（dst_db_path）を構築する。

        仕様（あなたの「仕様A」＋容量最適化）
        -----------------------------------
        - split モード（master_db_path が指定されている場合）:
            * dst は「work DB と同型」で新規作成する（tokens / status_tokenize のみ）。
            * documents / text / status_fetch は dst に入れない（容量対策）。
            * master は引き続き参照する前提（dst は work だけ持つ）。
        - 一元管理（master_db_path が None の場合）:
            * dst は現DBの完全コピーとして作り、tokens だけを作り直す。

        transform_fn の契約（固定スキーマ・token_id 維持）
        --------------------------------------------------
        - 入力: tokens の DataFrame（doc_id, token_id, word, pos, token_info を含む想定）
        - 出力: 同じスキーマ（doc_id, token_id, word, pos, token_info）を維持した DataFrame
                ※ 行を削る（フィルタ）用途を想定。列追加は非推奨（必要なら token_info に入れる）。
        - token_id は「維持」する（振り直しはしない）。
        """
        if not callable(transform_fn):
            raise TypeError("transform_fn は callable である必要があります。")

        # ------------------------------------------------------------
        # 0) dst DB を作る
        #   - split: dst を work同型（tokens/status_tokenizeのみ）で新規作成
        #   - single: dst を現DBの完全コピーとして作成（backup）
        # ------------------------------------------------------------
        dst_path = Path(dst_db_path)
        if dst_path.exists():
            dst_path.unlink()

        if self.master_db_path:
            # ---- split: dst は work-only DB として新規作成 ----
            with sqlite3.connect(dst_db_path) as con_out:
                con_out.execute("PRAGMA foreign_keys = ON;")

                # tokens（work固定スキーマ・FKなし）
                con_out.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tokens (
                        doc_id INTEGER,
                        token_id INTEGER,
                        word TEXT,
                        pos TEXT,
                        token_info TEXT,
                        PRIMARY KEY (doc_id, token_id)
                    );
                    """
                )

                # status_tokenize（work固定）
                con_out.execute(
                    """
                    CREATE TABLE IF NOT EXISTS status_tokenize (
                        doc_id INTEGER PRIMARY KEY,
                        tokenize_ok INTEGER DEFAULT 0,
                        error_message TEXT,
                        updated_at TEXT
                    );
                    """
                )

                # インデックス（work側の最低限）
                con_out.execute("CREATE INDEX IF NOT EXISTS idx_tokens_doc_id ON tokens(doc_id);")
                con_out.execute("CREATE INDEX IF NOT EXISTS idx_tokens_word ON tokens(word);")
                con_out.execute("CREATE INDEX IF NOT EXISTS idx_tokens_pos ON tokens(pos);")
                con_out.execute("CREATE INDEX IF NOT EXISTS idx_status_tokenize_ok ON status_tokenize(tokenize_ok);")

                con_out.commit()

            # dst の status_tokenize は、まず src(work) の status_tokenize を丸ごとコピーしておく
            # （doc_id 行が無いと UPDATE が効かないため）
            with sqlite3.connect(self.db_path) as con_src, sqlite3.connect(dst_db_path) as con_out:
                df_st = pd.read_sql("SELECT doc_id, tokenize_ok, error_message, updated_at FROM status_tokenize", con_src)
                if not df_st.empty:
                    df_st.to_sql("status_tokenize", con_out, if_exists="append", index=False, method="multi", chunksize=5000)
                con_out.commit()

        else:
            # ---- single: dst を現DBの完全コピーとして作成 ----
            src_con = sqlite3.connect(self.db_path)
            try:
                dst_con = sqlite3.connect(dst_db_path)
                try:
                    src_con.backup(dst_con)
                finally:
                    dst_con.close()
            finally:
                src_con.close()

        # ------------------------------------------------------------
        # 1) dst の tokens を全消去し、status_tokenize を未処理に戻す
        # ------------------------------------------------------------
        with sqlite3.connect(dst_db_path) as con_out:
            now = datetime.now().isoformat()

            con_out.execute("DELETE FROM tokens")

            # status_tokenize の行は残し、未処理に戻す
            con_out.execute(
                """
                UPDATE status_tokenize
                SET tokenize_ok = 0,
                    updated_at = ?,
                    error_message = NULL
                """,
                (now,),
            )
            con_out.commit()

        # ------------------------------------------------------------
        # 2) チャンク分割（text.char_count を使う）
        #   - split: self.t_text は master.text
        #   - single: self.t_text は text
        # ------------------------------------------------------------
        with self._connect() as con_in:
            df_docs = pd.read_sql(
                f"""
                SELECT doc_id, char_count
                FROM {self.t_text}
                ORDER BY doc_id
                """,
                con_in,
            )

        if df_docs.empty:
            return

        chunks: list[list[int]] = []
        current_ids: list[int] = []
        current_chars = 0

        for row in df_docs.itertuples(index=False):
            doc_id = int(row.doc_id)
            cc = int(row.char_count) if row.char_count is not None else 0

            # doc_id は必ず丸ごと同一チャンクに入れる（途中分割しない）
            if current_ids and (current_chars + cc > chunk_char_limit or len(current_ids) >= doc_batch_limit):
                chunks.append(current_ids)
                current_ids = []
                current_chars = 0

            current_ids.append(doc_id)
            current_chars += cc

        if current_ids:
            chunks.append(current_ids)

        # ------------------------------------------------------------
        # 3) チャンクごとに tokens を読む → transform_fn 適用 → dst に保存
        #   ※ tokens の読み取り元は「現在の work DB（self.db_path）」の main.tokens
        # ------------------------------------------------------------
        total_chunks = len(chunks)

        for ci, doc_ids in enumerate(chunks, start=1):
            t0 = time.time()
            try:
                ph = ",".join(["?"] * len(doc_ids))

                # --- 入力 tokens を読む（常に work DB の tokens） ---
                with sqlite3.connect(self.db_path) as con_tokens:
                    df_in = pd.read_sql(
                        f"""
                        SELECT doc_id, token_id, word, pos, token_info
                        FROM tokens
                        WHERE doc_id IN ({ph})
                        ORDER BY doc_id, token_id
                        """,
                        con_tokens,
                        params=doc_ids,
                    )

                # --- transform 適用 ---
                df_out = transform_fn(df_in)
                if df_out is None:
                    df_out = pd.DataFrame(columns=df_in.columns)

                if not isinstance(df_out, pd.DataFrame):
                    raise TypeError("transform_fn は pandas.DataFrame（または None）を返してください。")

                # ----------------------------------------------------
                # 品質保証（固定スキーマ・token_id維持）
                # ----------------------------------------------------
                required_cols = ["doc_id", "token_id", "word"]
                for c in required_cols:
                    if c not in df_out.columns:
                        raise ValueError(f"transform_fn の出力に必須列 {c} がありません。")

                # 余計な列は落として固定スキーマに寄せる
                valid_cols = ["doc_id", "token_id", "word", "pos", "token_info"]
                df_out = df_out[[c for c in valid_cols if c in df_out.columns]].copy()

                # チャンク外 doc_id の混入防止
                bad_ids = df_out.loc[~df_out["doc_id"].isin(doc_ids), "doc_id"].unique()
                if len(bad_ids) > 0:
                    raise ValueError(f"transform_fn 出力にチャンク外 doc_id が混入: {bad_ids[:10]}")

                # 必須列NAチェック
                if df_out[required_cols].isna().any().any():
                    raise ValueError("transform_fn 出力の必須列（doc_id/token_id/word）に NA があります。")

                # doc_id/token_id は整数化（token_idは維持前提）
                df_out["doc_id"] = df_out["doc_id"].astype(int)
                df_out["token_id"] = df_out["token_id"].astype(int)

                # (doc_id, token_id) 重複禁止
                if df_out.duplicated(subset=["doc_id", "token_id"]).any():
                    raise ValueError("transform_fn 出力で (doc_id, token_id) が重複しています。")

                # token_info の正規化（dict/list → JSON文字列）
                if "token_info" in df_out.columns:
                    df_out["token_info"] = df_out["token_info"].apply(
                        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
                    )

                # --- dst へ保存 ---
                with sqlite3.connect(dst_db_path) as con_out:
                    if not df_out.empty:
                        df_out.to_sql(
                            "tokens",
                            con_out,
                            if_exists="append",
                            index=False,
                            method="multi",
                            chunksize=5000,
                        )

                    # チャンク内 doc_id を「完了」にする
                    now = datetime.now().isoformat()
                    ph2 = ",".join(["?"] * len(doc_ids))
                    con_out.execute(
                        f"""
                        UPDATE status_tokenize
                        SET tokenize_ok = 1,
                            updated_at = ?,
                            error_message = NULL
                        WHERE doc_id IN ({ph2})
                        """,
                        [now, *doc_ids],
                    )
                    con_out.commit()

            except Exception as e:
                # チャンク単位で失敗しても、処理全体は止めずにエラーを記録する
                msg = "".join(traceback.format_exception(None, e, e.__traceback__))
                with sqlite3.connect(dst_db_path) as con_out:
                    now = datetime.now().isoformat()
                    ph2 = ",".join(["?"] * len(doc_ids))
                    con_out.execute(
                        f"""
                        UPDATE status_tokenize
                        SET tokenize_ok = 0,
                            updated_at = ?,
                            error_message = ?
                        WHERE doc_id IN ({ph2})
                        """,
                        [now, msg, *doc_ids],
                    )
                    con_out.commit()

            finally:
                if progress_every and (ci % progress_every == 0 or ci == total_chunks):
                    dt = time.time() - t0
                    print(f"[export_filtered_tokens_db] chunk {ci}/{total_chunks} 完了 ({dt:.2f}s)")

        # ------------------------------------------------------------
        # 4) 必要なら VACUUM（dst のみ）
        # ------------------------------------------------------------
        # VACUUM は autocommit モードで実行
        if vacuum:
            with sqlite3.connect(dst_db_path, isolation_level=None) as con_out:
                con_out.execute("VACUUM")

# ----------------------------------------------------------------------
# クラスの外に定義
# ----------------------------------------------------------------------

def corpus_reader(
    db_path: str,
    *,
    table_name: str = "tokens",
    id_col: str = "doc_id",
    word_col: str = "word",
    token_id_col: str = "token_id",  # ★追加: 順序保証用
    chunk_size: int = 1000
) -> Generator[list[str], None, None]:
    """
    指定されたSQLiteデータベースから文書ごとにトークン列を読み込むジェネレータ。
    メモリを節約するために、chunk_size 単位でデータを取得し、1文書ずつ yield する。

    重要: N-gram 等の文脈解析のために、必ず token_id 順でソートして取得する。

    Parameters
    ----------
    db_path : str
        SQLiteデータベースへのパス。
    table_name : str
        読み込み対象のテーブル名（既定: "tokens"）。
    id_col : str
        文書IDのカラム名（既定: "doc_id"）。
    word_col : str
        単語のカラム名（既定: "word"）。
    token_id_col : str
        トークン順序を表すカラム名（既定: "token_id"）。
    chunk_size : int
        一度に読み込む文書IDの範囲（既定: 1000）。

    Yields
    ------
    list[str]
        1文書分のトークンリスト。
    """
    with sqlite3.connect(db_path) as con:
        # 文書IDの最大値を取得して、ループの範囲を決める
        try:
            # プレースホルダはテーブル名/列名には使えないため、f-stringを使用
            max_id_df = pd.read_sql(f"SELECT MAX({id_col}) FROM {table_name}", con)
            max_id = max_id_df.iloc[0, 0]
        except Exception as e:
            print(f"corpus_reader: Error reading max id from table '{table_name}': {e}")
            return

        if max_id is None:
            return

        # chunk_size ごとに範囲を区切って読み込む
        for start_id in range(0, int(max_id) + 1, chunk_size):
            end_id = start_id + chunk_size
            
            # ★修正: ORDER BY を追加して語順を保証
            query = f"""
                SELECT {id_col}, {word_col}
                FROM {table_name}
                WHERE {id_col} >= {start_id} AND {id_col} < {end_id}
                ORDER BY {id_col}, {token_id_col}
            """
            
            try:
                df_chunk = pd.read_sql(query, con)
            except Exception:
                continue

            if df_chunk.empty:
                continue

            # 欠損除去
            df_chunk = df_chunk.dropna(subset=[word_col])
            
            if df_chunk.empty:
                continue

            # 文書IDごとにグループ化し、単語リストを作成して yield
            # データは既にSQLでソート済みなので、そのままリスト化してOK
            groups = df_chunk.groupby(id_col, sort=False)[word_col].apply(list)
            
            for tokens_list in groups:
                yield tokens_list
