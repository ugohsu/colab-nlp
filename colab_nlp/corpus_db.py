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
                
                # 【修正】INSERT OR IGNORE で一発登録（高速化・簡素化）
                cur = con.execute(
                    "INSERT OR IGNORE INTO documents (abs_path, rel_path, created_at) VALUES (?, ?, ?)",
                    (abs_p, rel_p, now)
                )
                
                # 新規登録された場合のみ (rowcount > 0) status を初期化
                if cur.rowcount > 0:
                    doc_id = cur.lastrowid
                    con.execute(
                        "INSERT INTO status (doc_id, fetch_ok, tokenize_ok) VALUES (?, 0, 0)",
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
                    self._flush_batch(con, batch_buffer, source_label, now)
                    total_processed += len(batch_buffer)
                    print(f"Imported {total_processed} records...")
                    
                    # バッファクリア
                    batch_buffer.clear()

            # 残りの端数を書き込み
            if batch_buffer:
                self._flush_batch(con, batch_buffer, source_label, now)
                total_processed += len(batch_buffer)
                
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
                "INSERT OR IGNORE INTO documents (abs_path, rel_path, created_at) VALUES (?, ?, ?)",
                (abs_path, str(original_id), now)
            )
            
            # 2. 正しい doc_id を取得 (これが最も確実)
            #    abs_path は UNIQUE なので必ず1つ特定できる
            cur = con.execute("SELECT doc_id FROM documents WHERE abs_path = ?", (abs_path,))
            row = cur.fetchone()
            
            if not row:
                continue
                
            doc_id = row[0]
            
            # 3. 取得した doc_id を使ってコンテンツデータを準備
            data_text.append((
                doc_id,
                len(text_body),
                text_body
            ))
            data_status.append((
                doc_id,
                now, 1, 0, now  # fetch_ok=1, tokenize_ok=0
            ))

        # 4. コンテンツの一括書き込み
        #    text/status は上書き(REPLACE)して最新状態にする
        con.executemany(
            "INSERT OR REPLACE INTO text (doc_id, char_count, text) VALUES (?, ?, ?)",
            data_text
        )
        con.executemany(
            "INSERT OR REPLACE INTO status (doc_id, fetched_at, fetch_ok, tokenize_ok, updated_at) VALUES (?, ?, ?, ?, ?)",
            data_status
        )
        con.commit()

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
        # 【修正】JOIN 用にエイリアス s を付与
        target_condition = "s.fetch_ok = 0" if fetch_only else "s.tokenize_ok = 0"

        with self._connect() as con:
            # 【修正】imported:// などの仮想パスを対象外にするフィルタを追加
            #  process_queue はファイル読み込みを伴うため、実ファイル以外は処理してはならない
            query = f"""
                SELECT s.doc_id 
                FROM status s
                JOIN documents d ON s.doc_id = d.doc_id
                WHERE {target_condition}
                  AND (d.abs_path NOT LIKE '%://%' OR d.abs_path LIKE 'file://%')
                ORDER BY s.doc_id
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
                # エラー情報をDBに記録
                self._log_error(doc_id, e)

        print("Queue processing finished.")

    def _process_one(self, doc_id, tokenize_fn, fetch_only=False):
        """1文書に対する処理（読込 -> [解析] -> 保存）"""
        
        # 1. パスを取得（都度接続）
        with self._connect() as con:
            row = con.execute(
                "SELECT abs_path FROM documents WHERE doc_id = ?", (doc_id,)
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
        if "://" in path_str and not path_str.startswith("file://"):
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
                    "INSERT OR REPLACE INTO text (doc_id, char_count, text) VALUES (?, ?, ?)",
                    (doc_id, len(text), text)
                )
                # status 更新 (fetch_ok=1, tokenize_ok=0)
                con.execute("""
                    UPDATE status 
                    SET fetched_at = ?, fetch_ok = 1, updated_at = ?, error_message = NULL
                    WHERE doc_id = ?
                """, (now, now, doc_id))
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

    # ==================================================================
    # ここから【新規追加メソッド】のみ
    # ==================================================================

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
                con.execute("DELETE FROM tokens")

                # tokenize_ok を戻す範囲
                if reset_only_fetched:
                    con.execute(
                        """
                        UPDATE status
                        SET tokenize_ok = 0, updated_at = ?, error_message = NULL
                        WHERE fetch_ok = 1
                        """,
                        (now,),
                    )
                else:
                    con.execute(
                        """
                        UPDATE status
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

                con.execute(f"DELETE FROM tokens WHERE doc_id IN ({ph})", doc_ids)

                con.execute(
                    f"""
                    UPDATE status
                    SET tokenize_ok = 0, updated_at = ?, error_message = NULL
                    WHERE doc_id IN ({ph})
                    """,
                    [now, *doc_ids],
                )

            con.commit()

        # --- 2) VACUUM (optional) ---
        if vacuum:
            # VACUUM はトランザクション外で実行する必要がある
            with self._connect() as con:
                con.execute("VACUUM")
                con.commit()

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
                    """
                    SELECT s.doc_id,
                           t.text,
                           t.char_count
                    FROM status s
                    JOIN text t ON t.doc_id = s.doc_id
                    WHERE s.fetch_ok = 1 AND s.tokenize_ok = 0
                    ORDER BY s.doc_id
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
                    SELECT s.doc_id,
                           t.text,
                           t.char_count
                    FROM status s
                    JOIN text t ON t.doc_id = s.doc_id
                    WHERE s.fetch_ok = 1 AND s.tokenize_ok = 0
                      AND s.doc_id IN ({ph})
                    ORDER BY s.doc_id
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
                        self._log_error(row.doc_id, e2)

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

            con.execute(f"DELETE FROM tokens WHERE doc_id IN ({ph})", ids)

            if not df_tokens.empty:
                df_tokens.to_sql(
                    "tokens",
                    con,
                    if_exists="append",
                    index=False,
                    method="multi",
                    chunksize=5000,
                )

            now = datetime.now().isoformat()
            con.execute(
                f"""
                UPDATE status
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