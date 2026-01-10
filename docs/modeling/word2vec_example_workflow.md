# Word2Vec モデル構築の実践例（Example Workflow）

本ドキュメントでは、特定のドメイン文書（例: 企業の戦略文書）から Word2Vec モデルを構築する際の一連のワークフローを**ケーススタディ**として紹介します。

ここでは、**「表記揺れの吸収（正規化）」と「文脈のニュアンス保持（表層形）」のバランスを取るために試行錯誤したプロセスと、その結果採用したハイブリッドなアプローチ**を例示します。皆様のプロジェクトに応用する際の参考としてご利用ください。

## 0. 本リポジトリの設計思想（Concept）

このワークフローは、本リポジトリ特有のライブラリ `CorpusDB` の機能を前提に設計されています。

1.  **データの分離管理（Separation Management）**:
    * **Master DB**: 原文やメタデータ（不変）
    * **Work DB**: 解析済みのトークン列（可変・実験用）
    * これらを物理的に分けることで、原本を保護しつつ、軽量なトークンDBで高速に試行錯誤を繰り返すことができます。
2.  形態素解析における **「Full First, Filter Later」戦略**:
    * 最初に可能な限り詳細な情報を含んだ「フルセット版トークンDB (`full`)」を作成します。
    * その後の実験（正規化やストップワード除去）は、この `full` DB からの「フィルタリング（抽出）」として行います。

## 1. 本例のディレクトリ構造

本例では、スクリプトをルートに置き、データやログを専用ディレクトリに整理する構成を採用しています。

```text
.
├── create_all_dbs.py      # ステップ1: データベース作成（前処理）
├── train_all_models.py    # ステップ2: 学習実行
├── compare_models.py      # ステップ3: 比較・評価
├── data/                  # データ格納用
│   ├── master_strategies.db      # 原本データ
│   ├── work_strategies_full.db   # ソースデータ（解析済み全量）
│   ├── work_strategies_filtered.db
│   ├── work_strategies_surface.db
│   └── work_strategies_hybrid.db
├── models/                # 学習済みモデル（.model）出力先
└── logs/                  # 実行ログ出力先

```

---

## ステップ1: データベースの作成（前処理の比較検討）

前処理の方針によってモデルの性質が大きく変わるため、本例ではあえて **3つの異なるパターン** のデータベースを作成し、比較検証を行いました。

### 検討した3つのパターン

| パターン名 | 方針・特徴 | 想定されるメリット・デメリット |
| :--- | :--- | :--- |
| **Filtered (正規化)** | 名詞・動詞などをすべて正規化形に変換し、数値や記号を削除。 | **メリット**: 表記揺れ（例: 引越/引越し）が統一され、語彙が凝縮される。<br>**デメリット**: 「する/した」の区別など、細かい文脈が消える。 |
| **Surface (表層形)** | 原文の表現（表層形）を維持。空白以外は基本的に残す。 | **メリット**: 文章の自然な流れやニュアンスを保持できる。<br>**デメリット**: 表記揺れでデータが分散しやすい（例: DX/dx）。 |
| **Hybrid (折衷案)** | **名詞は正規化、動詞は表層形、数値は[NUM]タグ化**して組み合わせる。 | 上記2つのいいとこ取りを狙った戦略。<br>※今回のデータセットでは最も良好な結果となりました。 |

### 実装例: [`create_all_dbs.py`](./example_codes/create_all_dbs.py)

```python

# ... (ライブラリインポート・パス設定・ログ設定は省略) ...

# =========================================================
# 変換ロジックの定義例
# =========================================================

# パターンA: 正規化重視
def logic_filtered(df: pd.DataFrame) -> pd.DataFrame:
    # 記号や空白を除去
    df = filter_tokens_df(df, pos_exclude=["空白", "補助記号", "記号"], strict=False)
    if df.empty: return df

    # 数値を [NUM] に置換（抽象化）
    pattern_num = r'^\d+(\.\d+)?$'
    is_number = df["word"].astype(str).str.match(pattern_num)
    if is_number.any():
        df.loc[is_number, "word"] = "[NUM]"

    # その他ノイズ除去
    pattern_junk = r'^([!-/:-@[-`{-~]+|[ぁ-ん])$'
    df = df[~df["word"].astype(str).str.match(pattern_junk)]
    return df

# パターンB: 表層形重視
def logic_surface(df: pd.DataFrame) -> pd.DataFrame:
    # Token情報からSurface（表層形）を取り出して上書き
    def extract_surface(row):
        try:
            info = json.loads(row["token_info"])
            return info.get("surface", row["word"])
        except: return row["word"]

    df["word"] = df.apply(extract_surface, axis=1)
    # 空白のみ削除し、助詞などは残す
    df = filter_tokens_df(df, pos_exclude=["空白"], strict=False)
    return df

# パターンC: ハイブリッド（今回の採用案）
def logic_hybrid(df: pd.DataFrame) -> pd.DataFrame:
    POS_NORM = ["名詞", "形状詞", "接尾辞", "接頭辞"] # 正規化したい品詞
    POS_STOP = ["空白", "補助記号", "記号", "感動詞"] # 削除したい品詞

    df = filter_tokens_df(df, pos_exclude=POS_STOP, strict=False)
    if df.empty: return df

    # 正規化対象 *以外*（動詞など）はSurfaceを適用
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

    # 数値を [NUM] 化
    pattern_num = r'^\d+(\.\d+)?$'
    is_number = df["word"].astype(str).str.match(pattern_num)
    if is_number.any():
        df.loc[is_number, "word"] = "[NUM]"

    return df

# ... (各ロジックで CorpusDB.export_filtered_tokens_db を実行する処理) ...

```

---

## ステップ2: モデルの学習（パラメータの調整例）

作成した3つのDBを使用し、それぞれの特性に合わせてパラメータ（特に Window サイズ）を調整して学習させました。

### 実装例: [`train_all_models.py`](./example_codes/train_all_models.py)

```python
# ... (インポート・設定省略) ...

# モデルごとの設定例
configs = [
    {
        "name": "Filtered",
        "db_name": "work_strategies_filtered.db",
        "model_name": "w2v_filtered.model",
        "window": 5, # 正規化されており単語密度が高いため、文脈は狭めで設定
        "check_words": ["戦略", "成長", "課題", "DX", "[NUM]"]
    },
    {
        "name": "Surface",
        "db_name": "work_strategies_surface.db",
        "model_name": "w2v_surface.model",
        "window": 10, # 助詞などが間に挟まるため、広めのWindowで文脈を拾う
        "check_words": ["戦略", "成長", "課題", "DX", "する"]
    },
    {
        "name": "Hybrid",
        "db_name": "work_strategies_hybrid.db",
        "model_name": "w2v_hybrid.model",
        "window": 10, # Surfaceの良さを活かすため広めに設定
        "check_words": ["戦略", "成長", "課題", "DX", "する", "[NUM]"]
    }
]

# ... (gensim.models.Word2Vec で学習し save するループ処理) ...

```

---

## ステップ3: 比較検証の結果例

作成されたモデルをロードし、主要な単語の類似語（Similar Words）を確認した結果の一例です。比較をおこなうコードは [`compare_models.py`](./example_codes/compare_models.py) で参照できます。

### ログ出力の抜粋（[`compare_models.log`](./example_codes/compare_models.log) より）

#### 1. 名詞の表記揺れについて (Target: DX)

| Model | 結果の傾向 |
| --- | --- |
| **Filtered** | 「デジタルトランス...」「トランス...」などが上位。正規化されているため精度が高い。 |
| **Surface** | 「ＤＸ」（全角）や「Digital」などが混在し、類似語スコアが分散する傾向が見られた。 |
| **Hybrid** | Filtered同様に正規化が効いており、かつスコアも高く、表記揺れに強い結果となった。 |

#### 2. 数値の意味獲得について (Target: [NUM])

| Model | 結果の傾向 |
| --- | --- |
| **Filtered / Hybrid** | `[NUM]` の類似語に **「定量目標」「中期」「期」** などが出現。数値が「目標値」や「期間」を表す文脈で使われていることを学習できた。 |
| **Surface** | 数値をそのまま学習させた（または削除した）ため、このような抽象的な概念は獲得できなかった。 |

### 結論（今回のケース）

このデータセットにおいては、**「Hybrid」アプローチ** が最も目的に合致する結果となりました。
しかし、分析の目的が「文体の揺らぎを見たい（Surface有利）」場合や、「とにかくシンプルに概観したい（Filtered有利）」場合もあるため、状況に応じて使い分けることが重要です。
