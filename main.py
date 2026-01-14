"""
競馬予測モデル（LightGBM）

このスクリプトは、日本中央競馬（JRA）のレース結果を予測するための
機械学習モデルを構築・評価するものです。

主な機能:
1. CSVデータの読み込みと前処理
2. 馬の過去成績に基づく適性特徴量の生成
3. 時系列クロスバリデーションによるモデル選定
4. 期待値（EV）ベースの購入戦略の最適化
5. Isotonic Regressionによる確率キャリブレーション

モデルの目的:
- 「回収率100%超え」を達成する購入戦略の発見
- 予測確率が実際の勝率と一致するようにキャリブレーション
"""

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# =============================================================================
# パラメータ設定
# =============================================================================
# このセクションでは、モデルの挙動を制御するハイパーパラメータを定義しています。
# 値を変更することで、特徴量の計算方法やモデルの複雑さを調整できます。

# --- 直近N走の加重平均の設定 ---
# 馬の「最近の調子」を捉えるため、直近の成績を重視した特徴量を作成します。
# ROLLING_WINDOW: 何走分のデータを使うか（5走＝約3-6ヶ月分の成績）
ROLLING_WINDOW = 5

# 重みパターン: 直近のレースほど大きな重みを与えることで、
# 「今の調子」をより強く反映させます。正規化は計算時に自動で行われます。
# - linear: 線形増加（5走前:1, 4走前:2, ..., 直前:5）
# - exp: 指数減衰（直近のレースの影響が指数的に大きくなる）
# - front_heavy: 直前2走を特に重視するパターン
WEIGHT_PATTERNS = {
    "linear": [1, 2, 3, 4, 5],
    "exp": [0.0625, 0.125, 0.25, 0.5, 1],  # alpha=0.5の指数減衰
    "front_heavy": [0.05, 0.1, 0.15, 0.3, 0.4],
}
SELECTED_WEIGHT = "exp"  # 実験の結果、指数減衰が最も良い性能を示した

# --- 適性特徴量のスムージング設定 ---
# 馬の「適性」（コース・距離・芝/ダートへの相性）を計算する際、
# 出走数が少ない馬は統計が不安定になります。
# ベイズ推定の考え方を用いて、全体平均に「引き戻す」ことでノイズを抑制します。
USE_SMOOTHED_APTITUDE = True  # スムージングを有効にするか
APTITUDE_PRIOR_N = 20  # 事前分布の強さ（大きいほど全体平均に近づく）

# 距離適性を計算する際のビン幅（200m刻みで距離を分類）
# 例: 1600m, 1800m, 2000m → それぞれ1600, 1800, 2000のビンに分類
DIST_BIN_WIDTH = 200
# =============================================================================


def main():
    """
    メイン処理関数

    処理の流れ:
    1. CSVデータの読み込み（カラム定義 → pandas DataFrame化）
    2. 前処理（日付型変換、数値型変換、欠損値処理）
    3. 特徴量エンジニアリング（適性特徴量、加重平均、騎手/調教師エンコーディング等）
    4. 時系列分割（Train/Valid/Test）
    5. 時系列CVでのモデル選定
    6. 購入戦略の最適化（期待値閾値、確率閾値）
    7. Testデータでの最終評価
    """

    # =========================================================================
    # 1. カラム名の定義 (全52カラム)
    # =========================================================================
    # CSVファイルにはヘッダーがないため、ここでカラム名を明示的に定義します。
    # カラムの順序はデータソースの仕様に準拠しており、変更すると読み込みが失敗します。
    COLUMNS = [
        "年",
        "月",
        "日",
        "回次",
        "場所",
        "日次",
        "レース番号",
        "レース名",
        "クラスコード",
        "芝・ダ",
        "トラックコード",
        "距離",
        "馬場状態",
        "馬名",
        "性別",
        "年齢",
        "騎手名",
        "斤量",
        "頭数",
        "馬番",
        "確定着順",
        "入線着順",
        "異常コード",
        "着差タイム",
        "人気順",
        "走破タイム",
        "走破時計",
        "補正タイム",
        "通過順1",
        "通過順2",
        "通過順3",
        "通過順4",
        "上がり3Fタイム",
        "馬体重",
        "調教師",
        "所属地",
        "賞金",
        "血統登録番号",
        "騎手コード",
        "調教師コード",
        "レースID",
        "馬主名",
        "生産者名",
        "父馬名",
        "母馬名",
        "母の父馬名",
        "毛色",
        "生年月日",
        "単勝オッズ",
        "馬印",
        "レース印",
        "PCI",  # ペースチェンジ指数（レース展開の指標）
    ]

    # =========================================================================
    # 2. データの読み込み
    # =========================================================================
    # cp932エンコーディングはWindows日本語環境で一般的なエンコーディングです。
    # データソースがShift-JIS系のため、cp932を使用します。
    print("データを読み込んでいます...")
    try:
        # low_memory=False: 大きなファイルでのデータ型推定の問題を回避
        # names/header: ヘッダーなしCSVに対してカラム名を明示的に指定
        df = pd.read_csv(
            "csv/data.csv",
            encoding="cp932",
            names=COLUMNS,
            header=None,
            low_memory=False,
        )
    except FileNotFoundError:
        print("エラー: csv/data.csv が見つかりませんでした。")
        return

    # =========================================================================
    # 3. 前処理
    # =========================================================================
    # このセクションでは、生データをモデルが扱える形式に変換します。
    # 主に以下の処理を行います:
    # - 文字列→数値への変換
    # - 日付型の作成
    # - 目的変数の作成（着順のクラス分類）
    # - 1走前情報の付与（時系列シフト）
    print("前処理を実行中...")

    # --- レースIDの作成 ---
    # 元のレースIDは「YYYYPPTTRRNNBB」形式（BBは馬番）
    # 馬番を除いた「レース単位」のIDを作成することで、
    # 同じレースに出走した馬をグループ化できるようにします
    df["race_id"] = df["レースID"].astype(str).str.slice(0, -2)

    # --- 目的変数の作成 ---
    # 4クラス分類問題として設定:
    # - クラス0: 4着以下（「その他」）
    # - クラス1: 1着
    # - クラス2: 2着
    # - クラス3: 3着
    # この設計により、「複勝圏内（3着以内）」の予測も可能になります
    df["rank_class"] = pd.to_numeric(df["確定着順"], errors="coerce")
    df["rank_class"] = (
        df["rank_class"].where(df["rank_class"].isin([1, 2, 3]), 0).astype(int)
    )
    # 単勝予測用の2値フラグ（1着かどうか）
    df["is_win"] = (df["rank_class"] == 1).astype(int)

    # --- 日付・数値系の整形 ---
    # 年が2桁の場合（10, 11, ... 24）は2000年代として4桁に変換
    # データは2010年〜2024年の範囲であることが確認済み
    df["年"] = pd.to_numeric(df["年"], errors="coerce")
    df["年"] = df["年"] + 2000

    df["月"] = pd.to_numeric(df["月"], errors="coerce")
    df["日"] = pd.to_numeric(df["日"], errors="coerce")

    # 日付型の作成（レース間隔の計算に使用）
    # 休み明けの馬は調子が不安定なことが多いため、間隔は重要な特徴量
    df["date"] = pd.to_datetime(
        df["年"].astype(str) + "-" + df["月"].astype(str) + "-" + df["日"].astype(str),
        format="%Y-%m-%d",
        errors="coerce",
    )

    # --- 数値型への変換 ---
    # errors="coerce" により、変換できない値はNaNになる（欠損値として扱う）
    df["レース番号"] = pd.to_numeric(df["レース番号"], errors="coerce")
    df["距離"] = pd.to_numeric(df["距離"], errors="coerce")
    df["斤量"] = pd.to_numeric(df["斤量"], errors="coerce")  # 騎手込みの負担重量
    df["人気順"] = pd.to_numeric(
        df["人気順"], errors="coerce"
    )  # オッズから計算された人気
    df["単勝オッズ"] = pd.to_numeric(df["単勝オッズ"], errors="coerce")
    df["馬体重"] = pd.to_numeric(df["馬体重"], errors="coerce")
    df["着差タイム"] = pd.to_numeric(
        df["着差タイム"], errors="coerce"
    )  # 勝ち馬との着差（秒）
    df["上がり3Fタイム"] = pd.to_numeric(
        df["上がり3Fタイム"], errors="coerce"
    )  # 最後の600m
    df["PCI"] = pd.to_numeric(df["PCI"], errors="coerce")  # ペースチェンジ指数

    # 通過順（コーナー通過時の順位）を数値化
    # 通過順は脚質（逃げ/先行/差し/追込）を判断するのに重要
    df["通過順1"] = pd.to_numeric(df["通過順1"], errors="coerce")  # 1コーナー
    df["通過順2"] = pd.to_numeric(df["通過順2"], errors="coerce")  # 2コーナー
    df["通過順3"] = pd.to_numeric(df["通過順3"], errors="coerce")  # 3コーナー
    df["通過順4"] = pd.to_numeric(df["通過順4"], errors="coerce")  # 4コーナー（最終）

    # =========================================================================
    # 1走前（前走）情報の付与
    # =========================================================================
    # 馬の「前走の状態」は今走の予測に非常に重要です。
    # 血統登録番号（馬の一意ID）をキーにして、時系列順にソートした上で
    # 1行前のデータを「前走情報」として付与します。
    #
    # ※馬名ではなく血統登録番号を使う理由:
    #   同名の馬が存在する可能性があるため、IDで識別する方が安全
    prev_source_cols = [
        "場所",  # 前走の競馬場
        "距離",  # 前走の距離
        "芝・ダ",  # 前走の芝/ダート
        "確定着順",  # 前走の着順
        "着差タイム",  # 前走での勝ち馬との差
        "斤量",  # 前走の負担重量
        "馬体重",  # 前走時の馬体重
        "通過順1",
        "通過順2",
        "通過順3",
        "通過順4",
        "date",  # 間隔計算用
    ]
    # 時系列順にソート（同じ馬のレースを古い順に並べる）
    df_sorted = df.sort_values(
        by=["血統登録番号", "年", "月", "日", "場所", "レース番号"]
    )
    # shift(1)で1行前（＝前走）のデータを取得
    for col in prev_source_cols:
        df[f"prev_{col}"] = df_sorted.groupby("血統登録番号")[col].shift(1)

    # レース間隔（日数）の計算: 休み明けかどうかの判断に使用
    # 一般的に、30日以上の間隔は「休み明け」とみなされる
    df["interval"] = (df["date"] - df["prev_date"]).dt.days

    # =========================================================================
    # 適性特徴量（過去全レースの累積統計）- 高速化版
    # =========================================================================
    # このセクションでは、馬の「適性」を表す特徴量を生成します。
    # 適性とは、特定の条件（距離、芝/ダート、競馬場）での過去の成績を指します。
    #
    # 設計思想:
    # - 「未来のデータを使わない」ことが最重要（データリーク防止）
    # - cumsum() - 現在行 の形式で「今走より前の累積」を計算
    # - スムージングにより、出走数が少ない馬の統計を安定化
    #
    # 計算される特徴量:
    # - 過去平均着順: 過去レースの平均着順（低いほど強い）
    # - 過去勝率: 1着になった割合
    # - 過去複勝率: 3着以内になった割合
    # - 各適性（コース/芝ダ/距離帯）の同上指標

    # 確定着順を数値化（「取消」「除外」などは NaN になる）
    df["確定着順_num"] = pd.to_numeric(df["確定着順"], errors="coerce")

    # 距離を200m刻みでビン化（距離適性の計算用）
    # 例: 1400m → 1400, 1600m → 1600, 1800m → 1800
    df["距離bin"] = (df["距離"] // DIST_BIN_WIDTH) * DIST_BIN_WIDTH

    # ソート済みDataFrameを再作成（新しいカラムを含める）
    df_sorted = df.sort_values(
        by=["血統登録番号", "年", "月", "日", "場所", "レース番号"]
    ).copy()

    # --- 累積計算用の補助列を作成 ---
    # NaN を含む着順データを効率的に処理するための準備
    df_sorted["_rank"] = df_sorted["確定着順_num"]
    df_sorted["_valid"] = df_sorted["_rank"].notna().astype(int)  # 有効なレース数
    df_sorted["_rank_filled"] = df_sorted["_rank"].fillna(0)  # 合計計算用（NaN→0）
    df_sorted["_win"] = (df_sorted["_rank"] == 1).astype(int)  # 1着フラグ
    df_sorted["_place"] = (df_sorted["_rank"] <= 3).astype(int)  # 3着以内フラグ

    # --- 全体平均（スムージング用の事前分布として使用） ---
    # ベイズ推定の考え方: データが少ない馬は全体平均に近づける
    total_valid = int(df_sorted["_valid"].sum())
    if total_valid > 0:
        global_avg_rank = float(df_sorted["_rank_filled"].sum() / total_valid)
        global_win_rate = float(df_sorted["_win"].sum() / total_valid)
        global_place_rate = float(df_sorted["_place"].sum() / total_valid)
    else:
        global_avg_rank = np.nan
        global_win_rate = np.nan
        global_place_rate = np.nan

    # --- 過去全レースの統計（ベクトル化で高速計算） ---
    # cumsum() - 現在行 = 「現在行より前の累積」
    # これにより、各行で「その時点で使えるデータのみ」を使った統計になる
    grp_horse = df_sorted.groupby("血統登録番号")

    past_cnt = grp_horse["_valid"].cumsum() - df_sorted["_valid"]
    past_rank_sum = grp_horse["_rank_filled"].cumsum() - df_sorted["_rank_filled"]
    past_win_sum = grp_horse["_win"].cumsum() - df_sorted["_win"]
    past_place_sum = grp_horse["_place"].cumsum() - df_sorted["_place"]

    df["過去出走数"] = past_cnt

    # スムージング処理:
    # smoothed_value = (sum + prior_n * global_avg) / (cnt + prior_n)
    # prior_n が大きいほど全体平均に近づき、小標本のノイズを抑制
    if USE_SMOOTHED_APTITUDE:
        denom = past_cnt + APTITUDE_PRIOR_N
        df["過去平均着順"] = (
            past_rank_sum + APTITUDE_PRIOR_N * global_avg_rank
        ) / denom
        df["過去勝率"] = (past_win_sum + APTITUDE_PRIOR_N * global_win_rate) / denom
        df["過去複勝率"] = (
            past_place_sum + APTITUDE_PRIOR_N * global_place_rate
        ) / denom
    else:
        df["過去平均着順"] = (past_rank_sum / past_cnt).where(past_cnt > 0)
        df["過去勝率"] = (past_win_sum / past_cnt).where(past_cnt > 0)
        df["過去複勝率"] = (past_place_sum / past_cnt).where(past_cnt > 0)

    # --- 同競馬場の適性（コース適性） ---
    # 競馬場ごとにコースの特徴（坂の有無、カーブの角度など）が異なるため、
    # 特定の競馬場で好成績を収める馬がいる（いわゆる「コース巧者」）
    grp_place = df_sorted.groupby(["血統登録番号", "場所"])

    place_cnt = grp_place["_valid"].cumsum() - df_sorted["_valid"]
    place_rank_sum = grp_place["_rank_filled"].cumsum() - df_sorted["_rank_filled"]
    place_win_sum = grp_place["_win"].cumsum() - df_sorted["_win"]
    place_place_sum = grp_place["_place"].cumsum() - df_sorted["_place"]

    df["コース適性_出走数"] = place_cnt
    if USE_SMOOTHED_APTITUDE:
        denom = place_cnt + APTITUDE_PRIOR_N
        df["コース適性_平均着順"] = (
            place_rank_sum + APTITUDE_PRIOR_N * global_avg_rank
        ) / denom
        df["コース適性_勝率"] = (
            place_win_sum + APTITUDE_PRIOR_N * global_win_rate
        ) / denom
        df["コース適性_複勝率"] = (
            place_place_sum + APTITUDE_PRIOR_N * global_place_rate
        ) / denom
    else:
        df["コース適性_平均着順"] = (place_rank_sum / place_cnt).where(place_cnt > 0)
        df["コース適性_勝率"] = (place_win_sum / place_cnt).where(place_cnt > 0)
        df["コース適性_複勝率"] = (place_place_sum / place_cnt).where(place_cnt > 0)

    # --- 同芝ダの適性（トラック種別適性） ---
    # 芝とダートでは求められる適性が大きく異なる
    # - 芝: スピードとキレ（瞬発力）が重要
    # - ダート: パワーとスタミナが重要
    # 多くの馬はどちらかに得意/不得意がある
    grp_surface = df_sorted.groupby(["血統登録番号", "芝・ダ"])

    surface_cnt = grp_surface["_valid"].cumsum() - df_sorted["_valid"]
    surface_rank_sum = grp_surface["_rank_filled"].cumsum() - df_sorted["_rank_filled"]
    surface_win_sum = grp_surface["_win"].cumsum() - df_sorted["_win"]
    surface_place_sum = grp_surface["_place"].cumsum() - df_sorted["_place"]

    df["芝ダ適性_出走数"] = surface_cnt
    if USE_SMOOTHED_APTITUDE:
        denom = surface_cnt + APTITUDE_PRIOR_N
        df["芝ダ適性_平均着順"] = (
            surface_rank_sum + APTITUDE_PRIOR_N * global_avg_rank
        ) / denom
        df["芝ダ適性_勝率"] = (
            surface_win_sum + APTITUDE_PRIOR_N * global_win_rate
        ) / denom
        df["芝ダ適性_複勝率"] = (
            surface_place_sum + APTITUDE_PRIOR_N * global_place_rate
        ) / denom
    else:
        df["芝ダ適性_平均着順"] = (surface_rank_sum / surface_cnt).where(
            surface_cnt > 0
        )
        df["芝ダ適性_勝率"] = (surface_win_sum / surface_cnt).where(surface_cnt > 0)
        df["芝ダ適性_複勝率"] = (surface_place_sum / surface_cnt).where(surface_cnt > 0)

    # --- 同距離帯の適性（距離適性） ---
    # 馬には得意な距離がある（スプリンター、マイラー、ステイヤーなど）
    # 200m刻みでビン化することで、近い距離をまとめて統計を安定化
    grp_dist = df_sorted.groupby(["血統登録番号", "距離bin"])

    dist_cnt = grp_dist["_valid"].cumsum() - df_sorted["_valid"]
    dist_rank_sum = grp_dist["_rank_filled"].cumsum() - df_sorted["_rank_filled"]
    dist_win_sum = grp_dist["_win"].cumsum() - df_sorted["_win"]
    dist_place_sum = grp_dist["_place"].cumsum() - df_sorted["_place"]

    df["距離適性_出走数"] = dist_cnt
    if USE_SMOOTHED_APTITUDE:
        denom = dist_cnt + APTITUDE_PRIOR_N
        df["距離適性_平均着順"] = (
            dist_rank_sum + APTITUDE_PRIOR_N * global_avg_rank
        ) / denom
        df["距離適性_勝率"] = (
            dist_win_sum + APTITUDE_PRIOR_N * global_win_rate
        ) / denom
        df["距離適性_複勝率"] = (
            dist_place_sum + APTITUDE_PRIOR_N * global_place_rate
        ) / denom
    else:
        df["距離適性_平均着順"] = (dist_rank_sum / dist_cnt).where(dist_cnt > 0)
        df["距離適性_勝率"] = (dist_win_sum / dist_cnt).where(dist_cnt > 0)
        df["距離適性_複勝率"] = (dist_place_sum / dist_cnt).where(dist_cnt > 0)

    # 一時カラムを削除（メモリ効率化）
    df_sorted = df_sorted.drop(
        columns=["_rank", "_valid", "_rank_filled", "_win", "_place"]
    )

    # =========================================================================
    # 直近N走の平均・加重平均（調子を捉える特徴量）
    # =========================================================================
    # 馬の「今の調子」を捉えるため、直近数走の成績を重視した特徴量を作成します。
    # 単純平均だと全ての過去走を同等に扱ってしまうため、
    # 直近ほど重みを大きくする「加重平均」も計算します。
    agg_cols = ["上がり3Fタイム", "PCI"]  # 加重平均を計算する対象カラム
    weights = np.array(WEIGHT_PATTERNS[SELECTED_WEIGHT])

    # --- 単純移動平均（Simple Moving Average） ---
    # shift(1)で「今走を含めない」ようにしてから、直近N走の平均を計算
    for col in agg_cols:
        df[f"avg{ROLLING_WINDOW}_{col}"] = (
            df_sorted.groupby("血統登録番号")[col]
            .apply(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    # --- 加重移動平均（Weighted Moving Average） ---
    # 直近のレースほど大きな重みを与えることで、最近の調子を強く反映
    def weighted_mean_n(x, n=ROLLING_WINDOW, w=weights):
        """直近n走の加重平均を計算する関数

        Args:
            x: 馬ごとにグループ化されたシリーズ
            n: ウィンドウサイズ
            w: 重みの配列（末尾が最新）

        Returns:
            加重平均のシリーズ

        Note:
            データが n 未満の場合、利用可能なデータ分だけ使用
            （重みは末尾から取る = 直近重視を維持）
        """
        shifted = x.shift(1)  # 今走を除外

        def calc(window):
            valid = window.dropna()
            n_valid = len(valid)
            if n_valid == 0:
                return np.nan
            # 直近n_valid個分の重みを使用（末尾から取る）
            # 例: 3走分しかない場合、[0.25, 0.5, 1] の末尾3つを使う
            w_slice = w[-n_valid:]
            w_norm = w_slice / w_slice.sum()  # 正規化
            return (valid * w_norm).sum()

        return shifted.rolling(n, min_periods=1).apply(calc, raw=False)

    for col in agg_cols:
        df[f"wma{ROLLING_WINDOW}_{col}"] = (
            df_sorted.groupby("血統登録番号")[col]
            .apply(weighted_mean_n)
            .reset_index(level=0, drop=True)
        )

    # =========================================================================
    # 輸送フラグ（長距離輸送によるストレス）
    # =========================================================================
    # 競走馬は長距離輸送に弱いとされています。
    # 美浦（関東）所属馬が関西・北海道で走る場合、
    # 栗東（関西）所属馬が関東・北海道で走る場合、
    # 輸送によるストレスでパフォーマンスが落ちる可能性があります。
    #
    # 所属地コード:
    # - '美': 美浦（関東）
    # - '栗': 栗東（関西）
    # - '地': 地方競馬
    # - '外': 海外

    # 競馬場の地域分類
    KANTO_PLACES = ["福島", "新潟", "東京", "中山"]  # 関東圏
    KANSAI_PLACES = ["中京", "京都", "阪神", "小倉"]  # 関西圏
    HOKKAIDO_PLACES = ["札幌", "函館"]  # 北海道

    place = df["場所"].astype(str)
    center = df["所属地"].astype(str)

    is_kanto_race = place.isin(KANTO_PLACES)
    is_kansai_race = place.isin(KANSAI_PLACES)
    is_hokkaido_race = place.isin(HOKKAIDO_PLACES)

    # 長距離輸送フラグ:
    # - 美浦所属が関西/北海道で出走 → 長距離輸送
    # - 栗東所属が関東/北海道で出走 → 長距離輸送
    df["長距離輸送"] = 0
    df.loc[(center == "美") & (is_kansai_race | is_hokkaido_race), "長距離輸送"] = 1
    df.loc[(center == "栗") & (is_kanto_race | is_hokkaido_race), "長距離輸送"] = 1

    # --- 前走からの変化量（状態の変化を捉える） ---
    # 斤量変化: 負担重量が増えると不利になる傾向
    df["斤量変化"] = df["斤量"] - df["prev_斤量"]

    # 馬体重変化: 急激な増減は調子の変化を示唆
    # 一般的に、±10kg以上の変動は注意が必要
    df["馬体重変化"] = df["馬体重"] - df["prev_馬体重"]

    # =========================================================================
    # 騎手・調教師ターゲットエンコーディング（スムージング付き）
    # =========================================================================
    # 騎手や調教師の「実力」を数値化するための特徴量です。
    #
    # ターゲットエンコーディングとは:
    # カテゴリ変数（騎手名など）を、その目的変数（勝率）の平均値で置き換える手法
    #
    # 課題と対策:
    # 1. データリーク: 未来のデータを使ってはいけない
    #    → 累積計算（cumsum - 現在行）で「その時点までの勝率」を計算
    # 2. 小標本問題: 騎乗数が少ない騎手は統計が不安定
    #    → スムージング（全体平均への引き戻し）で対応
    #
    # 計算式: (過去勝利数 + prior_n * 全体勝率) / (過去騎乗数 + prior_n)
    TARGET_ENCODING_PRIOR_N = 50  # スムージングの強さ（大きいほど全体平均に近づく）

    # 時系列順にソート（リーク防止のため）
    df_sorted_te = df.sort_values(by=["年", "月", "日", "場所", "レース番号"]).copy()
    df_sorted_te["_win"] = df_sorted_te["is_win"]
    df_sorted_te["_valid"] = 1  # 全レースが有効

    # 全体の勝率（事前分布として使用）
    global_win_rate_te = df_sorted_te["_win"].mean()

    # --- 騎手コード別の累積勝率 ---
    # 優秀な騎手ほど高い値になる
    grp_jockey = df_sorted_te.groupby("騎手コード")
    jockey_cnt = grp_jockey["_valid"].cumsum() - df_sorted_te["_valid"]
    jockey_win_sum = grp_jockey["_win"].cumsum() - df_sorted_te["_win"]
    denom = jockey_cnt + TARGET_ENCODING_PRIOR_N
    df["騎手勝率"] = (
        jockey_win_sum + TARGET_ENCODING_PRIOR_N * global_win_rate_te
    ) / denom

    # --- 調教師コード別の累積勝率 ---
    # 管理馬の成績が良い調教師ほど高い値になる
    grp_trainer = df_sorted_te.groupby("調教師コード")
    trainer_cnt = grp_trainer["_valid"].cumsum() - df_sorted_te["_valid"]
    trainer_win_sum = grp_trainer["_win"].cumsum() - df_sorted_te["_win"]
    denom = trainer_cnt + TARGET_ENCODING_PRIOR_N
    df["調教師勝率"] = (
        trainer_win_sum + TARGET_ENCODING_PRIOR_N * global_win_rate_te
    ) / denom

    # =========================================================================
    # 直近N走の着差加重平均
    # =========================================================================
    # 着差タイムは「どれだけ勝ち馬から離されたか」を示す
    # 直近の着差が小さいほど、調子が良いと判断できる
    df[f"wma{ROLLING_WINDOW}_着差タイム"] = (
        df_sorted.groupby("血統登録番号")["着差タイム"]
        .apply(weighted_mean_n)
        .reset_index(level=0, drop=True)
    )

    # =========================================================================
    # クラス替わり（昇/降級）フラグ
    # =========================================================================
    # 競馬にはクラス制度があり、馬は成績に応じて昇級・降級します。
    # - 昇級直後: 相手が強くなるため苦戦しやすい
    # - 降級直後: 相手が弱くなるため好走しやすい
    # この「クラス替わり」は馬券戦略上、重要な要素です。
    df["prev_クラスコード"] = df_sorted.groupby("血統登録番号")["クラスコード"].shift(1)
    df["クラスコード_num"] = pd.to_numeric(df["クラスコード"], errors="coerce")
    df["prev_クラスコード_num"] = pd.to_numeric(
        df["prev_クラスコード"], errors="coerce"
    )
    # クラス変化: 正=昇級、負=降級、0=同クラス
    df["クラス変化"] = df["クラスコード_num"] - df["prev_クラスコード_num"]
    df["昇級フラグ"] = (df["クラス変化"] > 0).astype(int)
    df["降級フラグ"] = (df["クラス変化"] < 0).astype(int)

    # =========================================================================
    # 脚質特徴量（前走の通過順から算出）
    # =========================================================================
    # 脚質とは、レース中の位置取りの傾向を指します:
    # - 逃げ: 最初から先頭を走る
    # - 先行: 前方の好位置につける
    # - 差し: 中団から徐々に上がる
    # - 追込: 後方から一気に追い上げる
    #
    # 脚質とコース/距離の相性は重要な要素です。
    # 例: 小回りコースでは逃げ・先行が有利

    # 前走の平均通過順（序盤の位置取り傾向）
    # 値が小さいほど前で競馬する馬
    df["prev_平均通過順"] = df[
        ["prev_通過順1", "prev_通過順2", "prev_通過順3", "prev_通過順4"]
    ].mean(axis=1)

    # 前走の脚質指数（序盤 vs 終盤の変化）
    # 通過順1 - 通過順4:
    #   正の値 = 序盤は前にいて終盤に下がる（逃げてバテる）
    #   負の値 = 序盤は後ろで終盤に上がる（追い込み型）
    df["prev_脚質指数"] = df["prev_通過順1"] - df["prev_通過順4"]

    # 直近5走の1コーナー通過順平均（安定した脚質判定）
    df["avg5_通過順1"] = (
        df_sorted.groupby("血統登録番号")["通過順1"]
        .apply(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    # =========================================================================
    # 特徴量の選択
    # =========================================================================
    # モデルに入力する特徴量を定義します。
    #
    # 重要な設計原則:
    # 「レース開始前に確定している情報のみを使用する」
    #
    # 使ってはいけない情報（データリーク）:
    # - 確定着順（これは予測対象）
    # - 走破タイム、着差タイム（レース結果）
    # - 払戻金額など
    #
    # 使える情報:
    # - 馬の過去成績（前走まで）
    # - 出走時点で確定している情報（枠順、斤量、オッズなど）
    features = [
        # --- 基本情報（レース条件・馬の属性） ---
        "場所",  # 競馬場（東京、中山など）
        "所属地",  # 所属トレセン（美浦/栗東）
        "クラスコード",  # クラス（G1、G2、条件戦など）
        "芝・ダ",  # 芝/ダートの区分
        "トラックコード",  # コースの詳細情報
        "距離",  # レース距離
        "馬場状態",  # 良/稍重/重/不良
        "性別",  # 牡/牝/騙
        "年齢",  # 馬の年齢
        "父馬名",  # 父馬（血統）
        "斤量",  # 負担重量
        "馬体重",  # 馬の体重
        "頭数",  # 出走頭数
        "馬番",  # ゲート番号
        "人気順",  # オッズから算出された人気
        "単勝オッズ",  # 単勝馬券のオッズ
        "騎手コード",  # 騎手の識別コード
        "調教師コード",  # 調教師の識別コード
        "生産者名",  # 生産牧場
        # --- 追加特徴量（エンジニアリングで作成） ---
        "長距離輸送",  # 遠征フラグ
        "斤量変化",  # 前走からの斤量変化
        "馬体重変化",  # 前走からの馬体重変化
        # --- 適性特徴量（過去全レースの累積統計） ---
        "過去平均着順",
        "過去勝率",
        "過去複勝率",
        "過去出走数",
        # --- コース適性（同競馬場での成績） ---
        "コース適性_平均着順",
        "コース適性_勝率",
        "コース適性_複勝率",
        "コース適性_出走数",
        # --- 芝ダ適性（芝/ダートでの成績） ---
        "芝ダ適性_平均着順",
        "芝ダ適性_勝率",
        "芝ダ適性_複勝率",
        "芝ダ適性_出走数",
        # --- 距離適性（同距離帯での成績） ---
        "距離適性_平均着順",
        "距離適性_勝率",
        "距離適性_複勝率",
        "距離適性_出走数",
        # --- 脚質特徴量（レース中の位置取り傾向） ---
        "prev_平均通過順",  # 前走の平均通過順
        "prev_脚質指数",  # 前走の脚質指数
        "avg5_通過順1",  # 直近5走の1コーナー通過順平均
        # --- 1走前情報（前走のレース条件・結果） ---
        "prev_場所",
        "prev_距離",
        "prev_芝・ダ",
        "prev_確定着順",
        "prev_着差タイム",
        # --- 直近N走の平均・加重平均（調子を表す） ---
        f"avg{ROLLING_WINDOW}_上がり3Fタイム",
        f"avg{ROLLING_WINDOW}_PCI",
        f"wma{ROLLING_WINDOW}_上がり3Fタイム",
        f"wma{ROLLING_WINDOW}_PCI",
        f"wma{ROLLING_WINDOW}_着差タイム",
        # --- 騎手・調教師の実力（ターゲットエンコーディング） ---
        "騎手勝率",
        "調教師勝率",
        # --- クラス替わり ---
        "昇級フラグ",
        "降級フラグ",
        # --- レース間隔（休み明けかどうか） ---
        "interval",
    ]

    # =========================================================================
    # カテゴリカル変数の指定
    # =========================================================================
    # LightGBMはカテゴリカル変数を効率的に扱うことができます。
    # 'category'型に変換することで、内部で最適な分割を見つけてくれます。
    # （One-Hot Encodingよりも効率的）
    categorical_features = [
        "場所",
        "所属地",
        "クラスコード",
        "芝・ダ",
        "トラックコード",
        "馬場状態",
        "性別",
        "父馬名",
        "騎手コード",
        "調教師コード",
        "生産者名",
        "prev_場所",
        "prev_芝・ダ",
    ]

    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # =========================================================================
    # 4. 時系列でのデータ分割（レース単位）
    # =========================================================================
    # 重要: 競馬予測では「未来のデータで学習しない」ことが絶対条件です。
    #
    # 通常のランダム分割を使うと、2024年のデータで学習したモデルで
    # 2023年のレースを予測することになり、現実的ではありません。
    #
    # 時系列分割:
    # - 過去のデータで学習
    # - 未来のデータで評価
    # これにより、実運用に近い条件でモデルを評価できます。
    print("データを分割中...")

    # データを時系列順にソート
    df = df.sort_values(by=["年", "月", "日", "場所", "レース番号"])

    # ユニークなレースIDを取得（順序を保持）
    unique_race_ids = df["race_id"].unique()
    n_races = len(unique_race_ids)

    # =========================================================================
    # 時系列クロスバリデーション（Time-Series CV）のFold定義
    # =========================================================================
    # 単一の Train/Valid 分割では、特定の期間に依存した結果になる可能性があります。
    # そのため、複数のFoldで評価を行い、モデルの安定性を確認します。
    #
    # Fold設計:
    # - 各FoldでTrain期間を少しずつずらす
    # - Test期間は全Foldで共通（最終評価用に「未見」のデータとして保持）
    #
    # Train [0:50%] | Valid [50:60%] | (Gap) | Test [70:100%]
    # Train [0:55%] | Valid [55:65%] | (Gap) | Test [70:100%]
    # Train [0:60%] | Valid [60:70%] | (Gap) | Test [70:100%]
    CV_FOLDS = [
        {"train_end": 0.50, "valid_end": 0.60},
        {"train_end": 0.55, "valid_end": 0.65},
        {"train_end": 0.60, "valid_end": 0.70},
    ]
    TEST_START = 0.70  # テストデータの開始位置（最後の30%）

    def get_fold_masks(fold_config):
        """指定されたFold設定に基づいて、Train/Validのマスクを返す"""
        train_end_idx = int(n_races * fold_config["train_end"])
        valid_end_idx = int(n_races * fold_config["valid_end"])
        train_ids = unique_race_ids[:train_end_idx].tolist()
        valid_ids = unique_race_ids[train_end_idx:valid_end_idx].tolist()
        train_m = df["race_id"].isin(train_ids)
        valid_m = df["race_id"].isin(valid_ids)
        return train_m, valid_m

    # テストデータのマスクを作成
    test_start_idx = int(n_races * TEST_START)
    test_race_ids = unique_race_ids[test_start_idx:].tolist()
    test_mask = df["race_id"].isin(test_race_ids)

    # 代表的なFold（最終Fold）のマスクを保持（特徴量確認用）
    train_end = int(n_races * CV_FOLDS[-1]["train_end"])
    valid_end = int(n_races * CV_FOLDS[-1]["valid_end"])
    train_race_ids = unique_race_ids[:train_end].tolist()
    valid_race_ids = unique_race_ids[train_end:valid_end].tolist()

    print(f"総レース数: {n_races}")
    print(f"時系列CV Fold数: {len(CV_FOLDS)}")
    print(
        f"代表Fold - 学習用レース数: {len(train_race_ids)}, 検証用レース数: {len(valid_race_ids)}, テスト用レース数: {len(test_race_ids)}"
    )

    train_mask = df["race_id"].isin(train_race_ids)
    valid_mask = df["race_id"].isin(valid_race_ids)

    print(
        f"学習データサンプル数: {int(train_mask.sum())}, 検証データサンプル数: {int(valid_mask.sum())}, テストデータサンプル数: {int(test_mask.sum())}"
    )

    # =========================================================================
    # 5. LightGBMモデルの学習と購入戦略の最適化
    # =========================================================================
    # このセクションでは、以下の2段階の最適化を行います:
    #
    # 1. モデル選定: 時系列CVで複数のハイパーパラメータセットを評価
    # 2. 戦略選定: Validデータで最適な購入閾値を探索
    #
    # 重要な方針:
    # - Validデータで戦略を選定し、Testは「最終確認」としてのみ使用
    # - Testデータでの評価は1回のみ（過学習を防ぐ）
    print("LightGBMモデルを学習中...")
    print("\n[方針] validの回収率でモデル/閾値を選定し、testは最後に1回だけ評価します")

    # --- 購入戦略の閾値パラメータ ---
    # 期待値（EV）閾値: EV = 予測勝率 × オッズ
    # EV > 1.0 なら理論上は「買い」だが、実際にはマージンを取る
    ev_thresholds = [1.0, 1.5, 2.0]

    # 勝率閾値: 最低限の勝率がないと的中率が低すぎる
    proba_thresholds = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]

    # 複勝確率閾値: 3着以内に入る確率でフィルタ
    place_proba_thresholds = [0.0, 0.4]

    # オッズ下限: 低オッズは回収率を上げにくいため除外
    FIXED_ODDS_MIN = 5.0

    # 安定性のための最低購入数（少なすぎると偶然の影響が大きい）
    min_bets_for_best = 8000

    # 確率の正規化モード（レース内で合計1にするか）
    win_proba_modes = ["raw"]  # "race_sum"は効果が薄いため削除

    def normalize_win_proba_by_race(df_eval, win_proba_raw):
        """レース内で勝率の合計が1になるように正規化"""
        s = pd.Series(win_proba_raw, index=df_eval.index, name="proba_1_raw")
        race_sum = s.groupby(df_eval["race_id"], dropna=False).transform("sum")
        return np.where(race_sum > 0, win_proba_raw / race_sum, 0)

    def get_win_proba(df_eval, win_proba_raw, proba_mode: str):
        """指定されたモードに応じた勝率を返す"""
        if proba_mode == "raw":
            return win_proba_raw
        if proba_mode == "race_sum":
            return normalize_win_proba_by_race(df_eval, win_proba_raw)
        raise ValueError(f"未知のproba_modeです: {proba_mode}")

    def select_best_strategy_on_valid(
        df_eval,
        y_true,
        y_pred_proba,
        *,
        win_proba_modes_override=None,
        proba_thresholds_override=None,
        ev_thresholds_override=None,
        place_proba_thresholds_override=None,
        isotonic_calibrator=None,
        use_calibrated_proba: bool = False,
        proba_mask_use_calibrated: bool = True,
        selection_objective: str = "return",
        ev_reliability_min_expected_return: float = 100.0,
    ):
        """
        Validデータ上で最適な購入戦略（閾値の組み合わせ）を探索する関数

        探索する閾値:
        - proba_mode: 確率の正規化方法
        - place_thr: 複勝確率の下限
        - proba_thr: 勝率の下限
        - ev_thr: 期待値の下限

        選定基準 (selection_objective):
        - "return": 回収率を最大化
        - "ev_reliability": 期待回収率と実績回収率の乖離を最小化

        Returns:
            (best_stable, best_overall): 安定した戦略と全体最良の戦略
            best_stable は min_bets_for_best 以上の購入数がある戦略の中で最良
        """
        win_proba_raw = y_pred_proba[:, 1]  # クラス1（1着）の予測確率
        # 複勝確率 = 1着 + 2着 + 3着 の確率
        place_proba = y_pred_proba[:, 1] + y_pred_proba[:, 2] + y_pred_proba[:, 3]

        # オッズを取得し、NaNを0に変換
        odds = np.nan_to_num(np.asarray(df_eval["単勝オッズ"]), nan=0.0)
        # 低オッズを除外するマスク
        odds_mask = odds >= FIXED_ODDS_MIN

        y_true_array = np.asarray(y_true)
        n_winners = int(np.sum(y_true_array == 1))  # 全勝者数（Recall計算用）

        # 閾値のオーバーライド処理（テスト時に異なる閾値を使う場合）

        win_proba_modes_local = (
            win_proba_modes_override
            if win_proba_modes_override is not None
            else win_proba_modes
        )
        proba_thresholds_local = (
            proba_thresholds_override
            if proba_thresholds_override is not None
            else proba_thresholds
        )
        ev_thresholds_local = (
            ev_thresholds_override
            if ev_thresholds_override is not None
            else ev_thresholds
        )
        place_proba_thresholds_local = (
            place_proba_thresholds_override
            if place_proba_thresholds_override is not None
            else place_proba_thresholds
        )

        sweep_results: list[dict] = []
        for proba_mode in win_proba_modes_local:
            win_proba = get_win_proba(df_eval, win_proba_raw, proba_mode=proba_mode)

            if use_calibrated_proba:
                if isotonic_calibrator is None:
                    raise ValueError(
                        "use_calibrated_proba=True の場合は isotonic_calibrator が必要です"
                    )

                win_proba_calibrated = isotonic_calibrator.predict(win_proba)
                win_proba_for_ev = win_proba_calibrated
                win_proba_for_proba_mask = (
                    win_proba_calibrated if proba_mask_use_calibrated else win_proba
                )
            else:
                win_proba_for_ev = win_proba
                win_proba_for_proba_mask = win_proba

            expected_values = win_proba_for_ev * odds

            for place_thr in place_proba_thresholds_local:
                place_mask = place_proba >= place_thr
                for proba_thr in proba_thresholds_local:
                    proba_mask = win_proba_for_proba_mask >= proba_thr
                    for ev_thr in ev_thresholds_local:
                        selected_mask = (
                            (expected_values >= ev_thr)
                            & proba_mask
                            & odds_mask
                            & place_mask
                        )
                        n_bets = int(np.sum(selected_mask))
                        if n_bets == 0:
                            continue

                        hit_mask = selected_mask & (y_true_array == 1)
                        hit_count = int(np.sum(hit_mask))

                        precision = hit_count / n_bets
                        recall = hit_count / n_winners if n_winners > 0 else 0.0

                        return_amount = float(np.sum(odds[hit_mask]))
                        return_rate = (return_amount / n_bets) * 100

                        expected_return_rate = float(
                            np.mean(expected_values[selected_mask]) * 100
                        )
                        return_vs_expected = (
                            return_rate / expected_return_rate
                            if expected_return_rate > 0
                            else np.nan
                        )
                        reliability_error = (
                            float(np.abs(return_vs_expected - 1.0))
                            if not np.isnan(return_vs_expected)
                            else np.nan
                        )

                        sweep_results.append(
                            {
                                "確率モード": proba_mode,
                                "複勝確率閾値": place_thr,
                                "proba閾値": proba_thr,
                                "期待値閾値": ev_thr,
                                "購入数": n_bets,
                                "的中率(Precision)": precision,
                                "再現率(Recall)": recall,
                                "回収率(%)": return_rate,
                                "期待回収率(%)": expected_return_rate,
                                "実績/期待": return_vs_expected,
                                "信頼性誤差": reliability_error,
                            }
                        )

        sweep_df = pd.DataFrame(sweep_results)
        if sweep_df.empty:
            return None, None

        stable_df = sweep_df.loc[sweep_df["購入数"] >= min_bets_for_best]

        if selection_objective == "return":
            best_stable = (
                stable_df.sort_values(by="回収率(%)", ascending=False).iloc[0].to_dict()
                if not stable_df.empty
                else None
            )
            best_overall = (
                sweep_df.sort_values(by="回収率(%)", ascending=False).iloc[0].to_dict()
            )
            return best_stable, best_overall

        if selection_objective != "ev_reliability":
            raise ValueError(f"未知のselection_objectiveです: {selection_objective}")

        def pick_best_by_reliability(df_candidates: pd.DataFrame):
            df_local = df_candidates.dropna(
                subset=["信頼性誤差", "期待回収率(%)", "実績/期待"]
            ).copy()
            if df_local.empty:
                return None

            positive_ev_df = df_local.loc[
                df_local["期待回収率(%)"] >= ev_reliability_min_expected_return
            ]
            if not positive_ev_df.empty:
                df_local = positive_ev_df

            return (
                df_local.sort_values(
                    by=["信頼性誤差", "期待回収率(%)", "回収率(%)"],
                    ascending=[True, False, False],
                )
                .iloc[0]
                .to_dict()
            )

        best_stable = (
            pick_best_by_reliability(stable_df) if not stable_df.empty else None
        )
        best_overall = pick_best_by_reliability(sweep_df)

        return best_stable, best_overall

    def apply_aptitude_features(prior_n, dist_bin_width):
        """適性特徴量を再計算して df に上書きする（全体→コース→芝ダ→距離帯）。"""

        df_sorted_local = df.sort_values(
            by=["血統登録番号", "年", "月", "日", "場所", "レース番号"]
        )

        df_sorted_local["_rank"] = df_sorted_local["確定着順_num"]
        df_sorted_local["_valid"] = df_sorted_local["_rank"].notna().astype(int)
        df_sorted_local["_rank_filled"] = df_sorted_local["_rank"].fillna(0)
        df_sorted_local["_win"] = (df_sorted_local["_rank"] == 1).astype(int)
        df_sorted_local["_place"] = (df_sorted_local["_rank"] <= 3).astype(int)
        df_sorted_local["_dist_bin"] = (
            df_sorted_local["距離"] // dist_bin_width
        ) * dist_bin_width

        total_valid = int(df_sorted_local["_valid"].sum())
        if total_valid > 0:
            global_avg_rank = float(df_sorted_local["_rank_filled"].sum() / total_valid)
            global_win_rate = float(df_sorted_local["_win"].sum() / total_valid)
            global_place_rate = float(df_sorted_local["_place"].sum() / total_valid)
        else:
            global_avg_rank = np.nan
            global_win_rate = np.nan
            global_place_rate = np.nan

        # --- 過去全レース ---
        grp_horse = df_sorted_local.groupby("血統登録番号")

        past_cnt = grp_horse["_valid"].cumsum() - df_sorted_local["_valid"]
        past_rank_sum = (
            grp_horse["_rank_filled"].cumsum() - df_sorted_local["_rank_filled"]
        )
        past_win_sum = grp_horse["_win"].cumsum() - df_sorted_local["_win"]
        past_place_sum = grp_horse["_place"].cumsum() - df_sorted_local["_place"]

        df["過去出走数"] = past_cnt

        if USE_SMOOTHED_APTITUDE:
            denom = past_cnt + prior_n
            df["過去平均着順"] = (past_rank_sum + prior_n * global_avg_rank) / denom
            df["過去勝率"] = (past_win_sum + prior_n * global_win_rate) / denom
            df["過去複勝率"] = (past_place_sum + prior_n * global_place_rate) / denom
        else:
            df["過去平均着順"] = (past_rank_sum / past_cnt).where(past_cnt > 0)
            df["過去勝率"] = (past_win_sum / past_cnt).where(past_cnt > 0)
            df["過去複勝率"] = (past_place_sum / past_cnt).where(past_cnt > 0)

        # --- 同競馬場 ---
        grp_place = df_sorted_local.groupby(["血統登録番号", "場所"], observed=True)

        place_cnt = grp_place["_valid"].cumsum() - df_sorted_local["_valid"]
        place_rank_sum = (
            grp_place["_rank_filled"].cumsum() - df_sorted_local["_rank_filled"]
        )
        place_win_sum = grp_place["_win"].cumsum() - df_sorted_local["_win"]
        place_place_sum = grp_place["_place"].cumsum() - df_sorted_local["_place"]

        df["コース適性_出走数"] = place_cnt

        if USE_SMOOTHED_APTITUDE:
            denom = place_cnt + prior_n
            df["コース適性_平均着順"] = (
                place_rank_sum + prior_n * global_avg_rank
            ) / denom
            df["コース適性_勝率"] = (place_win_sum + prior_n * global_win_rate) / denom
            df["コース適性_複勝率"] = (
                place_place_sum + prior_n * global_place_rate
            ) / denom
        else:
            df["コース適性_平均着順"] = (place_rank_sum / place_cnt).where(
                place_cnt > 0
            )
            df["コース適性_勝率"] = (place_win_sum / place_cnt).where(place_cnt > 0)
            df["コース適性_複勝率"] = (place_place_sum / place_cnt).where(place_cnt > 0)

        # --- 同芝ダ ---
        grp_surface = df_sorted_local.groupby(["血統登録番号", "芝・ダ"], observed=True)

        surface_cnt = grp_surface["_valid"].cumsum() - df_sorted_local["_valid"]
        surface_rank_sum = (
            grp_surface["_rank_filled"].cumsum() - df_sorted_local["_rank_filled"]
        )
        surface_win_sum = grp_surface["_win"].cumsum() - df_sorted_local["_win"]
        surface_place_sum = grp_surface["_place"].cumsum() - df_sorted_local["_place"]

        df["芝ダ適性_出走数"] = surface_cnt

        if USE_SMOOTHED_APTITUDE:
            denom = surface_cnt + prior_n
            df["芝ダ適性_平均着順"] = (
                surface_rank_sum + prior_n * global_avg_rank
            ) / denom
            df["芝ダ適性_勝率"] = (surface_win_sum + prior_n * global_win_rate) / denom
            df["芝ダ適性_複勝率"] = (
                surface_place_sum + prior_n * global_place_rate
            ) / denom
        else:
            df["芝ダ適性_平均着順"] = (surface_rank_sum / surface_cnt).where(
                surface_cnt > 0
            )
            df["芝ダ適性_勝率"] = (surface_win_sum / surface_cnt).where(surface_cnt > 0)
            df["芝ダ適性_複勝率"] = (surface_place_sum / surface_cnt).where(
                surface_cnt > 0
            )

        # --- 同距離帯（bin） ---
        grp_dist = df_sorted_local.groupby(["血統登録番号", "_dist_bin"])

        dist_cnt = grp_dist["_valid"].cumsum() - df_sorted_local["_valid"]
        dist_rank_sum = (
            grp_dist["_rank_filled"].cumsum() - df_sorted_local["_rank_filled"]
        )
        dist_win_sum = grp_dist["_win"].cumsum() - df_sorted_local["_win"]
        dist_place_sum = grp_dist["_place"].cumsum() - df_sorted_local["_place"]

        df["距離適性_出走数"] = dist_cnt

        if USE_SMOOTHED_APTITUDE:
            denom = dist_cnt + prior_n
            df["距離適性_平均着順"] = (
                dist_rank_sum + prior_n * global_avg_rank
            ) / denom
            df["距離適性_勝率"] = (dist_win_sum + prior_n * global_win_rate) / denom
            df["距離適性_複勝率"] = (
                dist_place_sum + prior_n * global_place_rate
            ) / denom
        else:
            df["距離適性_平均着順"] = (dist_rank_sum / dist_cnt).where(dist_cnt > 0)
            df["距離適性_勝率"] = (dist_win_sum / dist_cnt).where(dist_cnt > 0)
            df["距離適性_複勝率"] = (dist_place_sum / dist_cnt).where(dist_cnt > 0)

    # ==========================================================================
    # 特徴量パラメータは固定（探索廃止で高速化）
    # 過去の探索結果から最良値を採用: prior_n=50, dist_bin_width=400
    # ==========================================================================
    best_prior_n = 50
    best_dist_bin_width = 400
    print(
        f"\n[特徴量設定] prior_n={best_prior_n}, dist_bin_width={best_dist_bin_width} (固定)"
    )
    apply_aptitude_features(prior_n=best_prior_n, dist_bin_width=best_dist_bin_width)

    # レース単位IDに基づいてDataFrameを分割（最終設定で生成）
    train_df = df.loc[train_mask].copy()
    valid_df = df.loc[valid_mask].copy()
    test_df = df.loc[test_mask].copy()

    # 学習用・検証用・テスト用の説明変数(X)と目的変数(y)を作成
    X_train = train_df[features]
    y_train = train_df["rank_class"].astype(int)
    X_valid = valid_df[features]
    y_valid = valid_df["rank_class"].astype(int)
    X_test = test_df[features]
    y_test = test_df["rank_class"].astype(int)

    print(
        f"\n[採用特徴量設定] prior_n={best_prior_n}, dist_bin_width={best_dist_bin_width}"
    )

    # =========================================================================
    # モデル候補の定義（LightGBMのハイパーパラメータ）
    # =========================================================================
    # 競馬予測では「過学習」が大きな問題になります。
    # 訓練データに過度に適合すると、未来のレースで性能が出ません。
    #
    # そのため、正則化（regularization）を強めに設定しています:
    # - min_child_samples: 葉ノードの最小サンプル数（大きいほど汎化）
    # - reg_alpha/reg_lambda: L1/L2正則化の強さ
    # - subsample/colsample: サンプル/特徴量のサブサンプリング率
    #
    # 2つのモデル候補を用意し、CVで比較します:
    # - P1: 標準的な正則化（汎用的）
    # - P3: 高正則化（過学習対策を強化）
    model_candidates = [
        {
            "name": "P1_regularized",
            "params": {
                "n_estimators": 1500,  # 決定木の数
                "learning_rate": 0.05,  # 学習率
                "num_leaves": 31,  # 葉の最大数
                "max_depth": 6,  # 木の最大深さ
                "min_child_samples": 100,  # 葉の最小サンプル数
                "reg_alpha": 0.5,  # L1正則化
                "reg_lambda": 1.0,  # L2正則化
                "subsample": 0.8,  # 行のサブサンプリング
                "subsample_freq": 1,  # サブサンプリングの頻度
                "colsample_bytree": 0.8,  # 列のサブサンプリング
            },
        },
        {
            "name": "P3_high_reg",
            "params": {
                "n_estimators": 1200,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 150,  # より大きく設定
                "reg_alpha": 1.0,  # より強く
                "reg_lambda": 2.0,  # より強く
                "subsample": 0.7,  # より小さく
                "subsample_freq": 1,
                "colsample_bytree": 0.7,  # より小さく
            },
        },
    ]

    # =========================================================================
    # 時系列クロスバリデーションによるモデル選定
    # =========================================================================
    # 各モデル候補を複数のFoldで評価し、回収率の平均と最小値を比較します。
    #
    # 選定基準:
    # 1. CVで安定して80%以上の回収率を出せるか（cv_stable）
    # 2. 平均回収率が高いか
    # 3. 最小回収率が高いか（ワーストケースの確認）
    print(f"\n[時系列CV] {len(CV_FOLDS)}つのFoldでモデルを評価します")

    selection_rows: list[dict] = []
    best_model: lgb.LGBMClassifier | None = None
    best_strategy: dict | None = None
    best_model_name: str | None = None

    # CV全体で最良の戦略パラメータを保持
    best_cv_avg_return: float = -np.inf
    best_cv_min_return: float = -np.inf

    for cand in model_candidates:
        name = cand["name"]
        params = cand["params"]

        print(f"\n[候補] {name} を{len(CV_FOLDS)}つのFoldで評価...")

        fold_results: list[dict] = []
        fold_models: list[lgb.LGBMClassifier] = []

        for fold_idx, fold_config in enumerate(CV_FOLDS):
            # このFold用のtrain/validマスクを取得
            fold_train_mask, fold_valid_mask = get_fold_masks(fold_config)

            fold_train_df = df.loc[fold_train_mask].copy()
            fold_valid_df = df.loc[fold_valid_mask].copy()

            X_fold_train = fold_train_df[features]
            y_fold_train = fold_train_df["rank_class"].astype(int)
            X_fold_valid = fold_valid_df[features]
            y_fold_valid = fold_valid_df["rank_class"].astype(int)

            # モデル学習
            fold_clf = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=4,
                metric="multi_logloss",
                class_weight="balanced",
                random_state=42,
                verbose=-1,
                **params,
            )

            fold_clf.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_valid, y_fold_valid)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(0),  # 各Foldの詳細ログは抑制
                ],
            )

            fold_models.append(fold_clf)

            # Valid評価
            y_fold_valid_proba = np.asarray(fold_clf.predict_proba(X_fold_valid))
            best_stable, best_overall = select_best_strategy_on_valid(
                fold_valid_df, y_fold_valid, y_fold_valid_proba
            )

            chosen = best_stable if best_stable is not None else best_overall
            if chosen is None:
                fold_results.append(
                    {
                        "fold": fold_idx,
                        "return(%)": np.nan,
                        "bets": 0,
                        "stable": False,
                        "proba_mode": None,
                        "place_thr": np.nan,
                        "proba_thr": np.nan,
                        "ev_thr": np.nan,
                    }
                )
            else:
                fold_results.append(
                    {
                        "fold": fold_idx,
                        "return(%)": float(chosen["回収率(%)"]),
                        "bets": int(chosen["購入数"]),
                        "stable": best_stable is not None,
                        "proba_mode": str(chosen["確率モード"]),
                        "place_thr": float(chosen["複勝確率閾値"]),
                        "proba_thr": float(chosen["proba閾値"]),
                        "ev_thr": float(chosen["期待値閾値"]),
                    }
                )

        # CV結果の集計
        returns = [r["return(%)"] for r in fold_results if not np.isnan(r["return(%)"])]
        if len(returns) == 0:
            avg_return = np.nan
            min_return = np.nan
            cv_stable = False
        else:
            avg_return = float(np.mean(returns))
            min_return = float(np.min(returns))
            # 全Foldで80%以上の回収率があればstable
            cv_stable = all(r >= 80.0 for r in returns) and len(returns) == len(
                CV_FOLDS
            )

        # 最頻の戦略パラメータを採用（最終Foldを使用）
        final_fold_result = fold_results[-1]

        selection_rows.append(
            {
                "model": name,
                "cv_avg_return(%)": avg_return,
                "cv_min_return(%)": min_return,
                "fold_returns": returns,
                "cv_stable": cv_stable,
                "proba_mode": final_fold_result["proba_mode"],
                "place_thr": final_fold_result["place_thr"],
                "proba_thr": final_fold_result["proba_thr"],
                "ev_thr": final_fold_result["ev_thr"],
                "bets": final_fold_result["bets"],
            }
        )

        print(
            f"  Fold回収率: {returns} → 平均: {avg_return:.2f}%, 最小: {min_return:.2f}%, CV安定: {cv_stable}"
        )

        # 最良候補の更新
        # 優先順位: cv_stable > cv_avg_return > cv_min_return
        is_better = False
        if best_strategy is None:
            is_better = True
        elif cv_stable and not best_strategy.get("cv_stable", False):
            is_better = True
        elif cv_stable == best_strategy.get("cv_stable", False):
            if avg_return > best_cv_avg_return:
                is_better = True
            elif avg_return == best_cv_avg_return and min_return > best_cv_min_return:
                is_better = True

        if is_better and not np.isnan(avg_return):
            best_model = fold_models[-1]  # 最終Foldのモデルを採用
            best_strategy = selection_rows[-1]
            best_model_name = name
            best_cv_avg_return = avg_return
            best_cv_min_return = min_return

    print("\n[モデル選定結果] CVの平均回収率で比較")
    print(
        pd.DataFrame(selection_rows).sort_values(
            by=["cv_stable", "cv_avg_return(%)"], ascending=[False, False]
        )
    )

    if best_model is None or best_strategy is None or best_model_name is None:
        raise RuntimeError("モデル選定に失敗しました")

    # =========================================================================
    # 最終モデルの再学習（Train 0-70%で学習してTestを評価）
    # =========================================================================
    # CVで最良と判断されたモデルを、より多くのデータで再学習します。
    # CVではFoldごとに学習データが異なっていましたが、
    # 最終学習では Test 開始前（70%）までの全データを使用します。
    #
    # 注意: Test データは一切学習に使いません（リーク防止）
    print(f"\n[最終学習] 選定されたモデル {best_model_name} を全Trainデータで再学習...")

    final_train_end_idx = int(n_races * TEST_START)
    final_train_ids = unique_race_ids[:final_train_end_idx].tolist()
    final_train_mask = df["race_id"].isin(final_train_ids)

    final_train_df = df.loc[final_train_mask].copy()
    X_final_train = final_train_df[features]
    y_final_train = final_train_df["rank_class"].astype(int)

    # CVで選ばれたモデルのパラメータを取得
    final_params = next(
        c["params"] for c in model_candidates if c["name"] == best_model_name
    )

    final_clf = lgb.LGBMClassifier(
        objective="multiclass",  # 多クラス分類
        num_class=4,  # 0:その他, 1:1着, 2:2着, 3:3着
        metric="multi_logloss",  # 損失関数
        class_weight="balanced",  # クラス不均衡対策
        random_state=42,  # 再現性のため固定
        verbose=-1,  # ログ抑制
        **final_params,
    )

    # Early stoppingを使わずに学習（Testデータをリークさせない）
    final_clf.fit(X_final_train, y_final_train)

    best_model = final_clf

    # =========================================================================
    # 確率キャリブレーション（Isotonic Regression）
    # =========================================================================
    # LightGBMの出力する確率は、実際の「勝つ確率」とずれていることがあります。
    # 例えば「予測確率10%の馬」が実際には8%しか勝たないなど。
    #
    # Isotonic Regression を使って、予測確率を実績勝率に合わせることで、
    # 期待値（EV）の計算がより正確になります。
    #
    # 学習データ: Validデータ（Trainデータでキャリブレーションを学習すると過学習）
    from sklearn.isotonic import IsotonicRegression

    print("\n[キャリブレーション] Validデータで Isotonic Regression を学習...")

    # Validデータへの予測
    y_valid_pred_proba = np.asarray(final_clf.predict_proba(X_valid))
    valid_proba_1_raw = y_valid_pred_proba[:, 1]  # 1着の予測確率

    # 確率モードに応じた変換
    valid_win_proba = get_win_proba(
        valid_df, valid_proba_1_raw, proba_mode=str(best_strategy["proba_mode"])
    )

    # Isotonic Regressionの学習
    # 入力: 予測確率、出力: 実際に1着だったかどうか（0/1）
    y_valid_binary = (np.asarray(y_valid) == 1).astype(int)
    isotonic_calibrator = IsotonicRegression(out_of_bounds="clip")
    isotonic_calibrator.fit(valid_win_proba, y_valid_binary)

    print("  Isotonic Regression 学習完了")

    # 補正後EVで戦略を再探索（Valid）
    print(
        "\n[戦略再探索] Isotonic補正後EVでvalidの閾値を再探索... (方針=B: 実績/期待の信頼性重視)"
    )

    calibrated_ev_thresholds = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

    best_stable_cal, best_overall_cal = select_best_strategy_on_valid(
        valid_df,
        y_valid,
        y_valid_pred_proba,
        win_proba_modes_override=[str(best_strategy["proba_mode"])],
        ev_thresholds_override=calibrated_ev_thresholds,
        isotonic_calibrator=isotonic_calibrator,
        use_calibrated_proba=True,
        selection_objective="ev_reliability",
        ev_reliability_min_expected_return=0.0,
    )
    chosen_cal = best_stable_cal if best_stable_cal is not None else best_overall_cal

    calibrated_strategy = None
    if chosen_cal is None:
        print("  該当なし（補正後EVで戦略を選べませんでした）")
    else:
        calibrated_strategy = {
            "proba_mode": str(chosen_cal["確率モード"]),
            "place_thr": float(chosen_cal["複勝確率閾値"]),
            "proba_thr": float(chosen_cal["proba閾値"]),
            "ev_thr": float(chosen_cal["期待値閾値"]),
            "bets": int(chosen_cal["購入数"]),
            "valid_return(%)": float(chosen_cal["回収率(%)"]),
            "valid_expected_return(%)": float(chosen_cal["期待回収率(%)"]),
            "valid_return_vs_expected": float(chosen_cal["実績/期待"]),
            "stable": best_stable_cal is not None,
            "objective": "ev_reliability",
        }

        print(
            f"  [採用(補正後EV:B)] proba_mode={calibrated_strategy['proba_mode']}, place>={calibrated_strategy['place_thr']}, proba>={calibrated_strategy['proba_thr']}, EV>={calibrated_strategy['ev_thr']} "
            f"(valid回収率={calibrated_strategy['valid_return(%)']:.2f}%, 期待回収率={calibrated_strategy['valid_expected_return(%)']:.2f}%, 実績/期待={calibrated_strategy['valid_return_vs_expected']:.3f}, 購入数={calibrated_strategy['bets']}, stable={calibrated_strategy['stable']})"
        )

    # 参考: 回収率最大（方針=A）
    best_stable_cal_a, best_overall_cal_a = select_best_strategy_on_valid(
        valid_df,
        y_valid,
        y_valid_pred_proba,
        win_proba_modes_override=[str(best_strategy["proba_mode"])],
        ev_thresholds_override=calibrated_ev_thresholds,
        isotonic_calibrator=isotonic_calibrator,
        use_calibrated_proba=True,
        selection_objective="return",
    )
    chosen_cal_a = (
        best_stable_cal_a if best_stable_cal_a is not None else best_overall_cal_a
    )

    calibrated_strategy_a = None
    if chosen_cal_a is not None:
        calibrated_strategy_a = {
            "proba_mode": str(chosen_cal_a["確率モード"]),
            "place_thr": float(chosen_cal_a["複勝確率閾値"]),
            "proba_thr": float(chosen_cal_a["proba閾値"]),
            "ev_thr": float(chosen_cal_a["期待値閾値"]),
            "bets": int(chosen_cal_a["購入数"]),
            "valid_return(%)": float(chosen_cal_a["回収率(%)"]),
            "valid_expected_return(%)": float(chosen_cal_a["期待回収率(%)"]),
            "valid_return_vs_expected": float(chosen_cal_a["実績/期待"]),
            "stable": best_stable_cal_a is not None,
            "objective": "return",
        }

        print(
            f"  [参考(補正後EV:A)] proba_mode={calibrated_strategy_a['proba_mode']}, place>={calibrated_strategy_a['place_thr']}, proba>={calibrated_strategy_a['proba_thr']}, EV>={calibrated_strategy_a['ev_thr']} "
            f"(valid回収率={calibrated_strategy_a['valid_return(%)']:.2f}%, 期待回収率={calibrated_strategy_a['valid_expected_return(%)']:.2f}%, 実績/期待={calibrated_strategy_a['valid_return_vs_expected']:.3f}, 購入数={calibrated_strategy_a['bets']}, stable={calibrated_strategy_a['stable']})"
        )

    # Testデータの準備
    test_df = df.loc[test_mask].copy()
    X_test = test_df[features]
    y_test = test_df["rank_class"].astype(int)

    clf = best_model

    # CVで選ばれた戦略（補正前）
    cv_selected_proba_mode = str(best_strategy["proba_mode"])
    cv_selected_place_thr = float(best_strategy["place_thr"])
    cv_selected_proba_thr = float(best_strategy["proba_thr"])
    cv_selected_ev_thr = float(best_strategy["ev_thr"])

    # test評価に使う戦略（デフォルトは補正後EV）
    if calibrated_strategy is None:
        selected_proba_mode = cv_selected_proba_mode
        selected_place_thr = cv_selected_place_thr
        selected_proba_thr = cv_selected_proba_thr
        selected_ev_thr = cv_selected_ev_thr
        selected_strategy_label = "CV戦略"
    else:
        selected_proba_mode = str(calibrated_strategy["proba_mode"])
        selected_place_thr = float(calibrated_strategy["place_thr"])
        selected_proba_thr = float(calibrated_strategy["proba_thr"])
        selected_ev_thr = float(calibrated_strategy["ev_thr"])
        selected_strategy_label = "補正後EV戦略(B)"

    print(
        f"\n[採用] model={best_model_name}, cv_avg_return={best_strategy['cv_avg_return(%)']:.2f}%, cv_min={best_strategy['cv_min_return(%)']:.2f}%, (CV戦略) proba_mode={cv_selected_proba_mode}, place>={cv_selected_place_thr}, proba>={cv_selected_proba_thr}, EV>={cv_selected_ev_thr} (購入数={best_strategy['bets']}, cv_stable={best_strategy['cv_stable']})"
    )
    print(
        f"[採用] test評価に使用: {selected_strategy_label} proba_mode={selected_proba_mode}, place>={selected_place_thr}, proba>={selected_proba_thr}, EV>={selected_ev_thr}"
    )

    # =========================================================================
    # 6. モデル評価（Testデータ）
    # =========================================================================
    # Testデータは「未見のデータ」として、最終評価に1回だけ使用します。
    # ここでの結果が、実運用時のパフォーマンスの目安になります。
    print("\n評価結果:")

    # 予測の実行
    y_pred = clf.predict(X_test)  # クラス予測
    y_pred_proba = np.asarray(clf.predict_proba(X_test))  # 確率予測

    # --- 分類性能の評価 ---
    # Accuracy: 全体の正解率（ただし競馬では重要ではない）
    acc = accuracy_score(y_test, y_pred)
    print(f"正解率 (Accuracy): {acc:.4f}")

    # 分類レポート: 各クラスごとのPrecision/Recall/F1
    # 競馬では「1着の Precision（的中率）」が特に重要
    print("\n分類レポート:")
    target_names = ["その他", "1着", "2着", "3着"]
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=target_names,
            zero_division="warn",
        )
    )

    # --- 特徴量の重要度 ---
    # モデルが予測に使っている特徴量のランキング
    # 高い特徴量 = 予測に大きく寄与している
    print("\n特徴量の重要度:")
    importances = pd.DataFrame(
        {"feature": features, "importance": clf.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    print(importances)

    # =========================================================================
    # 7. 期待値（Expected Value）による回収率分析
    # =========================================================================
    # 競馬予測の最終目標は「回収率100%超え」です。
    #
    # 期待値（EV）= 予測勝率 × オッズ
    # - EV > 1.0 なら理論上はプラス期待値
    # - しかし実際にはEV > 1.5 程度でないと回収率100%は難しい
    #
    # このセクションでは、異なる戦略（閾値の組み合わせ）での
    # Test データにおける回収率を比較します。
    print("\n期待値（確率 × オッズ）によるパフォーマンス分析:")
    print(f"testで戦略を比較: CV戦略(補正前) vs {selected_strategy_label}")
    print(
        f"  CV戦略(補正前): proba_mode={cv_selected_proba_mode}, place>={cv_selected_place_thr}, proba>={cv_selected_proba_thr}, EV>={cv_selected_ev_thr}"
    )
    print(
        f"  {selected_strategy_label}: proba_mode={selected_proba_mode}, place>={selected_place_thr}, proba>={selected_proba_thr}, EV>={selected_ev_thr}"
    )

    # --- 期待値の計算 ---
    test_df["単勝オッズ"] = pd.to_numeric(test_df["単勝オッズ"], errors="coerce")

    # 1着確率（モデルの出力）
    win_proba_raw = y_pred_proba[:, 1]
    test_df["proba_1_raw"] = win_proba_raw

    # 戦略ごとの確率変換
    win_proba_cv = get_win_proba(
        test_df, win_proba_raw, proba_mode=cv_selected_proba_mode
    )
    win_proba = get_win_proba(test_df, win_proba_raw, proba_mode=selected_proba_mode)

    test_df["proba_1"] = win_proba
    # 複勝確率（3着以内）= 1着確率 + 2着確率 + 3着確率
    place_proba_test = y_pred_proba[:, 1] + y_pred_proba[:, 2] + y_pred_proba[:, 3]
    test_df["place_proba"] = place_proba_test

    # Isotonic補正後の確率を計算
    proba_1_calibrated = isotonic_calibrator.predict(win_proba)
    test_df["proba_1_calibrated"] = proba_1_calibrated

    odds = np.nan_to_num(np.asarray(test_df["単勝オッズ"]), nan=0.0)
    # 期待値 = 確率 × オッズ
    expected_values_cv = win_proba_cv * odds
    expected_values = win_proba * odds
    test_df["expected_value"] = expected_values

    # 補正後のEVも計算
    expected_values_calibrated = proba_1_calibrated * odds
    test_df["expected_value_calibrated"] = expected_values_calibrated

    y_test_array = np.asarray(y_test)
    n_winners = int(np.sum(y_test_array == 1))  # 全勝者数

    # --- 戦略比較関数 ---
    def summarize_strategy(
        mask,
        label: str,
        proba_thr: float,
        ev_thr: float,
        place_thr: float,
        ev_values_for_expectation,
    ):
        """
        指定された購入条件（mask）での回収率等を計算する

        Returns:
            dict: 戦略の各種指標
            - 購入数: 購入対象となる馬券数
            - 的中率: 購入した中で1着になった割合
            - 回収率: 購入額に対する払戻額の割合（100%超えが目標）
            - 期待回収率: 理論上の期待回収率（EVの平均）
            - 実績/期待: 実績回収率 / 期待回収率（1.0に近いほど予測が正確）
        """
        n_bets_local = int(np.sum(mask))
        if n_bets_local == 0:
            return {
                "戦略": label,
                "proba閾値": proba_thr,
                "期待値閾値": ev_thr,
                "複勝確率閾値": place_thr,
                "購入数": 0,
                "的中率(Precision)": np.nan,
                "再現率(Recall)": np.nan,
                "回収率(%)": np.nan,
                "期待回収率(%)": np.nan,
                "実績/期待": np.nan,
            }

        hit_mask_local = mask & (y_test_array == 1)  # 購入して当たった
        hit_count_local = int(np.sum(hit_mask_local))

        precision_local = hit_count_local / n_bets_local  # 的中率
        recall_local = hit_count_local / n_winners if n_winners > 0 else 0.0  # 再現率

        # 回収率 = 払戻金額 / 購入金額 × 100
        return_amount_local = float(np.sum(odds[hit_mask_local]))
        return_rate_local = (return_amount_local / n_bets_local) * 100

        expected_return_rate_local = float(
            np.mean(ev_values_for_expectation[mask]) * 100
        )
        ratio_local = (
            return_rate_local / expected_return_rate_local
            if expected_return_rate_local > 0
            else np.nan
        )

        return {
            "戦略": label,
            "proba閾値": proba_thr,
            "期待値閾値": ev_thr,
            "複勝確率閾値": place_thr,
            "購入数": n_bets_local,
            "的中率(Precision)": precision_local,
            "再現率(Recall)": recall_local,
            "回収率(%)": return_rate_local,
            "期待回収率(%)": expected_return_rate_local,
            "実績/期待": ratio_local,
        }

    strategy_mask_cv = (
        (win_proba_cv >= cv_selected_proba_thr)
        & (expected_values_cv >= cv_selected_ev_thr)
        & (odds >= FIXED_ODDS_MIN)
        & (place_proba_test >= cv_selected_place_thr)
    )

    # 採用戦略（補正後EV戦略なら補正後確率で評価）
    if calibrated_strategy is not None:
        selected_win_proba = proba_1_calibrated
        selected_ev_values = expected_values_calibrated
        selected_ev_column = "expected_value_calibrated"
        selected_proba_column = "proba_1_calibrated"
    else:
        selected_win_proba = win_proba
        selected_ev_values = expected_values
        selected_ev_column = "expected_value"
        selected_proba_column = "proba_1"

    strategy_mask_selected = (
        (selected_win_proba >= selected_proba_thr)
        & (selected_ev_values >= selected_ev_thr)
        & (odds >= FIXED_ODDS_MIN)
        & (place_proba_test >= selected_place_thr)
    )

    summary_rows = [
        summarize_strategy(
            strategy_mask_cv,
            "CV戦略(補正前)",
            cv_selected_proba_thr,
            cv_selected_ev_thr,
            cv_selected_place_thr,
            expected_values_cv,
        ),
        summarize_strategy(
            strategy_mask_selected,
            selected_strategy_label,
            selected_proba_thr,
            selected_ev_thr,
            selected_place_thr,
            selected_ev_values,
        ),
    ]

    if calibrated_strategy_a is not None:
        strategy_mask_a = (
            (proba_1_calibrated >= calibrated_strategy_a["proba_thr"])
            & (expected_values_calibrated >= calibrated_strategy_a["ev_thr"])
            & (odds >= FIXED_ODDS_MIN)
            & (place_proba_test >= calibrated_strategy_a["place_thr"])
        )
        summary_rows.append(
            summarize_strategy(
                strategy_mask_a,
                "補正後EV戦略(A)",
                calibrated_strategy_a["proba_thr"],
                calibrated_strategy_a["ev_thr"],
                calibrated_strategy_a["place_thr"],
                expected_values_calibrated,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df)

    # 以降の詳細分析は採用戦略で実施
    strategy_mask = strategy_mask_selected
    strategy_mask_series = pd.Series(strategy_mask, index=test_df.index)
    n_bets = int(np.sum(strategy_mask))

    # 8. フィルタ適用後の期待値上位サンプルの詳細分析
    print(f"\n期待値が高い馬のサンプル (TOP 10) [{selected_strategy_label}]:")

    if n_bets == 0:
        print("該当なし")
    else:
        filtered_results = test_df.loc[strategy_mask_series].copy()
        filtered_results["target"] = y_test_array[strategy_mask]

        # 期待値の高い順に並び替え（全購入対象）
        top_ev = filtered_results.sort_values(
            by=selected_ev_column, ascending=False
        ).head(10)
        print(
            top_ev[
                [
                    "年",
                    "月",
                    "日",
                    "レース名",
                    "馬名",
                    "単勝オッズ",
                    "proba_1",
                    "proba_1_calibrated",
                    "expected_value",
                    "expected_value_calibrated",
                    "target",
                ]
            ]
        )

        # 的中馬のみの期待値TOP10
        hit_results = filtered_results[filtered_results["target"] == 1]
        n_hits = len(hit_results)
        print(f"\n的中馬の期待値TOP10 (的中数={n_hits}):")
        if n_hits > 0:
            top_hit_ev = hit_results.sort_values(
                by=selected_ev_column, ascending=False
            ).head(10)
            print(
                top_hit_ev[
                    [
                        "年",
                        "月",
                        "日",
                        "レース名",
                        "馬名",
                        "単勝オッズ",
                        "proba_1",
                        "proba_1_calibrated",
                        "expected_value",
                        "expected_value_calibrated",
                        "target",
                    ]
                ]
            )
        else:
            print("的中馬なし")

        # 的中/非的中の比較分析
        print("\n的中/非的中の特徴量比較 [購入対象内]:")

        miss_results = filtered_results[filtered_results["target"] == 0]

        comparison_features = list(
            dict.fromkeys(
                features
                + [
                    "単勝オッズ",
                    "proba_1",
                    "proba_1_calibrated",
                    "expected_value",
                    "expected_value_calibrated",
                    "place_proba",
                ]
            )
        )

        numeric_features = [
            f
            for f in comparison_features
            if f in filtered_results.columns
            and pd.api.types.is_numeric_dtype(filtered_results[f])
        ]

        hit_means = hit_results[numeric_features].mean()
        hit_medians = hit_results[numeric_features].median()
        miss_means = miss_results[numeric_features].mean()
        miss_medians = miss_results[numeric_features].median()

        comparison_df = pd.DataFrame(
            {
                "hit_mean": hit_means,
                "hit_median": hit_medians,
                "miss_mean": miss_means,
                "miss_median": miss_medians,
                "diff": hit_means - miss_means,
            }
        )
        comparison_df = comparison_df.sort_values("diff", key=abs, ascending=False)
        print(comparison_df.head(20))

    # =========================================================================
    # 9. EV評価分析（フィルタ前の全テストデータ対象）
    # =========================================================================
    # このセクションでは、モデルの予測確率が「正しく調整されているか」を評価します。
    #
    # 理想的なモデル:
    # - EV（期待値）が高い馬ほど、実際の回収率も高い
    # - 期待回収率 ≒ 実績回収率（予測が現実と一致）
    #
    # 分析方法:
    # 1. 全馬をEVで10分位に分割
    # 2. 各分位での実績回収率を計算
    # 3. 期待回収率と実績回収率を比較
    print("\n" + "=" * 70)
    print("EV評価分析（全テストデータ対象）")
    print("=" * 70)

    # --- EV10分位ごとの実績集計 ---
    # 10分位 = データを10等分したグループ
    # 上位分位ほどEVが高い
    print("\n[EV10分位ごとの実績]")
    ev_analysis_df = test_df[["単勝オッズ", "proba_1", "expected_value"]].copy()
    ev_analysis_df["target"] = y_test_array
    ev_analysis_df["hit"] = (y_test_array == 1).astype(int)

    # EVが0より大きいデータのみ対象（オッズ欠損除外）
    ev_valid = ev_analysis_df[ev_analysis_df["expected_value"] > 0].copy()

    # 10分位に分割
    ev_valid["ev_decile"] = pd.qcut(
        ev_valid["expected_value"], q=10, labels=False, duplicates="drop"
    )

    decile_stats = (
        ev_valid.groupby("ev_decile")
        .agg(
            n_bets=("hit", "count"),
            n_hits=("hit", "sum"),
            avg_odds=("単勝オッズ", "mean"),
            avg_proba=("proba_1", "mean"),
            avg_ev=("expected_value", "mean"),
            ev_min=("expected_value", "min"),
            ev_max=("expected_value", "max"),
            total_return=(
                "単勝オッズ",
                lambda x: x[ev_valid.loc[x.index, "hit"] == 1].sum(),
            ),
        )
        .reset_index()
    )
    decile_stats["hit_rate"] = decile_stats["n_hits"] / decile_stats["n_bets"]
    decile_stats["return_rate"] = (
        decile_stats["total_return"] / decile_stats["n_bets"] * 100
    )
    decile_stats["expected_return"] = decile_stats["avg_ev"] * 100

    print(
        decile_stats[
            [
                "ev_decile",
                "n_bets",
                "n_hits",
                "hit_rate",
                "avg_odds",
                "avg_proba",
                "avg_ev",
                "ev_min",
                "ev_max",
                "return_rate",
                "expected_return",
            ]
        ].to_string(index=False)
    )

    # 期待回収率 vs 実績回収率
    print("\n[期待回収率 vs 実績回収率（EV分位別）]")
    decile_stats["return_vs_expected"] = (
        decile_stats["return_rate"] / decile_stats["expected_return"]
    )
    print(
        decile_stats[
            ["ev_decile", "expected_return", "return_rate", "return_vs_expected"]
        ].to_string(index=False)
    )

    # 全体サマリ
    total_bets = len(ev_valid)
    total_hits = ev_valid["hit"].sum()
    total_return = ev_valid.loc[ev_valid["hit"] == 1, "単勝オッズ"].sum()
    overall_return_rate = (total_return / total_bets) * 100
    overall_expected_return = ev_valid["expected_value"].mean() * 100

    print(f"\n[全体サマリ]")
    print(f"  購入数: {total_bets}")
    print(f"  的中数: {total_hits}")
    print(f"  的中率: {total_hits / total_bets:.4f}")
    print(f"  実績回収率: {overall_return_rate:.2f}%")
    print(f"  期待回収率(平均EV): {overall_expected_return:.2f}%")
    print(f"  実績/期待: {overall_return_rate / overall_expected_return:.4f}")

    # --- キャリブレーション評価 ---
    # キャリブレーションとは「予測確率の正確さ」を評価する指標です。
    #
    # 理想的なモデル:
    # 「予測確率10%の馬」は実際に10%の確率で勝つ
    #
    # 評価指標:
    # - Brier Score: 予測確率と実績（0/1）の二乗誤差の平均。低いほど良い。
    # - キャリブレーション曲線: 予測確率ごとの実績勝率をプロット
    print("\n[キャリブレーション評価]")
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    # 1着かどうかの2値
    y_true_binary = (y_test_array == 1).astype(int)
    y_prob = np.asarray(test_df["proba_1"])

    # Brier Score: 確率予測の精度を測る指標
    brier = brier_score_loss(y_true_binary, y_prob)
    print(f"  Brier Score: {brier:.6f} (低いほど良い)")

    # キャリブレーション曲線（10分割）
    avg_calib_error = None
    try:
        prob_true, prob_pred = calibration_curve(
            y_true_binary, y_prob, n_bins=10, strategy="quantile"
        )
        calib_df = pd.DataFrame(
            {"prob_pred (予測確率)": prob_pred, "prob_true (実績勝率)": prob_true}
        )
        calib_df["diff (実績-予測)"] = (
            calib_df["prob_true (実績勝率)"] - calib_df["prob_pred (予測確率)"]
        )
        print("\n  キャリブレーション曲線（10分位）:")
        print(calib_df.to_string(index=False))

        # 平均キャリブレーション誤差
        avg_calib_error = np.mean(np.abs(prob_true - prob_pred))
        print(f"\n  平均キャリブレーション誤差: {avg_calib_error:.6f}")
    except Exception as e:
        print(f"  キャリブレーション曲線の計算に失敗: {e}")

    # =========================================================================
    # 10. Isotonic補正後のEV評価（補正前後の比較）
    # =========================================================================
    # Isotonic Regression による確率補正の効果を評価します。
    #
    # 期待される効果:
    # - Brier Score の低下（予測精度の向上）
    # - キャリブレーション誤差の低下
    # - 実績/期待 が 1.0 に近づく（EVの信頼性向上）
    print("\n" + "=" * 70)
    print("Isotonic補正後のEV評価（補正前後の比較）")
    print("=" * 70)

    # 補正後のキャリブレーション評価
    y_prob_calibrated = np.asarray(test_df["proba_1_calibrated"])
    brier_calibrated = brier_score_loss(y_true_binary, y_prob_calibrated)
    print(f"\n[補正後] Brier Score: {brier_calibrated:.6f} (補正前: {brier:.6f})")

    try:
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_true_binary, y_prob_calibrated, n_bins=10, strategy="quantile"
        )
        calib_df_cal = pd.DataFrame(
            {
                "prob_pred (予測確率)": prob_pred_cal,
                "prob_true (実績勝率)": prob_true_cal,
            }
        )
        calib_df_cal["diff (実績-予測)"] = (
            calib_df_cal["prob_true (実績勝率)"] - calib_df_cal["prob_pred (予測確率)"]
        )
        print("\n  補正後キャリブレーション曲線（10分位）:")
        print(calib_df_cal.to_string(index=False))

        avg_calib_error_cal = np.mean(np.abs(prob_true_cal - prob_pred_cal))
        print(
            f"\n  補正後 平均キャリブレーション誤差: {avg_calib_error_cal:.6f} (補正前: {avg_calib_error:.6f})"
        )
    except Exception as e:
        print(f"  補正後キャリブレーション曲線の計算に失敗: {e}")
        avg_calib_error_cal = None

    # 補正後EV10分位ごとの実績集計
    print("\n[補正後EV 10分位ごとの実績]")
    ev_analysis_cal_df = test_df[
        ["単勝オッズ", "proba_1_calibrated", "expected_value_calibrated"]
    ].copy()
    ev_analysis_cal_df["target"] = y_test_array
    ev_analysis_cal_df["hit"] = (y_test_array == 1).astype(int)

    ev_valid_cal = ev_analysis_cal_df[
        ev_analysis_cal_df["expected_value_calibrated"] > 0
    ].copy()

    ev_valid_cal["ev_decile"] = pd.qcut(
        ev_valid_cal["expected_value_calibrated"], q=10, labels=False, duplicates="drop"
    )

    decile_stats_cal = (
        ev_valid_cal.groupby("ev_decile")
        .agg(
            n_bets=("hit", "count"),
            n_hits=("hit", "sum"),
            avg_odds=("単勝オッズ", "mean"),
            avg_proba=("proba_1_calibrated", "mean"),
            avg_ev=("expected_value_calibrated", "mean"),
            ev_min=("expected_value_calibrated", "min"),
            ev_max=("expected_value_calibrated", "max"),
            total_return=(
                "単勝オッズ",
                lambda x: x[ev_valid_cal.loc[x.index, "hit"] == 1].sum(),
            ),
        )
        .reset_index()
    )
    decile_stats_cal["hit_rate"] = (
        decile_stats_cal["n_hits"] / decile_stats_cal["n_bets"]
    )
    decile_stats_cal["return_rate"] = (
        decile_stats_cal["total_return"] / decile_stats_cal["n_bets"] * 100
    )
    decile_stats_cal["expected_return"] = decile_stats_cal["avg_ev"] * 100

    print(
        decile_stats_cal[
            [
                "ev_decile",
                "n_bets",
                "n_hits",
                "hit_rate",
                "avg_odds",
                "avg_proba",
                "avg_ev",
                "ev_min",
                "ev_max",
                "return_rate",
                "expected_return",
            ]
        ].to_string(index=False)
    )

    # 補正後の期待回収率 vs 実績回収率
    print("\n[補正後 期待回収率 vs 実績回収率（EV分位別）]")
    decile_stats_cal["return_vs_expected"] = (
        decile_stats_cal["return_rate"] / decile_stats_cal["expected_return"]
    )
    print(
        decile_stats_cal[
            ["ev_decile", "expected_return", "return_rate", "return_vs_expected"]
        ].to_string(index=False)
    )

    # 補正後全体サマリ
    total_bets_cal = len(ev_valid_cal)
    total_hits_cal = ev_valid_cal["hit"].sum()
    total_return_cal = ev_valid_cal.loc[ev_valid_cal["hit"] == 1, "単勝オッズ"].sum()
    overall_return_rate_cal = (total_return_cal / total_bets_cal) * 100
    overall_expected_return_cal = ev_valid_cal["expected_value_calibrated"].mean() * 100

    print(f"\n[補正後 全体サマリ]")
    print(f"  購入数: {total_bets_cal}")
    print(f"  的中数: {total_hits_cal}")
    print(f"  的中率: {total_hits_cal / total_bets_cal:.4f}")
    print(f"  実績回収率: {overall_return_rate_cal:.2f}%")
    print(f"  期待回収率(平均EV): {overall_expected_return_cal:.2f}%")
    print(f"  実績/期待: {overall_return_rate_cal / overall_expected_return_cal:.4f}")

    # 補正前後の比較サマリ
    print("\n[補正前後の比較サマリ]")
    print(f"  {'指標':<25} {'補正前':>12} {'補正後':>12} {'改善':>10}")
    print(f"  {'-' * 60}")
    print(
        f"  {'Brier Score':<25} {brier:>12.6f} {brier_calibrated:>12.6f} {'✓' if brier_calibrated < brier else ''}"
    )
    if avg_calib_error is not None and avg_calib_error_cal is not None:
        print(
            f"  {'平均キャリブ誤差':<25} {avg_calib_error:>12.6f} {avg_calib_error_cal:>12.6f} {'✓' if avg_calib_error_cal < avg_calib_error else ''}"
        )
    print(
        f"  {'期待回収率':<25} {overall_expected_return:>11.2f}% {overall_expected_return_cal:>11.2f}%"
    )
    print(
        f"  {'実績/期待':<25} {overall_return_rate / overall_expected_return:>12.4f} {overall_return_rate_cal / overall_expected_return_cal:>12.4f} {'✓' if (overall_return_rate_cal / overall_expected_return_cal) > (overall_return_rate / overall_expected_return) else ''}"
    )

    # 深掘り分析: 上位の馬の特徴量を表示
    if n_bets > 0:
        print("\n[深掘り分析] TOP3の馬の特徴量詳細:")
        features_to_show = list(
            dict.fromkeys(
                features
                + [
                    "単勝オッズ",
                    "proba_1",
                    "proba_1_calibrated",
                    "expected_value",
                    "expected_value_calibrated",
                ]
            )
        )

        # TOP3のインデックスを取得
        top3_indices = top_ev.index[:3]

        for idx in top3_indices:
            print(
                f"\n馬名: {filtered_results.loc[idx, '馬名']} (Target: {filtered_results.loc[idx, 'target']}, Odds: {filtered_results.loc[idx, '単勝オッズ']})"
            )
            horse_data = filtered_results.loc[idx, features_to_show]
            print(horse_data)


if __name__ == "__main__":
    main()
