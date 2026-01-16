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
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import numpy as np


# =============================================================================
# 設定・定数
# =============================================================================

# --- ハイパーパラメータ ---
# 直近N走の成績から「現在の調子」を捉えるためのウィンドウ
ROLLING_WINDOW = 5

# 直近レースほど重みを大きくすることで「今の調子」を強調する
WEIGHT_PATTERNS = {
    "linear": [1, 2, 3, 4, 5],
    "exp": [0.0625, 0.125, 0.25, 0.5, 1],
    "front_heavy": [0.05, 0.1, 0.15, 0.3, 0.4],
}
# 過去の検証で相対的に安定したパターンを採用
SELECTED_WEIGHT = "exp"

# 適性特徴量の小標本ノイズを抑えるためのスムージング
USE_SMOOTHED_APTITUDE = True
APTITUDE_PRIOR_N = 50

# 距離適性を安定させるために距離をビン化する
DIST_BIN_WIDTH = 400

# ターゲットエンコーディングの小標本補正
TARGET_ENCODING_PRIOR_N = 50

# --- カラム定義 ---
# 元CSVにヘッダーがないため、列順を固定して読み込む
# ここを変えると読み込みデータがずれるため、仕様変更時のみ更新
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
    "PCI",
]

# --- 特徴量リスト ---
# レース開始前に確定している情報のみを採用（リーク防止）
FEATURES = [
    # 基本情報
    "場所",
    "所属地",
    "クラスコード",
    "芝・ダ",
    "トラックコード",
    "距離",
    "馬場状態",
    "性別",
    "年齢",
    "父馬名",
    "斤量",
    "馬体重",
    "頭数",
    "馬番",
    "騎手コード",
    "調教師コード",
    "生産者名",
    # 追加特徴量
    "長距離輸送",
    "斤量変化",
    "馬体重変化",
    # 適性特徴量
    "過去平均着順",
    "過去勝率",
    "過去複勝率",
    "過去出走数",
    "コース適性_平均着順",
    "コース適性_勝率",
    "コース適性_複勝率",
    "コース適性_出走数",
    "芝ダ適性_平均着順",
    "芝ダ適性_勝率",
    "芝ダ適性_複勝率",
    "芝ダ適性_出走数",
    "距離適性_平均着順",
    "距離適性_勝率",
    "距離適性_複勝率",
    "距離適性_出走数",
    # 脚質特徴量
    "prev_平均通過順",
    "prev_脚質指数",
    "avg5_通過順1",
    # 1走前情報
    "prev_場所",
    "prev_距離",
    "prev_芝・ダ",
    "prev_確定着順",
    "prev_着差タイム",
    # 直近N走の平均・加重平均
    f"avg{ROLLING_WINDOW}_上がり3Fタイム",
    f"avg{ROLLING_WINDOW}_PCI",
    f"wma{ROLLING_WINDOW}_上がり3Fタイム",
    f"wma{ROLLING_WINDOW}_PCI",
    f"wma{ROLLING_WINDOW}_着差タイム",
    # ターゲットエンコーディング
    "騎手勝率",
    "調教師勝率",
    "騎手場所適性",
    "種牡馬芝ダ適性",
    # クラス替わり
    "昇級フラグ",
    "降級フラグ",
    # レース間隔
    "interval",
]

# LightGBMにカテゴリとして扱わせる列（one-hotを避ける）
CATEGORICAL_FEATURES = [
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

# --- CV設定 ---
# 未来データのリークを避けるため、時系列でTrain/Validをずらす
CV_FOLDS = [
    {"train_end": 0.50, "valid_end": 0.60},
    {"train_end": 0.55, "valid_end": 0.65},
    {"train_end": 0.60, "valid_end": 0.70},
]
TEST_START = 0.70  # テスト期間は最後の30%

# --- 戦略パラメータ ---
# 購入閾値探索に使う範囲（計算量と安定性のバランスを取る）
STRATEGY_PARAMS = {
    "ev_thresholds": [1.0, 1.5, 2.0],
    "proba_thresholds": [0.0, 0.02, 0.05, 0.1, 0.15, 0.2],
    "place_proba_thresholds": [0.0, 0.4],
    "fixed_odds_min": 5.0,
    "min_bets_for_best": 8000,
}

# --- モデル候補 ---
# 正則化強度を変えた候補を用意し、過学習を避ける
MODEL_CANDIDATES = [
    {
        "name": "P1_regularized",
        "params": {
            "n_estimators": 1500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 100,
            "reg_alpha": 0.5,
            "reg_lambda": 1.0,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
        },
    },
    {
        "name": "P3_high_reg",
        "params": {
            "n_estimators": 1200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 150,
            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.7,
        },
    },
]
# =============================================================================
# ヘルパー関数
# =============================================================================


def weighted_mean_n(x, n=ROLLING_WINDOW, w=None):
    """直近n走の加重平均を計算する関数"""
    if w is None:
        w = np.array(WEIGHT_PATTERNS[SELECTED_WEIGHT])

    shifted = x.shift(1)  # 今走を除外

    def calc(window):
        valid = window.dropna()
        n_valid = len(valid)
        if n_valid == 0:
            return np.nan
        # 直近n_valid個分の重みを使用（末尾から取る）
        w_slice = w[-n_valid:]
        w_norm = w_slice / w_slice.sum()  # 正規化
        return (valid * w_norm).sum()

    return shifted.rolling(n, min_periods=1).apply(calc, raw=False)


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
    """
    win_proba_raw = y_pred_proba[:, 1]
    place_proba = y_pred_proba[:, 1] + y_pred_proba[:, 2] + y_pred_proba[:, 3]

    odds = np.nan_to_num(np.asarray(df_eval["単勝オッズ"]), nan=0.0)
    odds_mask = odds >= STRATEGY_PARAMS["fixed_odds_min"]

    y_true_array = np.asarray(y_true)
    n_winners = int(np.sum(y_true_array == 1))

    # パラメータ設定（オーバーライドなければデフォルト使用）
    # D-1対応: デフォルトで["raw", "race_sum"]の両方を比較
    win_proba_modes_local = win_proba_modes_override or ["raw", "race_sum"]
    proba_thresholds_local = (
        proba_thresholds_override or STRATEGY_PARAMS["proba_thresholds"]
    )
    ev_thresholds_local = ev_thresholds_override or STRATEGY_PARAMS["ev_thresholds"]
    place_proba_thresholds_local = (
        place_proba_thresholds_override or STRATEGY_PARAMS["place_proba_thresholds"]
    )
    min_bets = STRATEGY_PARAMS["min_bets_for_best"]

    sweep_results: list[dict] = []

    for proba_mode in win_proba_modes_local:
        win_proba = get_win_proba(df_eval, win_proba_raw, proba_mode=proba_mode)

        if use_calibrated_proba:
            if isotonic_calibrator is None:
                raise ValueError(
                    "use_calibrated_proba=True requires isotonic_calibrator"
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

    stable_df = sweep_df.loc[sweep_df["購入数"] >= min_bets]

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

    if selection_objective == "ev_reliability":

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

    raise ValueError(f"未知のselection_objectiveです: {selection_objective}")


# =============================================================================
# データ処理関数
# =============================================================================


def load_data():
    """CSVデータを読み込む"""
    print("データを読み込んでいます...")
    try:
        # 元データはcp932（Shift-JIS系）で保存されているため明示的に指定
        # header=Noneでヘッダー無しCSVを正しく読み込む
        # low_memory=Falseで型推定の警告を抑制
        df = pd.read_csv(
            "csv/data.csv",
            encoding="cp932",
            names=COLUMNS,
            header=None,
            low_memory=False,
        )
        return df
    except FileNotFoundError:
        print("エラー: csv/data.csv が見つかりませんでした。")
        return None


def preprocess_base(df):
    """基本的な前処理（型変換、ID作成、目的変数作成）"""
    print("前処理を実行中...")

    # レースIDの作成（末尾2桁=馬番を除外してレース単位にする）
    df["race_id"] = df["レースID"].astype(str).str.slice(0, -2)

    # 目的変数の作成（4クラス分類: 1着/2着/3着/その他）
    df["rank_class"] = pd.to_numeric(df["確定着順"], errors="coerce")
    df["rank_class"] = (
        df["rank_class"].where(df["rank_class"].isin([1, 2, 3]), 0).astype(int)
    )
    # 単勝予測用の2値フラグ
    df["is_win"] = (df["rank_class"] == 1).astype(int)

    # 日付・数値系の整形（前走間隔や時系列分割に必要）
    # 年は2桁で入っているため2000年代として補正
    df["年"] = pd.to_numeric(df["年"], errors="coerce") + 2000
    df["月"] = pd.to_numeric(df["月"], errors="coerce")
    df["日"] = pd.to_numeric(df["日"], errors="coerce")

    # 日付型を作ってレース間隔の計算に使う
    df["date"] = pd.to_datetime(
        df["年"].astype(str) + "-" + df["月"].astype(str) + "-" + df["日"].astype(str),
        format="%Y-%m-%d",
        errors="coerce",
    )

    # 数値型への変換（モデル入力で扱えるようにする）
    cols_to_numeric = [
        "レース番号",
        "距離",
        "斤量",
        "人気順",
        "単勝オッズ",
        "馬体重",
        "着差タイム",
        "上がり3Fタイム",
        "PCI",
        "通過順1",
        "通過順2",
        "通過順3",
        "通過順4",
    ]
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_lag_features(df):
    """1走前情報とレース間隔の追加"""
    # 前走情報として引き継ぐ列（血統登録番号で紐づける）
    prev_source_cols = [
        "場所",
        "距離",
        "芝・ダ",
        "確定着順",
        "着差タイム",
        "斤量",
        "馬体重",
        "通過順1",
        "通過順2",
        "通過順3",
        "通過順4",
        "date",
    ]

    # 同一馬の時系列を保証するため、血統登録番号 + 日付でソート
    df_sorted = df.sort_values(
        by=["血統登録番号", "年", "月", "日", "場所", "レース番号"]
    )

    # shift(1)で「前走のみ」を参照（リーク防止）
    for col in prev_source_cols:
        df[f"prev_{col}"] = df_sorted.groupby("血統登録番号")[col].shift(1)

    # レース間隔（日数）は休み明け判定や疲労の proxy として使う
    df["interval"] = (df["date"] - df["prev_date"]).dt.days

    # 前走からの変化量（状態変化を捉える）
    df["斤量変化"] = df["斤量"] - df["prev_斤量"]
    df["馬体重変化"] = df["馬体重"] - df["prev_馬体重"]

    # クラス替わり（昇級・降級の影響を捉える）
    df["prev_クラスコード"] = df_sorted.groupby("血統登録番号")["クラスコード"].shift(1)
    df["クラスコード_num"] = pd.to_numeric(df["クラスコード"], errors="coerce")
    df["prev_クラスコード_num"] = pd.to_numeric(
        df["prev_クラスコード"], errors="coerce"
    )
    df["クラス変化"] = df["クラスコード_num"] - df["prev_クラスコード_num"]
    df["昇級フラグ"] = (df["クラス変化"] > 0).astype(int)
    df["降級フラグ"] = (df["クラス変化"] < 0).astype(int)

    # 脚質特徴量（前走の位置取り傾向を数値化）
    df["prev_平均通過順"] = df[
        ["prev_通過順1", "prev_通過順2", "prev_通過順3", "prev_通過順4"]
    ].mean(axis=1)
    # 序盤(1角) − 終盤(4角): 正なら先行型、負なら追い込み型
    df["prev_脚質指数"] = df["prev_通過順1"] - df["prev_通過順4"]

    return df


def add_aptitude_features(df):
    """
    適性特徴量（過去統計）の追加

    B-1/E-2対応: 同日の結果を使わないよう日単位でブロックし、
    global priorも時点整合（過去日までの累積平均）にする
    """
    # 確定着順を数値化
    df["確定着順_num"] = pd.to_numeric(df["確定着順"], errors="coerce")
    # 距離をビン化して近い距離をまとめ、サンプル不足の揺らぎを抑える
    df["距離bin"] = (df["距離"] // DIST_BIN_WIDTH) * DIST_BIN_WIDTH

    # 時系列順に並べ、過去レースのみを参照できるようにする
    df_sorted = df.sort_values(by=["年", "月", "日", "場所", "レース番号"]).copy()

    # 累積計算用の補助列（NaN対応）
    df_sorted["_rank"] = df_sorted["確定着順_num"]
    df_sorted["_valid"] = df_sorted["_rank"].notna().astype(int)
    df_sorted["_rank_filled"] = df_sorted["_rank"].fillna(0)
    df_sorted["_win"] = (df_sorted["_rank"] == 1).astype(int)
    df_sorted["_place"] = (df_sorted["_rank"] <= 3).astype(int)

    prior_n = APTITUDE_PRIOR_N

    # =========================================================================
    # B-1: global priorを時点整合（過去日までの累積平均）にする
    # =========================================================================
    # 日単位で集計し、過去日までの累積を計算
    daily_global = (
        df_sorted.groupby("date", sort=True)
        .agg(
            day_valid=("_valid", "sum"),
            day_rank_sum=("_rank_filled", "sum"),
            day_win=("_win", "sum"),
            day_place=("_place", "sum"),
        )
        .reset_index()
    )
    # 過去日までの累積（当日を含まない）
    daily_global["cum_valid"] = (
        daily_global["day_valid"].cumsum().shift(1, fill_value=0)
    )
    daily_global["cum_rank_sum"] = (
        daily_global["day_rank_sum"].cumsum().shift(1, fill_value=0)
    )
    daily_global["cum_win"] = daily_global["day_win"].cumsum().shift(1, fill_value=0)
    daily_global["cum_place"] = (
        daily_global["day_place"].cumsum().shift(1, fill_value=0)
    )

    # 時点整合のglobal prior（その日より前までの全体平均）
    daily_global["global_avg_rank"] = (
        daily_global["cum_rank_sum"] / daily_global["cum_valid"]
    ).where(daily_global["cum_valid"] > 0, np.nan)
    daily_global["global_win_rate"] = (
        daily_global["cum_win"] / daily_global["cum_valid"]
    ).where(daily_global["cum_valid"] > 0, np.nan)
    daily_global["global_place_rate"] = (
        daily_global["cum_place"] / daily_global["cum_valid"]
    ).where(daily_global["cum_valid"] > 0, np.nan)

    # 各行にglobal priorをマージ
    df_sorted = df_sorted.merge(
        daily_global[
            ["date", "global_avg_rank", "global_win_rate", "global_place_rate"]
        ],
        on="date",
        how="left",
    )

    # =========================================================================
    # E-2: 同日ブロック - 日単位で集約してから累積を取る
    # =========================================================================
    def calc_stats_daily_block(group_keys, prefix):
        """
        日単位でブロックして適性特徴量を計算する
        同日の他レース結果を使わないように、(groupキー + date)で日次集約→
        date単位でcumsum→当日分を除外して各行へ割り当て
        """
        if isinstance(group_keys, str):
            group_keys = [group_keys]
        else:
            group_keys = list(group_keys)

        # Step1: (groupキー + date)で日次集約
        daily_agg = (
            df_sorted.groupby(group_keys + ["date"], sort=False, dropna=False)
            .agg(
                day_valid=("_valid", "sum"),
                day_rank_sum=("_rank_filled", "sum"),
                day_win=("_win", "sum"),
                day_place=("_place", "sum"),
            )
            .reset_index()
        )

        # Step2: groupキー内で日付順にソートして累積（当日を含まない=shift）
        daily_agg = daily_agg.sort_values(by=group_keys + ["date"])
        daily_agg["cum_valid"] = (
            daily_agg.groupby(group_keys, dropna=False)["day_valid"]
            .cumsum()
            .shift(1, fill_value=0)
        )
        daily_agg["cum_rank_sum"] = (
            daily_agg.groupby(group_keys, dropna=False)["day_rank_sum"]
            .cumsum()
            .shift(1, fill_value=0)
        )
        daily_agg["cum_win"] = (
            daily_agg.groupby(group_keys, dropna=False)["day_win"]
            .cumsum()
            .shift(1, fill_value=0)
        )
        daily_agg["cum_place"] = (
            daily_agg.groupby(group_keys, dropna=False)["day_place"]
            .cumsum()
            .shift(1, fill_value=0)
        )

        # Step3: 各行にマージ
        merge_cols = group_keys + ["date"]
        merged = df_sorted[merge_cols].merge(
            daily_agg[
                merge_cols + ["cum_valid", "cum_rank_sum", "cum_win", "cum_place"]
            ],
            on=merge_cols,
            how="left",
        )

        cnt = merged["cum_valid"].values
        rank_sum = merged["cum_rank_sum"].values
        win_sum = merged["cum_win"].values
        place_sum = merged["cum_place"].values

        # 時点整合のglobal priorを使用
        g_avg_rank = df_sorted["global_avg_rank"].values
        g_win_rate = df_sorted["global_win_rate"].values
        g_place_rate = df_sorted["global_place_rate"].values

        # Step4: スムージング付きで特徴量を計算
        if USE_SMOOTHED_APTITUDE:
            denom = cnt + prior_n
            # global priorがNaNの場合（最初期）はprior_nで割るだけ
            avg_rank = np.where(
                denom > 0,
                (rank_sum + prior_n * np.nan_to_num(g_avg_rank, nan=0)) / denom,
                np.nan,
            )
            win_rate = np.where(
                denom > 0,
                (win_sum + prior_n * np.nan_to_num(g_win_rate, nan=0)) / denom,
                np.nan,
            )
            place_rate = np.where(
                denom > 0,
                (place_sum + prior_n * np.nan_to_num(g_place_rate, nan=0)) / denom,
                np.nan,
            )
        else:
            avg_rank = np.where(cnt > 0, rank_sum / cnt, np.nan)
            win_rate = np.where(cnt > 0, win_sum / cnt, np.nan)
            place_rate = np.where(cnt > 0, place_sum / cnt, np.nan)

        # df_sortedの順序でセットし、元のdfのindex順に戻す
        df.loc[df_sorted.index, f"{prefix}出走数"] = cnt
        df.loc[df_sorted.index, f"{prefix}平均着順"] = avg_rank
        df.loc[df_sorted.index, f"{prefix}勝率"] = win_rate
        df.loc[df_sorted.index, f"{prefix}複勝率"] = place_rate

    # 1. 過去全レース（総合成績のベースライン）
    calc_stats_daily_block("血統登録番号", "過去")

    # 2. コース適性（競馬場ごとの差を捉える）
    calc_stats_daily_block(["血統登録番号", "場所"], "コース適性_")

    # 3. 芝ダ適性（芝/ダートの向き不向き）
    calc_stats_daily_block(["血統登録番号", "芝・ダ"], "芝ダ適性_")

    # 4. 距離適性（得意距離帯を捉える）
    calc_stats_daily_block(["血統登録番号", "距離bin"], "距離適性_")

    return df


def add_trend_features(df):
    """直近N走の傾向（移動平均など）を追加"""
    # 直近成績のみを使うため、馬単位で時系列に並べ替える
    df_sorted = df.sort_values(
        by=["血統登録番号", "年", "月", "日", "場所", "レース番号"]
    )

    # 調子の指標として使うカラム（終いの脚・ペース指標）
    agg_cols = ["上がり3Fタイム", "PCI"]

    # 単純移動平均（shiftで今走を除外してリーク防止）
    for col in agg_cols:
        df[f"avg{ROLLING_WINDOW}_{col}"] = (
            df_sorted.groupby("血統登録番号")[col]
            .apply(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    # 加重移動平均（上がり3F, PCI）
    for col in agg_cols:
        df[f"wma{ROLLING_WINDOW}_{col}"] = (
            df_sorted.groupby("血統登録番号")[col]
            .apply(weighted_mean_n)
            .reset_index(level=0, drop=True)
        )

    # 着差タイムの加重平均
    df[f"wma{ROLLING_WINDOW}_着差タイム"] = (
        df_sorted.groupby("血統登録番号")["着差タイム"]
        .apply(weighted_mean_n)
        .reset_index(level=0, drop=True)
    )

    # 直近5走の通過順平均
    df["avg5_通過順1"] = (
        df_sorted.groupby("血統登録番号")["通過順1"]
        .apply(lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    return df


def add_encoding_features(df):
    """カテゴリ変数のエンコーディングとフラグ作成"""
    # 輸送フラグ（長距離輸送による負荷を簡易的に表現）
    KANTO_PLACES = ["福島", "新潟", "東京", "中山"]
    KANSAI_PLACES = ["中京", "京都", "阪神", "小倉"]
    HOKKAIDO_PLACES = ["札幌", "函館"]

    place = df["場所"].astype(str)
    center = df["所属地"].astype(str)

    is_kanto_race = place.isin(KANTO_PLACES)
    is_kansai_race = place.isin(KANSAI_PLACES)
    is_hokkaido_race = place.isin(HOKKAIDO_PLACES)

    df["長距離輸送"] = 0
    df.loc[(center == "美") & (is_kansai_race | is_hokkaido_race), "長距離輸送"] = 1
    df.loc[(center == "栗") & (is_kanto_race | is_hokkaido_race), "長距離輸送"] = 1

    # ターゲットエンコーディング（過去勝率で騎手/調教師の強さを表現）
    # B-2/E-2対応: 同日の結果を使わないよう日単位でブロックし、
    # global priorも時点整合（過去日までの累積平均）にする
    df_sorted_te = df.sort_values(by=["年", "月", "日", "場所", "レース番号"]).copy()
    df_sorted_te["_win"] = df_sorted_te["is_win"]
    df_sorted_te["_valid"] = 1

    # 日単位でglobal win rateの累積を計算（時点整合のprior）
    daily_global_te = (
        df_sorted_te.groupby("date", sort=True)
        .agg(day_valid=("_valid", "sum"), day_win=("_win", "sum"))
        .reset_index()
    )
    daily_global_te["cum_valid"] = (
        daily_global_te["day_valid"].cumsum().shift(1, fill_value=0)
    )
    daily_global_te["cum_win"] = (
        daily_global_te["day_win"].cumsum().shift(1, fill_value=0)
    )
    daily_global_te["global_win_rate_te"] = (
        daily_global_te["cum_win"] / daily_global_te["cum_valid"]
    ).where(daily_global_te["cum_valid"] > 0, 0.0)

    # 各行にglobal priorをマージ
    df_sorted_te = df_sorted_te.merge(
        daily_global_te[["date", "global_win_rate_te"]], on="date", how="left"
    )

    def calc_te(cols, name):
        """
        日単位でブロックしてターゲットエンコーディングを計算
        同日の他レース結果を使わないように、(キー + date)で日次集約→
        date単位でcumsum→当日分を除外
        """
        cols_list = [cols] if isinstance(cols, str) else list(cols)

        # Step1: (キー + date)で日次集約
        daily_agg = (
            df_sorted_te.groupby(cols_list + ["date"], sort=False, dropna=False)
            .agg(day_valid=("_valid", "sum"), day_win=("_win", "sum"))
            .reset_index()
        )

        # Step2: キー内で日付順にソートして累積（当日を含まない=shift相当）
        daily_agg = daily_agg.sort_values(by=cols_list + ["date"])
        daily_agg["cum_valid"] = (
            daily_agg.groupby(cols_list, dropna=False)["day_valid"]
            .cumsum()
            .shift(1, fill_value=0)
        )
        daily_agg["cum_win"] = (
            daily_agg.groupby(cols_list, dropna=False)["day_win"]
            .cumsum()
            .shift(1, fill_value=0)
        )

        # Step3: 各行にマージ
        merge_cols = cols_list + ["date"]
        merged = df_sorted_te[merge_cols + ["global_win_rate_te"]].merge(
            daily_agg[merge_cols + ["cum_valid", "cum_win"]],
            on=merge_cols,
            how="left",
        )

        cnt = merged["cum_valid"].values
        win_sum = merged["cum_win"].values
        g_win_rate = merged["global_win_rate_te"].values

        # スムージング付きでTEを計算（時点整合のglobal priorを使用）
        denom = cnt + TARGET_ENCODING_PRIOR_N
        te_values = (
            win_sum + TARGET_ENCODING_PRIOR_N * np.nan_to_num(g_win_rate, nan=0)
        ) / denom

        df.loc[df_sorted_te.index, name] = te_values

    calc_te("騎手コード", "騎手勝率")
    calc_te("調教師コード", "調教師勝率")
    calc_te(["騎手コード", "場所"], "騎手場所適性")
    calc_te(["父馬名", "芝・ダ"], "種牡馬芝ダ適性")

    # LightGBMにカテゴリとして渡すため型を変換
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def create_features(df):
    """全特徴量の生成パイプライン"""
    # 前走情報 → 適性 → 直近傾向 → エンコーディングの順で依存を解消
    df = add_lag_features(df)
    df = add_aptitude_features(df)
    df = add_trend_features(df)
    df = add_encoding_features(df)
    return df


# =============================================================================
# 学習・評価関数
# =============================================================================


def get_fold_masks(df, unique_race_ids, fold_config):
    """CV用のマスクを取得"""
    n_races = len(unique_race_ids)
    train_end_idx = int(n_races * fold_config["train_end"])
    valid_end_idx = int(n_races * fold_config["valid_end"])
    train_ids = unique_race_ids[:train_end_idx].tolist()
    valid_ids = unique_race_ids[train_end_idx:valid_end_idx].tolist()
    train_m = df["race_id"].isin(train_ids)
    valid_m = df["race_id"].isin(valid_ids)
    return train_m, valid_m


def train_and_evaluate_cv(df):
    """
    時系列CVによるモデル選定と戦略探索

    C-1/C-2対応:
    - 各foldのvalid期間を前半/後半に分割し、前半で閾値選択、後半で評価
    - foldごとに選んだ閾値を集計して中央値を最終戦略とする
    """
    # ランダム分割では未来情報が混ざるため、時系列で固定
    df_sorted = df.sort_values(by=["年", "月", "日", "場所", "レース番号"])
    unique_race_ids = df_sorted["race_id"].unique()

    print(
        f"\n[時系列CV] {len(CV_FOLDS)}つのFoldでモデルを評価します（valid内ネスト探索）"
    )

    selection_rows = []
    best_model = None
    best_strategy = None
    best_model_name = None
    best_cv_avg_return = -np.inf
    best_cv_min_return = -np.inf

    for cand in MODEL_CANDIDATES:
        name = cand["name"]
        params = cand["params"]
        print(f"\n[候補] {name} を評価...")

        fold_results = []
        fold_thresholds = []  # C-2: 各foldで選んだ閾値を保存
        current_fold_models = []

        for fold_idx, fold_config in enumerate(CV_FOLDS):
            train_mask, valid_mask = get_fold_masks(df, unique_race_ids, fold_config)

            X_train = df.loc[train_mask, FEATURES]
            y_train = df.loc[train_mask, "rank_class"].astype(int)
            X_valid = df.loc[valid_mask, FEATURES]
            y_valid = df.loc[valid_mask, "rank_class"].astype(int)
            valid_df_subset = df.loc[valid_mask].copy()

            # E-1対応: class_weight="balanced"を削除
            clf = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=4,
                metric="multi_logloss",
                random_state=42,
                verbose=-1,
                **params,
            )

            clf.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(0),
                ],
            )
            current_fold_models.append(clf)

            # C-1対応: validを前半/後半に分割してネスト探索
            y_valid_proba = np.asarray(clf.predict_proba(X_valid))

            # valid期間内のrace_idを取得して前半/後半に分割
            valid_race_ids = valid_df_subset["race_id"].unique()
            n_valid_races = len(valid_race_ids)
            split_idx = n_valid_races // 2

            valid_first_half_ids = set(valid_race_ids[:split_idx])
            valid_second_half_ids = set(valid_race_ids[split_idx:])

            first_half_mask = valid_df_subset["race_id"].isin(valid_first_half_ids)
            second_half_mask = valid_df_subset["race_id"].isin(valid_second_half_ids)

            # 前半で閾値を選択
            valid_first_df = valid_df_subset.loc[first_half_mask].copy()
            y_valid_first = y_valid.loc[first_half_mask]
            y_valid_proba_first = y_valid_proba[first_half_mask.values]

            best_stable_first, best_overall_first = select_best_strategy_on_valid(
                valid_first_df, y_valid_first, y_valid_proba_first
            )
            chosen_first = (
                best_stable_first
                if best_stable_first is not None
                else best_overall_first
            )

            if chosen_first is None:
                fold_results.append({"return(%)": np.nan})
                continue

            # 選んだ閾値を保存
            selected_proba_mode = str(chosen_first["確率モード"])
            selected_place_thr = float(chosen_first["複勝確率閾値"])
            selected_proba_thr = float(chosen_first["proba閾値"])
            selected_ev_thr = float(chosen_first["期待値閾値"])

            fold_thresholds.append(
                {
                    "proba_mode": selected_proba_mode,
                    "place_thr": selected_place_thr,
                    "proba_thr": selected_proba_thr,
                    "ev_thr": selected_ev_thr,
                }
            )

            # 後半で選んだ閾値を使って評価
            valid_second_df = valid_df_subset.loc[second_half_mask].copy()
            y_valid_second = y_valid.loc[second_half_mask]
            y_valid_proba_second = y_valid_proba[second_half_mask.values]

            # 後半データで回収率を計算
            win_proba_raw_second = y_valid_proba_second[:, 1]
            win_proba_second = get_win_proba(
                valid_second_df, win_proba_raw_second, proba_mode=selected_proba_mode
            )
            place_proba_second = (
                y_valid_proba_second[:, 1]
                + y_valid_proba_second[:, 2]
                + y_valid_proba_second[:, 3]
            )

            odds_second = np.nan_to_num(
                np.asarray(valid_second_df["単勝オッズ"]), nan=0.0
            )
            expected_values_second = win_proba_second * odds_second

            y_valid_second_array = np.asarray(y_valid_second)
            n_winners_second = int(np.sum(y_valid_second_array == 1))

            strategy_mask_second = (
                (win_proba_second >= selected_proba_thr)
                & (expected_values_second >= selected_ev_thr)
                & (odds_second >= STRATEGY_PARAMS["fixed_odds_min"])
                & (place_proba_second >= selected_place_thr)
            )

            n_bets_second = int(np.sum(strategy_mask_second))
            if n_bets_second == 0:
                fold_results.append({"return(%)": np.nan})
                continue

            hit_mask_second = strategy_mask_second & (y_valid_second_array == 1)
            hit_count_second = int(np.sum(hit_mask_second))

            return_amount_second = float(np.sum(odds_second[hit_mask_second]))
            return_rate_second = (return_amount_second / n_bets_second) * 100

            fold_results.append(
                {
                    "return(%)": return_rate_second,
                    "bets": n_bets_second,
                    "proba_mode": selected_proba_mode,
                    "place_thr": selected_place_thr,
                    "proba_thr": selected_proba_thr,
                    "ev_thr": selected_ev_thr,
                    "stable": best_stable_first is not None,
                }
            )

        # CV結果集計
        returns = [
            r["return(%)"]
            for r in fold_results
            if "return(%)" in r and not np.isnan(r["return(%)"])
        ]
        if not returns:
            avg_return = np.nan
            min_return = np.nan
            cv_stable = False
        else:
            avg_return = float(np.mean(returns))
            min_return = float(np.min(returns))
            # すべてのFoldで一定以上の回収率を出せるかを安定性指標にする
            cv_stable = all(r >= 80.0 for r in returns) and len(returns) == len(
                CV_FOLDS
            )

        # C-2対応: foldごとの閾値を中央値で集計
        if fold_thresholds:
            # proba_modeは最頻値を使用
            proba_modes = [t["proba_mode"] for t in fold_thresholds]
            median_proba_mode = max(set(proba_modes), key=proba_modes.count)

            # 数値閾値は中央値を使用
            median_place_thr = float(
                np.median([t["place_thr"] for t in fold_thresholds])
            )
            median_proba_thr = float(
                np.median([t["proba_thr"] for t in fold_thresholds])
            )
            median_ev_thr = float(np.median([t["ev_thr"] for t in fold_thresholds]))

            # 最終foldの購入数を参考値として使用
            last_bets = fold_results[-1].get("bets", 0) if fold_results else 0

            row = {
                "model": name,
                "cv_avg_return(%)": avg_return,
                "cv_min_return(%)": min_return,
                "cv_stable": cv_stable,
                "proba_mode": median_proba_mode,
                "place_thr": median_place_thr,
                "proba_thr": median_proba_thr,
                "ev_thr": median_ev_thr,
                "bets": last_bets,
            }
            selection_rows.append(row)

            print(
                f"  Fold回収率(後半評価): {returns} → 平均: {avg_return:.2f}%, CV安定: {cv_stable}"
            )
            print(
                f"  閾値中央値: proba_mode={median_proba_mode}, place>={median_place_thr}, "
                f"proba>={median_proba_thr}, EV>={median_ev_thr}"
            )

            # 最良候補更新（安定性 > 平均回収率 > 最小回収率の順で重視）
            is_better = False
            if best_strategy is None:
                is_better = True
            elif cv_stable and not best_strategy.get("cv_stable", False):
                is_better = True
            elif cv_stable == best_strategy.get("cv_stable", False):
                if avg_return > best_cv_avg_return:
                    is_better = True
                elif (
                    avg_return == best_cv_avg_return and min_return > best_cv_min_return
                ):
                    is_better = True

            if is_better:
                best_model = current_fold_models[-1]
                best_strategy = row
                best_model_name = name
                best_cv_avg_return = avg_return
                best_cv_min_return = min_return

    print("\n[モデル選定結果]")
    if selection_rows:
        print(
            pd.DataFrame(selection_rows).sort_values(
                by=["cv_stable", "cv_avg_return(%)"], ascending=[False, False]
            )
        )

    return best_model_name, best_strategy


def train_final_and_calibrate(df, best_model_name, best_strategy):
    """最終モデルの学習とキャリブレーション"""
    print(f"\n[最終学習] {best_model_name} を学習中...")

    df_sorted = df.sort_values(by=["年", "月", "日", "場所", "レース番号"])
    unique_race_ids = df_sorted["race_id"].unique()
    n_races = len(unique_race_ids)

    # Train/Valid分割（Testは除外）
    # Validデータ(60-70%)はキャリブレーション学習用に残すため、Trainには含めない
    cv_last_fold = CV_FOLDS[-1]
    train_end_idx = int(n_races * cv_last_fold["train_end"])
    valid_end_idx = int(n_races * TEST_START)  # Test開始直前まで

    train_ids = unique_race_ids[:train_end_idx].tolist()
    valid_ids = unique_race_ids[train_end_idx:valid_end_idx].tolist()

    train_mask = df["race_id"].isin(train_ids)
    valid_mask = df["race_id"].isin(valid_ids)

    X_train = df.loc[train_mask, FEATURES]
    y_train = df.loc[train_mask, "rank_class"].astype(int)
    X_valid = df.loc[valid_mask, FEATURES]
    y_valid = df.loc[valid_mask, "rank_class"].astype(int)
    valid_df = df.loc[valid_mask].copy()

    # パラメータ取得
    params = next(c["params"] for c in MODEL_CANDIDATES if c["name"] == best_model_name)

    # E-1対応: class_weight="balanced"を削除
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,
        metric="multi_logloss",
        random_state=42,
        verbose=-1,
        **params,
    )

    # 全Trainデータで学習
    clf.fit(X_train, y_train)

    # --- キャリブレーション ---
    print("\n[キャリブレーション] Isotonic Regression 学習...")
    y_valid_proba = np.asarray(clf.predict_proba(X_valid))
    valid_proba_1 = y_valid_proba[:, 1]

    # 戦略の確率モードに合わせて変換
    valid_win_proba = get_win_proba(
        valid_df, valid_proba_1, proba_mode=best_strategy["proba_mode"]
    )

    y_valid_binary = (np.asarray(y_valid) == 1).astype(int)
    iso_cal = IsotonicRegression(out_of_bounds="clip")
    iso_cal.fit(valid_win_proba, y_valid_binary)

    # 補正後EVで戦略再探索（方針B: 実績/期待の信頼性重視）
    print(
        "\n[戦略再探索] Isotonic補正後EVでvalidの閾値を再探索... (方針=B: 実績/期待の信頼性重視)"
    )
    calibrated_ev_thresholds = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

    best_stable_cal, best_overall_cal = select_best_strategy_on_valid(
        valid_df,
        y_valid,
        y_valid_proba,
        win_proba_modes_override=[best_strategy["proba_mode"]],
        ev_thresholds_override=calibrated_ev_thresholds,
        isotonic_calibrator=iso_cal,
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
            "  [採用(補正後EV:B)] "
            f"proba_mode={calibrated_strategy['proba_mode']}, "
            f"place>={calibrated_strategy['place_thr']}, "
            f"proba>={calibrated_strategy['proba_thr']}, "
            f"EV>={calibrated_strategy['ev_thr']} "
            f"(valid回収率={calibrated_strategy['valid_return(%)']:.2f}%, "
            f"期待回収率={calibrated_strategy['valid_expected_return(%)']:.2f}%, "
            f"実績/期待={calibrated_strategy['valid_return_vs_expected']:.3f}, "
            f"購入数={calibrated_strategy['bets']}, "
            f"stable={calibrated_strategy['stable']})"
        )

    # 参考: 回収率最大（方針A）
    best_stable_cal_a, best_overall_cal_a = select_best_strategy_on_valid(
        valid_df,
        y_valid,
        y_valid_proba,
        win_proba_modes_override=[best_strategy["proba_mode"]],
        ev_thresholds_override=calibrated_ev_thresholds,
        isotonic_calibrator=iso_cal,
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
            "  [参考(補正後EV:A)] "
            f"proba_mode={calibrated_strategy_a['proba_mode']}, "
            f"place>={calibrated_strategy_a['place_thr']}, "
            f"proba>={calibrated_strategy_a['proba_thr']}, "
            f"EV>={calibrated_strategy_a['ev_thr']} "
            f"(valid回収率={calibrated_strategy_a['valid_return(%)']:.2f}%, "
            f"期待回収率={calibrated_strategy_a['valid_expected_return(%)']:.2f}%, "
            f"実績/期待={calibrated_strategy_a['valid_return_vs_expected']:.3f}, "
            f"購入数={calibrated_strategy_a['bets']}, "
            f"stable={calibrated_strategy_a['stable']})"
        )

    return clf, iso_cal, calibrated_strategy, calibrated_strategy_a


def evaluate_model(
    df,
    clf,
    iso_cal,
    best_model_name,
    best_strategy,
    calibrated_strategy,
    calibrated_strategy_a,
):
    """Testデータでの最終評価"""
    print("\n" + "=" * 70)
    print("最終評価 (Test Data)")
    print("=" * 70)

    df_sorted = df.sort_values(by=["年", "月", "日", "場所", "レース番号"])
    unique_race_ids = df_sorted["race_id"].unique()
    n_races = len(unique_race_ids)
    # 最終評価は最後の期間だけで実施（過去学習の汎化性能を見る）
    test_start_idx = int(n_races * TEST_START)
    test_race_ids = unique_race_ids[test_start_idx:].tolist()
    test_mask = df["race_id"].isin(test_race_ids)

    test_df = df.loc[test_mask].copy()
    X_test = test_df[FEATURES]
    y_test = test_df["rank_class"].astype(int)

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
        f"\n[採用] model={best_model_name}, cv_avg_return={best_strategy['cv_avg_return(%)']:.2f}%, "
        f"cv_min={best_strategy['cv_min_return(%)']:.2f}%, (CV戦略) "
        f"proba_mode={cv_selected_proba_mode}, place>={cv_selected_place_thr}, "
        f"proba>={cv_selected_proba_thr}, EV>={cv_selected_ev_thr} "
        f"(購入数={best_strategy['bets']}, cv_stable={best_strategy['cv_stable']})"
    )
    print(
        f"[採用] test評価に使用: {selected_strategy_label} proba_mode={selected_proba_mode}, "
        f"place>={selected_place_thr}, proba>={selected_proba_thr}, EV>={selected_ev_thr}"
    )

    # =========================================================================
    # 6. モデル評価（Testデータ）
    # =========================================================================
    print("\n評価結果:")

    y_pred = clf.predict(X_test)
    y_pred_proba = np.asarray(clf.predict_proba(X_test))

    acc = accuracy_score(y_test, y_pred)
    print(f"正解率 (Accuracy): {acc:.4f}")

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

    print("\n特徴量の重要度:")
    importances = pd.DataFrame(
        {"feature": FEATURES, "importance": clf.feature_importances_}
    ).sort_values(by="importance", ascending=False)
    print(importances)

    # =========================================================================
    # 7. 期待値（Expected Value）による回収率分析
    # =========================================================================
    print("\n期待値（確率 × オッズ）によるパフォーマンス分析:")
    print(f"testで戦略を比較: CV戦略(補正前) vs {selected_strategy_label}")
    print(
        f"  CV戦略(補正前): proba_mode={cv_selected_proba_mode}, place>={cv_selected_place_thr}, proba>={cv_selected_proba_thr}, EV>={cv_selected_ev_thr}"
    )
    print(
        f"  {selected_strategy_label}: proba_mode={selected_proba_mode}, place>={selected_place_thr}, proba>={selected_proba_thr}, EV>={selected_ev_thr}"
    )

    test_df["単勝オッズ"] = pd.to_numeric(test_df["単勝オッズ"], errors="coerce")

    win_proba_raw = y_pred_proba[:, 1]
    test_df["proba_1_raw"] = win_proba_raw

    win_proba_cv = get_win_proba(
        test_df, win_proba_raw, proba_mode=cv_selected_proba_mode
    )
    win_proba = get_win_proba(test_df, win_proba_raw, proba_mode=selected_proba_mode)

    test_df["proba_1"] = win_proba
    place_proba_test = y_pred_proba[:, 1] + y_pred_proba[:, 2] + y_pred_proba[:, 3]
    test_df["place_proba"] = place_proba_test

    proba_1_calibrated = iso_cal.predict(win_proba)
    test_df["proba_1_calibrated"] = proba_1_calibrated

    odds = np.nan_to_num(np.asarray(test_df["単勝オッズ"]), nan=0.0)
    expected_values_cv = win_proba_cv * odds
    expected_values = win_proba * odds
    test_df["expected_value"] = expected_values

    expected_values_calibrated = proba_1_calibrated * odds
    test_df["expected_value_calibrated"] = expected_values_calibrated

    y_test_array = np.asarray(y_test)
    n_winners = int(np.sum(y_test_array == 1))

    print("\n[予測確率の整合性チェック（確率10分位）]")
    y_true_binary = (y_test_array == 1).astype(int)

    diag_df = test_df[["単勝オッズ", "proba_1", "proba_1_calibrated"]].copy()
    diag_df["hit"] = y_true_binary
    diag_df = diag_df.dropna(subset=["単勝オッズ", "proba_1", "proba_1_calibrated"])

    if diag_df.empty:
        print("  集計対象がありません（単勝オッズ/確率の欠損が多い可能性）")
    else:

        def _summarize_by_prob_quantile(proba_col: str, label: str):
            local = diag_df.copy()
            local["q"] = pd.qcut(
                local[proba_col], q=10, labels=False, duplicates="drop"
            )
            summary = (
                local.groupby("q")
                .agg(
                    n=("hit", "count"),
                    actual_win_rate=("hit", "mean"),
                    mean_proba=(proba_col, "mean"),
                    mean_odds=("単勝オッズ", "mean"),
                )
                .reset_index()
            )
            summary["predicted_ev"] = summary["mean_proba"] * summary["mean_odds"]
            summary["actual_ev"] = summary["actual_win_rate"] * summary["mean_odds"]
            summary["actual/pred"] = summary["actual_ev"] / summary["predicted_ev"]
            print(f"\n  {label}:")
            print(summary.to_string(index=False))

        _summarize_by_prob_quantile("proba_1", "補正前 proba_1")
        _summarize_by_prob_quantile(
            "proba_1_calibrated", "Isotonic補正後 proba_1_calibrated"
        )

    def summarize_strategy(
        mask,
        label: str,
        proba_thr: float,
        ev_thr: float,
        place_thr: float,
        ev_values_for_expectation,
    ):
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

        hit_mask_local = mask & (y_test_array == 1)
        hit_count_local = int(np.sum(hit_mask_local))

        precision_local = hit_count_local / n_bets_local
        recall_local = hit_count_local / n_winners if n_winners > 0 else 0.0

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
        & (odds >= STRATEGY_PARAMS["fixed_odds_min"])
        & (place_proba_test >= cv_selected_place_thr)
    )

    if calibrated_strategy is not None:
        selected_win_proba = proba_1_calibrated
        selected_ev_values = expected_values_calibrated
        selected_ev_column = "expected_value_calibrated"
    else:
        selected_win_proba = win_proba
        selected_ev_values = expected_values
        selected_ev_column = "expected_value"

    strategy_mask_selected = (
        (selected_win_proba >= selected_proba_thr)
        & (selected_ev_values >= selected_ev_thr)
        & (odds >= STRATEGY_PARAMS["fixed_odds_min"])
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
            & (odds >= STRATEGY_PARAMS["fixed_odds_min"])
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

    strategy_mask = strategy_mask_selected
    strategy_mask_series = pd.Series(strategy_mask, index=test_df.index)
    n_bets = int(np.sum(strategy_mask))

    print(f"\n期待値が高い馬のサンプル (TOP 10) [{selected_strategy_label}]:")

    filtered_results = None
    top_ev = None
    hit_results = None

    if n_bets == 0:
        print("該当なし")
    else:
        filtered_results = test_df.loc[strategy_mask_series].copy()
        filtered_results["target"] = y_test_array[strategy_mask]

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

        print("\n的中/非的中の特徴量比較 [購入対象内]:")

        miss_results = filtered_results[filtered_results["target"] == 0]

        comparison_features = list(
            dict.fromkeys(
                FEATURES
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

        hit_means = (
            hit_results[numeric_features].mean() if hit_results is not None else None
        )
        hit_medians = (
            hit_results[numeric_features].median() if hit_results is not None else None
        )
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
    print("\n" + "=" * 70)
    print("EV評価分析（全テストデータ対象）")
    print("=" * 70)

    print("\n[EV10分位ごとの実績]")
    ev_analysis_df = test_df[["単勝オッズ", "proba_1", "expected_value"]].copy()
    ev_analysis_df["target"] = y_test_array
    ev_analysis_df["hit"] = (y_test_array == 1).astype(int)

    ev_valid = ev_analysis_df[ev_analysis_df["expected_value"] > 0].copy()

    if ev_valid.empty:
        print("  EVが正のデータがありません")
        total_bets = 0
        total_hits = 0
        overall_return_rate = np.nan
        overall_expected_return = np.nan
        ratio = np.nan

        print(f"\n[全体サマリ]")
        print(f"  購入数: {total_bets}")
        print(f"  的中数: {total_hits}")
        print("  的中率: -")
        print("  実績回収率: -")
        print("  期待回収率(平均EV): -")
        print("  実績/期待: -")
    else:
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

        print("\n[期待回収率 vs 実績回収率（EV分位別）]")
        decile_stats["return_vs_expected"] = (
            decile_stats["return_rate"] / decile_stats["expected_return"]
        )
        print(
            decile_stats[
                ["ev_decile", "expected_return", "return_rate", "return_vs_expected"]
            ].to_string(index=False)
        )

        total_bets = len(ev_valid)
        total_hits = ev_valid["hit"].sum()
        total_return = ev_valid.loc[ev_valid["hit"] == 1, "単勝オッズ"].sum()
        overall_return_rate = (total_return / total_bets) * 100
        overall_expected_return = ev_valid["expected_value"].mean() * 100
        ratio = (
            overall_return_rate / overall_expected_return
            if overall_expected_return > 0
            else np.nan
        )

        print(f"\n[全体サマリ]")
        print(f"  購入数: {total_bets}")
        print(f"  的中数: {total_hits}")
        print(f"  的中率: {total_hits / total_bets:.4f}")
        print(f"  実績回収率: {overall_return_rate:.2f}%")
        print(f"  期待回収率(平均EV): {overall_expected_return:.2f}%")
        print(f"  実績/期待: {ratio:.4f}")

    print("\n[キャリブレーション評価]")

    y_true_binary = (y_test_array == 1).astype(int)
    y_prob = np.asarray(test_df["proba_1"])

    brier = brier_score_loss(y_true_binary, y_prob)
    print(f"  Brier Score: {brier:.6f} (低いほど良い)")

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

        avg_calib_error = np.mean(np.abs(prob_true - prob_pred))
        print(f"\n  平均キャリブレーション誤差: {avg_calib_error:.6f}")
    except Exception as e:
        print(f"  キャリブレーション曲線の計算に失敗: {e}")

    print("\n" + "=" * 70)
    print("Isotonic補正後のEV評価（補正前後の比較）")
    print("=" * 70)

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

    print("\n[補正後EV 10分位ごとの実績]")
    ev_analysis_cal_df = test_df[
        ["単勝オッズ", "proba_1_calibrated", "expected_value_calibrated"]
    ].copy()
    ev_analysis_cal_df["target"] = y_test_array
    ev_analysis_cal_df["hit"] = (y_test_array == 1).astype(int)

    ev_valid_cal = ev_analysis_cal_df[
        ev_analysis_cal_df["expected_value_calibrated"] > 0
    ].copy()

    if ev_valid_cal.empty:
        print("  補正後EVが正のデータがありません")
        total_bets_cal = 0
        total_hits_cal = 0
        overall_return_rate_cal = np.nan
        overall_expected_return_cal = np.nan
        ratio_cal = np.nan

        print(f"\n[補正後 全体サマリ]")
        print(f"  購入数: {total_bets_cal}")
        print(f"  的中数: {total_hits_cal}")
        print("  的中率: -")
        print("  実績回収率: -")
        print("  期待回収率(平均EV): -")
        print("  実績/期待: -")
    else:
        ev_valid_cal["ev_decile"] = pd.qcut(
            ev_valid_cal["expected_value_calibrated"],
            q=10,
            labels=False,
            duplicates="drop",
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

        print("\n[補正後 期待回収率 vs 実績回収率（EV分位別）]")
        decile_stats_cal["return_vs_expected"] = (
            decile_stats_cal["return_rate"] / decile_stats_cal["expected_return"]
        )
        print(
            decile_stats_cal[
                ["ev_decile", "expected_return", "return_rate", "return_vs_expected"]
            ].to_string(index=False)
        )

        total_bets_cal = len(ev_valid_cal)
        total_hits_cal = ev_valid_cal["hit"].sum()
        total_return_cal = ev_valid_cal.loc[
            ev_valid_cal["hit"] == 1, "単勝オッズ"
        ].sum()
        overall_return_rate_cal = (total_return_cal / total_bets_cal) * 100
        overall_expected_return_cal = (
            ev_valid_cal["expected_value_calibrated"].mean() * 100
        )
        ratio_cal = (
            overall_return_rate_cal / overall_expected_return_cal
            if overall_expected_return_cal > 0
            else np.nan
        )

        print(f"\n[補正後 全体サマリ]")
        print(f"  購入数: {total_bets_cal}")
        print(f"  的中数: {total_hits_cal}")
        print(f"  的中率: {total_hits_cal / total_bets_cal:.4f}")
        print(f"  実績回収率: {overall_return_rate_cal:.2f}%")
        print(f"  期待回収率(平均EV): {overall_expected_return_cal:.2f}%")
        print(f"  実績/期待: {ratio_cal:.4f}")

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

    if n_bets > 0 and filtered_results is not None and top_ev is not None:
        print("\n[深掘り分析] TOP3の馬の特徴量詳細:")
        features_to_show = list(
            dict.fromkeys(
                FEATURES
                + [
                    "単勝オッズ",
                    "proba_1",
                    "proba_1_calibrated",
                    "expected_value",
                    "expected_value_calibrated",
                ]
            )
        )

        top3_indices = top_ev.index[:3]

        for idx in top3_indices:
            print(
                f"\n馬名: {filtered_results.loc[idx, '馬名']} (Target: {filtered_results.loc[idx, 'target']}, Odds: {filtered_results.loc[idx, '単勝オッズ']})"
            )
            horse_data = filtered_results.loc[idx, features_to_show]
            print(horse_data)


# =============================================================================
# Main
# =============================================================================


def main():
    """メイン処理"""
    # 1. データ読み込み
    df = load_data()
    if df is None:
        return

    # 2. 前処理 & 特徴量生成
    df = preprocess_base(df)
    df = create_features(df)

    print(f"\n特徴量生成完了: {df.shape}")
    print(f"採用特徴量数: {len(FEATURES)}")

    # 3. 時系列CVでモデルと購入戦略を選定
    best_model_name, best_strategy = train_and_evaluate_cv(df)

    if best_model_name is None:
        print("モデル選定に失敗しました")
        return

    print(f"\n選定モデル: {best_model_name}")
    print(f"選定戦略: {best_strategy}")

    # 4. 最終学習 & キャリブレーション（未見期間で補正）
    clf, iso_cal, calibrated_strategy, calibrated_strategy_a = (
        train_final_and_calibrate(df, best_model_name, best_strategy)
    )

    # 5. 最終評価
    evaluate_model(
        df,
        clf,
        iso_cal,
        best_model_name,
        best_strategy,
        calibrated_strategy,
        calibrated_strategy_a,
    )


if __name__ == "__main__":
    main()
