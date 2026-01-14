import gzip
from io import StringIO
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_race_html(
    html_content: str, race_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """HTMLからレース結果、払い戻し、ラップタイムを抽出してDataFrameに変換"""
    try:
        dfs = pd.read_html(StringIO(html_content))
        if not dfs:
            return None

        # 1. レース結果 (data[0])
        result_df = dfs[0]
        result_df["race_id"] = race_id

        # 2. 払い戻し (data[1], data[2])
        # 構造が複雑なため、単純に結合して race_id を付与する
        # 実運用ではより詳細なクリーニングが必要になる可能性あり
        try:
            payout_df = pd.concat([dfs[1], dfs[2]], ignore_index=True)
            payout_df["race_id"] = race_id
        except (IndexError, ValueError):
            payout_df = pd.DataFrame()

        # 3. ラップタイム (data[5])
        # インデックスが変わる可能性を考慮し、"ラップ"が含まれるテーブルを探すなどのロジックも考えられるが
        # ここでは固定インデックスで取得を試みる
        try:
            lap_df = dfs[5]
            lap_df["race_id"] = race_id
        except (IndexError, ValueError):
            lap_df = pd.DataFrame()

        return result_df, payout_df, lap_df

    except (ValueError, IndexError):
        return None
    except Exception:
        return None


def load_all_race_html(
    data_dir: str = "data/html/race",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """指定ディレクトリ内の全HTMLファイルを読み込んで結合"""
    data_path = Path(data_dir)
    html_files = list(data_path.glob("*.html.gz"))

    print(f"Found {len(html_files)} HTML files")

    results_list = []
    payouts_list = []
    laps_list = []

    for html_file in tqdm(html_files, desc="Processing HTML files"):
        race_id = html_file.stem.replace(".html", "")
        try:
            with gzip.open(html_file, "rt", encoding="utf-8") as f:
                html_content = f.read()

            extracted = parse_race_html(html_content, race_id)
            if extracted is not None:
                r_df, p_df, l_df = extracted
                results_list.append(r_df)
                if not p_df.empty:
                    payouts_list.append(p_df)
                if not l_df.empty:
                    laps_list.append(l_df)

        except Exception as e:
            print(f"Error reading {html_file}: {e}")
            continue

    if not results_list:
        print("No dataframes to concatenate")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"Concatenating dataframes...")
    combined_results = pd.concat(results_list, ignore_index=True)
    combined_results.set_index("race_id", inplace=True)

    combined_payouts = pd.DataFrame()
    if payouts_list:
        combined_payouts = pd.concat(payouts_list, ignore_index=True)
        combined_payouts.set_index("race_id", inplace=True)

    combined_laps = pd.DataFrame()
    if laps_list:
        combined_laps = pd.concat(laps_list, ignore_index=True)
        combined_laps.set_index("race_id", inplace=True)

    print(f"Results rows: {len(combined_results)}")
    print(f"Payouts rows: {len(combined_payouts)}")
    print(f"Laps rows: {len(combined_laps)}")

    return combined_results, combined_payouts, combined_laps


if __name__ == "__main__":
    results_df, payouts_df, laps_df = load_all_race_html()

    results_df.to_pickle("data/race_results.pkl")
    payouts_df.to_pickle("data/race_payouts.pkl")
    laps_df.to_pickle("data/race_laps.pkl")

    print("\n--- Race Results Sample ---")
    print(results_df.head())
    print("\n--- Payouts Sample ---")
    print(payouts_df.head())
    print("\n--- Laps Sample ---")
    print(laps_df.head())
