
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

class RaceFeatureEngineer:
    """
    レースデータの特徴量エンジニアリングを行うクラス
    
    主な処理フロー:
    1. Parquetファイルの読み込み (race_info, race_results)
    2. データのクリーニングと型変換 (数値変換、欠損処理など)
    3. テーブルの結合
    4. カテゴリ変数のエンコーディング (Label Encoding)
    5. ターゲット変数 (目的変数) の作成
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.race_info = None
        self.results = None
        self.merged_data = None
        
    def load_data(self):
        """
        保存済みのParquetファイルを読み込みます。
        - race_info.parquet: レース単位の情報（距離、天候、馬場状態など）
        - race_results.parquet: 出走馬ごとの結果情報（着順、タイム、オッズなど）
        """
        print("Loading data...")
        self.race_info = pd.read_parquet(os.path.join(self.data_dir, 'race_info.parquet'))
        self.results = pd.read_parquet(os.path.join(self.data_dir, 'race_results.parquet'))
        print(f"Loaded {len(self.race_info)} races and {len(self.results)} results.")

    def clean_data(self):
        """
        データのクリーニングと適切なデータ型への変換を行います。
        文字列として読み込まれた数値を数値型に変換し、変換できない値（例: "---" や取消）はNaNとして扱います。
        """
        print("Cleaning data...")
        
        # --- Race Info (レース情報) ---
        # 距離などは抽出時に数値化済みですが、念のため数値型であることを保証します
        self.race_info['distance'] = pd.to_numeric(self.race_info['distance'], errors='coerce')
        
        # --- Results (レース結果) ---
        # 着順: 数値以外（取消、中止、除外など）はNaNにして除外するか、大きな数字にします
        # 機械学習の学習データとしては、完走した馬（数値が入っているレコード）を中心にランク付けを行うのが一般的です
        self.results['rank'] = pd.to_numeric(self.results['rank'], errors='coerce')
        
        # 枠番、馬番を数値に変換
        self.results['frame_number'] = pd.to_numeric(self.results['frame_number'], errors='coerce')
        self.results['horse_number'] = pd.to_numeric(self.results['horse_number'], errors='coerce')
        
        # 性齢 ("牡3" など) -> 性別(category)と年齢(int)に分割して処理しやすい形にします
        # 性別: 牡, 牝, セ
        self.results['sex'] = self.results['sex_age'].str[0]
        self.results['age'] = pd.to_numeric(self.results['sex_age'].str[1:], errors='coerce')
        
        # 斤量（騎手の体重含む）
        self.results['jockey_weight'] = pd.to_numeric(self.results['jockey_weight'], errors='coerce')
        
        # 単勝オッズ
        self.results['odds'] = pd.to_numeric(self.results['odds'], errors='coerce')
        
        # 人気順
        self.results['popularity'] = pd.to_numeric(self.results['popularity'], errors='coerce')
        
        # タイム変換 (分:秒.ミリ秒 -> 秒数に統一)
        # 例: "1:58.3" -> 118.3秒
        # 欠損や "---" がある場合はNaNとします
        def convert_time(t_str):
            if not isinstance(t_str, str):
                return np.nan
            try:
                if ':' in t_str:
                    m, s = t_str.split(':')
                    return int(m) * 60 + float(s)
                else:
                    return float(t_str) # 秒だけの場合のケア
            except:
                return np.nan

        self.results['time_seconds'] = self.results['time'].apply(convert_time)
        
        # 分析や学習に不要になった元の列などを削除する場合はここで実施します
        # self.results.drop(columns=['sex_age', 'time'], inplace=True)

    def merge_data(self):
        """
        レース結果データ(results)に対し、共通のキー(race_id)を用いてレース情報(race_info)を結合します。
        これにより、各馬の戦績データに「そのレースの距離や天候」などの情報を付与します。
        """
        print("Merging data...")
        self.merged_data = pd.merge(self.results, self.race_info, on='race_id', how='left')
        
    def encode_features(self):
        """
        カテゴリ変数（文字列データなど）を機械学習モデルが扱える数値形式に変換します。
        ここでは Label Encoding（各カテゴリを一意の整数にマッピング）を使用しています。
        """
        print("Encoding features...")
        
        # エンコーディング対象のカラム
        # 決定木系のモデル（LightGBMなど）ではLabel Encodingで十分機能することが多いです
        cols_to_encode = ['race_id', 'horse_name', 'jockey', 'sex', 'surface', 'weather', 'track_condition']
        
        for col in cols_to_encode:
            if col in self.merged_data.columns:
                le = LabelEncoder()
                # 欠損値(NaN)があるとエラーになることがあるため、文字列型に変換してからエンコードします
                self.merged_data[col] = le.fit_transform(self.merged_data[col].astype(str))
                
    def create_target(self):
        """
        機械学習の予測ターゲット（目的変数）を作成します。
        例: 3着以内（馬券圏内）であれば 1、それ以外なら 0 とする2値分類ラベルを作成
        """
        self.merged_data['target'] = (self.merged_data['rank'] <= 3).astype(int)

    def process(self):
        """
        一連の特徴量エンジニアリング処理を実行し、加工済みデータを返します。
        """
        self.load_data()
        self.clean_data()
        self.merge_data()
        self.encode_features()
        self.create_target()
        return self.merged_data

if __name__ == "__main__":
    # データディレクトリの設定
    DATA_DIR = 'data/processed'
    
    # 必要な入力ファイルが存在するか確認してから実行
    if os.path.exists(os.path.join(DATA_DIR, 'race_info.parquet')):
        engineer = RaceFeatureEngineer(DATA_DIR)
        df = engineer.process()
        
        # 結果の確認（先頭5行とデータ情報の表示）
        print(df.head())
        print(df.info())
        
        # 加工済みデータを保存
        output_path = os.path.join(DATA_DIR, 'features.parquet')
        df.to_parquet(output_path, index=False)
        print(f"Saved features to {output_path}")
    else:
        print("Data not found. Please run html_parser.py first.")
