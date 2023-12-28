import pandas as pd
from ._abstract_data_processor import AbstractDataProcessor

class ResultsProcessor(AbstractDataProcessor):
    def __init__(self, path_list):
        super().__init__(path_list)
    
    def _preprocess(self):
        df = self.raw_data.copy()
        
        df = self._preprocess_rank(df)
        
        # 性齢を性と年齢に分ける
        df["性"] = df["性齢"].map(lambda x: str(x)[0])
        df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

        # 馬体重を体重と体重変化に分ける
        df["体重"] = df["馬体重"].str.split("(", expand=True)[0]
        df["体重変化"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1]
        
        #errors='coerce'で、"計不"など変換できない時に欠損値にする
        df['体重'] = pd.to_numeric(df['体重'], errors='coerce')
        df['体重変化'] = pd.to_numeric(df['体重変化'], errors='coerce')

        # 各列を数値型に変換
        df["単勝"] = df["単勝"].astype(float)
        df["斤量"] = df["斤量"].astype(float)
        df["枠番"] = df["枠番"].astype(int)
        df["馬番"] = df["馬番"].astype(int)
        
        #6/6出走数追加
        df['n_horses'] = df.index.map(df.index.value_counts())
        
        df = self._select_columns(df)
        
        return df
        
        
    def _preprocess_rank(self, raw):
        df = raw.copy()
        # 着順に数字以外の文字列が含まれているものを取り除く
        df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
        df.dropna(subset=['着順'], inplace=True)
        df['着順'] = df['着順'].astype(int)
        df['rank'] = df['着順'].map(lambda x:1 if x<4 else 0)
        return df
    
    def _select_columns(self, raw):
        df = raw.copy()[[
            '枠番','馬番','斤量','単勝','horse_id','jockey_id','性', '年齢','体重','体重変化','n_horses', 'rank'
            ]]
        return df
