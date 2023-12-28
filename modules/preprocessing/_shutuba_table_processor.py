from importlib.resources import path
import pandas as pd
from ._results_processor import ResultsProcessor

class ShutubaTableProcessor(ResultsProcessor):
    def __init__(self, file_path: str):
        super().__init__([file_path])

    def _preprocess(self):
        df = super()._preprocess()
        df['開催'] = df.index.map(lambda x:str(x)[4:6])
        df["date"] = pd.to_datetime(df["date"])
        return df
    
    def _preprocess_rank(self, raw):
        return raw
    
    def _select_columns(self, raw):
        df = raw.copy()[[
            '枠番','馬番','斤量','単勝','horse_id','jockey_id','性', '年齢','体重','体重変化','n_horses',
            'course_len','weather','race_type','ground_state','date'
            ]]
        return df

