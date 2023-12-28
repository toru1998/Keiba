import pandas as pd

from ._data_merger import DataMerger
from modules.preprocessing import ShutubaTableProcessor
from modules.preprocessing import HorseResultsProcessor
from modules.preprocessing import PedsProcessor

class ShutubaDataMerger(DataMerger):
    def __init__(self,
                 shutuba_table_processor: ShutubaTableProcessor, 
                 horse_results_processor: HorseResultsProcessor, 
                 peds_processor: PedsProcessor, 
                 target_cols: list, 
                 group_cols: list
                 ):
        self._results = shutuba_table_processor.preprocessed_data #レース結果
        self._horse_results = horse_results_processor.preprocessed_data #馬の過去成績
        self._peds = peds_processor.preprocessed_data #血統
        self._target_cols = target_cols #集計対象列
        self._group_cols = group_cols #horse_idと一緒にターゲットエンコーディングしたいカテゴリ変数
        self._merged_data = pd.DataFrame() #全てのマージが完了したデータ
        self._separated_results_dict = {} #日付(date列)ごとに分かれたレース結果
        self._separated_horse_results_dict = {} #レース結果データのdateごとに分かれた馬の過去成績
        
    def merge(self):
        self._merge_horse_results()
        self._merge_peds()