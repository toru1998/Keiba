from numpy import dtype
import pandas as pd
from psutil import MACOS
from sklearn.preprocessing import LabelEncoder
from ._data_merger import DataMerger
from modules.constants import MergedDataCols as Cols
from modules.constants import Master

class FeatureEngineering:
    """
    使うテーブルを全てマージした後の処理をするクラス。
    新しい特徴量を作りたいときは、メソッド単位で追加していく。
    各メソッドは依存関係を持たないよう注意。
    """
    def __init__(self, data_merger: DataMerger):
        self.__data = data_merger.merged_data.copy()
        
    @property
    def featured_data(self):
        return self.__data
    
    def add_interval(self):
        """
        前走からの経過日数
        """
        self.__data['interval'] = (self.__data['date'] - self.__data['latest']).dt.days
        self.__data.drop('latest', axis=1, inplace=True)
        return self
    
    def dumminize_weather(self):
        """
        weatherカラムをダミー変数化する
        """
        self.__data[Cols.WEATHER] = pd.Categorical(self.__data[Cols.WEATHER], Master.WEATHER_LIST)
        self.__data = pd.get_dummies(self.__data, columns=[Cols.WEATHER])
        return self
    
    def dumminize_race_type(self):
        """
        race_typeカラムをダミー変数化する
        """
        self.__data[Cols.RACE_TYPE] = pd.Categorical(
            self.__data[Cols.RACE_TYPE], list(Master.RACE_TYPE_DICT.values())
            )
        self.__data = pd.get_dummies(self.__data, columns=[Cols.RACE_TYPE])
        return self
    
    def dumminize_ground_state(self):
        """
        ground_stateカラムをダミー変数化する
        """
        self.__data[Cols.GROUND_STATE] = pd.Categorical(
            self.__data[Cols.GROUND_STATE], Master.GROUND_STATE_LIST
            )
        self.__data = pd.get_dummies(self.__data, columns=[Cols.GROUND_STATE])
        return self
    
    def dumminize_sex(self):
        """
        sexカラムをダミー変数化する
        """
        self.__data[Cols.SEX] = pd.Categorical(self.__data[Cols.SEX], Master.SEX_LIST)
        self.__data = pd.get_dummies(self.__data, columns=[Cols.SEX])
        return self
    
    def encode_horse_id(self):
        """
        horse_idをラベルエンコーディングして、Categorical型に変換する。
        """
        csv_path = 'data/master/horse_id.csv'
        horse_master = pd.read_csv(csv_path, dtype=object)
        new_horses = self.__data[[Cols.HORSE_ID]][
            ~self.__data[Cols.HORSE_ID].isin(horse_master['horse_id'])
            ].drop_duplicates(subset=['horse_id'])
        new_horses['encoded_id'] = [i+len(horse_master) for i in range(len(new_horses))]
        new_horse_master = pd.concat([horse_master, new_horses]).set_index('horse_id')['encoded_id']
        new_horse_master.to_csv(csv_path)
        self.__data[Cols.HORSE_ID] = pd.Categorical(self.__data[Cols.HORSE_ID].map(new_horse_master))
        return self
    
    def encode_jockey_id(self):
        """
        jockey_idをラベルエンコーディングして、Categorical型に変換する。
        """
        csv_path = 'data/master/jockey_id.csv'
        jockey_master = pd.read_csv(csv_path, dtype=object)
        new_jockeys = self.__data[[Cols.JOCKEY_ID]][
            ~self.__data[Cols.JOCKEY_ID].isin(jockey_master['jockey_id'])
            ].drop_duplicates(subset=['jockey_id'])
        new_jockeys['encoded_id'] = [i+len(jockey_master) for i in range(len(new_jockeys))]
        new_jockey_master = pd.concat([jockey_master, new_jockeys]).set_index('jockey_id')['encoded_id']
        new_jockey_master.to_csv(csv_path)
        self.__data[Cols.JOCKEY_ID] = pd.Categorical(self.__data[Cols.JOCKEY_ID].map(new_jockey_master))
        return self
    
    def dumminize_kaisai(self):
        self.__data[Cols.KAISAI] = pd.Categorical(
            self.__data[Cols.KAISAI], list(Master.PLACE_DICT.values())
            )
        self.__data = pd.get_dummies(self.__data, columns=[Cols.KAISAI])
        return self