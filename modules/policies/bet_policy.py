from abc import ABCMeta, abstractstaticmethod
import pandas as pd

class AbstractBetPolicy(metaclass=ABCMeta):
    """
    クラスの型を決めるための抽象クラス。
    """
    @abstractstaticmethod
    def judge(score_table, **params):
        """
        bet_dictは{race_id: {馬券の種類: 馬番のリスト}}の形式で返す。
        
        例)
        {'202101010101': {'tansho': [6, 8], 'fukusho': [4, 5]},
        '202101010102': {'tansho': [1], 'fukusho': [4]},
        '202101010103': {'tansho': [6], 'fukusho': []},
        '202101010104': {'tansho': [5], 'fukusho': [11]},
        ...}
        """
        pass
        
class BetPolicyTansho:
    """
    thresholdを超えた馬に単勝で賭ける戦略。
    """
    @staticmethod
    def judge(score_table: pd.DataFrame, threshold: float) -> dict:
        bet_dict = {}
        filtered_table = score_table.query('score >= @threshold')
        for race_id, table in filtered_table.groupby(level=0):
            bet_dict_1R = {}
            bet_dict_1R['tansho'] = list(table['馬番'])
            bet_dict[race_id] = bet_dict_1R
        return bet_dict
    
class BetPolicyFukusho:
    """
    thresholdを超えた馬に複勝で賭ける戦略。
    """
    @staticmethod
    def judge(score_table: pd.DataFrame, threshold: float) -> dict:
        bet_dict = {}
        filtered_table = score_table.query('score >= @threshold')
        for race_id, table in filtered_table.groupby(level=0):
            bet_dict_1R = {}
            bet_dict_1R['fukusho'] = list(table['馬番'])
            bet_dict[race_id] = bet_dict_1R
        return bet_dict
    
class BetPolicyUmarenBox:
    def judge(score_table: pd.DataFrame, threshold: float) -> dict:
        bet_dict = {}
        filtered_table = score_table.query('score >= @threshold')
        for race_id, table in filtered_table.groupby(level=0):
            if len(table) >= 2:
                bet_dict_1R = {}
                bet_dict_1R['fukusho'] = list(table['馬番'])
                bet_dict[race_id] = bet_dict_1R
        return bet_dict
    
class BetPolicyUmatanBox:
    def judge(score_table: pd.DataFrame, threshold: float) -> dict:
        bet_dict = {}
        filtered_table = score_table.query('score >= @threshold')
        for race_id, table in filtered_table.groupby(level=0):
            if len(table) >= 2:
                bet_dict_1R = {}
                bet_dict_1R['umatan'] = list(table['馬番'])
                bet_dict[race_id] = bet_dict_1R
        return bet_dict

class BetPolicyWideBox:
    def judge(score_table: pd.DataFrame, threshold: float) -> dict:
        bet_dict = {}
        filtered_table = score_table.query('score >= @threshold')
        for race_id, table in filtered_table.groupby(level=0):
            if len(table) >= 2:
                bet_dict_1R = {}
                bet_dict_1R['wide'] = list(table['馬番'])
                bet_dict[race_id] = bet_dict_1R
        return bet_dict

class BetPolicySanrenpukuBox:
    def judge(score_table: pd.DataFrame, threshold: float) -> dict:
        bet_dict = {}
        filtered_table = score_table.query('score >= @threshold')
        for race_id, table in filtered_table.groupby(level=0):
            if len(table) >= 3:
                bet_dict_1R = {}
                bet_dict_1R['sanrenpuku'] = list(table['馬番'])
                bet_dict[race_id] = bet_dict_1R
        return bet_dict
    
class BetPolicySanrentanBox:
    def judge(score_table: pd.DataFrame, threshold: float) -> dict:
        bet_dict = {}
        filtered_table = score_table.query('score >= @threshold')
        for race_id, table in filtered_table.groupby(level=0):
            if len(table) >= 3:
                bet_dict_1R = {}
                bet_dict_1R['sanrentan'] = list(table['馬番'])
                bet_dict[race_id] = bet_dict_1R
        return bet_dict
    
class BetPolicyUmatanNagashi:
    """
    threshold1を超えた馬を軸にし、threshold2を超えた馬を相手にして馬単で賭ける。（未実装）
    """
    def judge(score_table: pd.DataFrame, threshold1: float, threshold2: float) -> dict:
        bet_dict = {}
        filtered_table = score_table.query('score >= @threshold2')
        filtered_table['flg'] = filtered_table['score'].map(lambda x: 'jiku' if x >= threshold1 else 'aite')
        for race_id, table in filtered_table.groupby(level=0):
            bet_dict_1R = {}
            bet_dict_1R['tansho'] = list(table.query('flg == "tansho"')['馬番'])
            bet_dict_1R['fukusho'] = list(table.query('flg == "fukusho"')['馬番'])
            bet_dict[race_id] = bet_dict_1R
        return bet_dict