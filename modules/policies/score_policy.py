from abc import ABCMeta, abstractstaticmethod
import pandas as pd

class AbstractScorePolicy(metaclass=ABCMeta):
    @abstractstaticmethod
    def calc(model, X: pd.DataFrame):
        raise NotImplementedError

class BasicScorePolicy(AbstractScorePolicy):
    """
    LightGBMの出力をそのままscoreとして計算。
    """
    @staticmethod
    def calc(model, X: pd.DataFrame):
        score_table = X[['馬番', '単勝']].copy()
        score = model.predict_proba(X.drop(['単勝'], axis=1))[:, 1]
        score_table['score'] = score
        return score_table    

class StdScorePolicy(AbstractScorePolicy):
    """
    レース内で標準化して、相対評価する。「レース内偏差値」みたいなもの。
    """
    @staticmethod
    def calc(model, X: pd.DataFrame):
        score_table = X[['馬番', '単勝']].copy()
        score = model.predict_proba(X.drop(['単勝'], axis=1))[:, 1]
        score_table['score'] = score
        standard_scaler = lambda x: (x - x.mean()) / x.std(ddof=0)
        score_table['score'] = score_table['score'].groupby(level=0).transform(standard_scaler)
        return score_table
    
class MinMaxScorePolicy(AbstractScorePolicy):
    """
    レース内で標準化して、相対評価すした後、全体を0~1にスケーリング。
    """
    @staticmethod
    def calc(model, X: pd.DataFrame):
        score_table = X[['馬番', '単勝']].copy()
        score = model.predict_proba(X.drop(['単勝'], axis=1))[:, 1]
        standard_scaler = lambda x: (x - x.mean()) / x.std(ddof=0)
        score = score.groupby(level=0).transform(standard_scaler)
        score = (score - score.min()) / (score.max() - score.min())
        score_table['score'] = score
        return score_table