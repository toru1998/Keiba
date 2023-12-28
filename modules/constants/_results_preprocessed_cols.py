import dataclasses


@dataclasses.dataclass(frozen=True)
class ResultsPreprocessedCols:
    WAKUBAN: str = '枠番'
    UMABAN: str = '馬番'
    KNIRYO: str = '斤量'
    ODDS: str = '単勝'
    COURSE_LEN: str = 'course_len'
    WEATHER: str = 'weather'
    RACE_TYPE: str = 'race_type'
    GROUND_STATE: str = 'ground_state'
    DATE: str = 'date'
    HORSE_ID: str = 'horse_id'
    JOCKEY_ID: str = 'jockey_id'
    RANK: str = 'rank'
    SEX: str = '性'
    AGE: str = '年齢'
    WEIGHT: str = '体重'
    WEIGHT_DIFF: str = '体重変化'
    PLACE: str = '開催'
    N_HORSES: str = 'n_horses'