import dataclasses

@dataclasses.dataclass(frozen=True)
class ResultsRawCols:
    RANK: str = '着順'
    WAKUBAN: str = '枠番'
    UMABAN: str = '馬番'
    HORSE_NAME: str = '馬名'
    SEX_AGE: str = '性齢'
    KINRYO: str = '斤量'
    JOCKEY: str = '騎手'
    TIME: str = 'タイム'
    RANK_DIFF: str = 7
    ODDS: str = '単勝'
    POPULARITY: str = '人気'
    WEIGHT: str = '馬体重'
    TRAINER: str = '調教師'
    COURSE_LEN: str = 'course_len'
    WEATHER: str = 'weather'
    RACE_TYPE: str = 'race_type'
    GROUND_STATE: str = 'ground_state'
    DATE: str = 'date'
    HORSE_ID: str = 'horse_id'
    JOCKEY_ID: str = 'jockey_id'