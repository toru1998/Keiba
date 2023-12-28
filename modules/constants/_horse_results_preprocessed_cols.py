import dataclasses


@dataclasses.dataclass(frozen=True)
class HorseResultsPreprocessedCols:
    PLACE: str = '開催'
    WEATHER: str = '天気'
    R: str = 'R'
    RACE_NAME: str = 'レース名'
    N_HORSES: str = '頭数'
    WAKUBAN: str = '枠番'
    UMABAN: str = '馬番'
    ODDS: str = 'オッズ'
    POPULARITY: str = '人気'
    RANK: str = '着順'
    JOCKEY: str = '騎手'
    KINRYO: str = '斤量'
    GROUND_STATE: str = '馬場'
    TIME: str = 'タイム'
    RANK_DIFF: str = '着差'
    CORNER: str = '通過'
    PACE: str = 'ペース'
    NOBORI: str = '上り'
    WEIGHT_AND_DIF: str = '馬体重'
    PRIZE: str = '賞金'
    DATE: str = 'date'
    FIRST_CORNER: str = 'first_corner'
    FINAL_CORNER: str = 'final_corner'
    FINAL_TO_RANK: str = 'final_to_rank'
    FIRST_TO_RANK: str = 'first_to_rank'
    FIRST_TO_FINAL: str = 'first_to_final'
    RACE_TYPE: str = 'race_type'
    COURSE_LEN: str = 'course_len'