import dataclasses
    
    
@dataclasses.dataclass(frozen=True)
class MergedDataCols:
    WEATHER: str = 'weather'
    RACE_TYPE: str = 'race_type'
    GROUND_STATE: str = 'ground_state'
    SEX: str = '性'
    HORSE_ID: str = 'horse_id'
    JOCKEY_ID: str = 'jockey_id'
    KAISAI: str = '開催'