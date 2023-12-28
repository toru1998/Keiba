import dataclasses
from types import MappingProxyType

@dataclasses.dataclass(frozen=True)
class Master:
    PLACE_DICT: dict = MappingProxyType({
        '札幌':'01',
        '函館':'02',
        '福島':'03',
        '新潟':'04',
        '東京':'05',
        '中山':'06',
        '中京':'07',
        '京都':'08',
        '阪神':'09',
        '小倉':'10',
        })

    RACE_TYPE_DICT: dict = MappingProxyType({
        '芝': '芝',
        'ダ': 'ダート',
        '障': '障害',
        })
    
    WEATHER_LIST: tuple = ('晴', '曇', '小雨', '雨', '小雪', '雪')
    
    GROUND_STATE_LIST: tuple = ('良', '稍重', '重', '不良')
    
    SEX_LIST: tuple = ('牡', '牝', 'セ')
    
    