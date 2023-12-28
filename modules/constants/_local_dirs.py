import os
import dataclasses

@dataclasses.dataclass(frozen=True)
class LocalDirs:
    # パス
    ## プロジェクトルートの絶対パス
    BASE_DIR: str = os.path.abspath('./')
    ## dataディレクトリまでの絶対パス
    DATA_DIR: str = os.path.join(os.path.abspath('./'),'data')
    ### HTMLディレクトリのパス
    HTML_DIR: str = os.path.join(DATA_DIR, 'html')
    HTML_RACE_DIR: str = os.path.join(HTML_DIR, 'race')
    HTML_HORSE_DIR: str = os.path.join(HTML_DIR, 'horse')
    HTML_PED_DIR: str = os.path.join(HTML_DIR, 'ped')
    
    ### rawディレクトリのパス
    RAW_DIR: str = os.path.join(DATA_DIR, 'raw')
    RAW_RESULTS_DIR: str = os.path.join(RAW_DIR, 'results')
    RAW_RACE_INFO_DIR: str = os.path.join(RAW_DIR, 'race_info')
    RAW_RETURN_DIR: str = os.path.join(RAW_DIR, 'return_tables')
    RAW_HORSE_RESULTS_DIR: str = os.path.join(RAW_DIR, 'horse_results')
    RAW_PEDS_DIR: str = os.path.join(RAW_DIR, 'peds')