import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    """
    アプリケーション全体の設定値を管理するクラス
    """
    # スクレイピング時の待機時間（秒）
    SCRAPING_INTERVAL: float = 2.5
    
    # ユーザーエージェント
    USER_AGENT: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
