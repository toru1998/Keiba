
import os
import gzip
import re
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging

# ロギング設定: 処理状況やエラーをログに出力します
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetkeibaHtmlParser:
    """
    保存されたnetkeibaのHTMLファイルを解析し、構造化データ（DataFrameなど）に変換するクラス
    
    主な機能:
    1. GZIP圧縮されたHTMLファイルの読み込みと文字コード自動判定
    2. BeautifulSoupを用いたHTML解析
    3. レース基本情報（場所、距離、天候など）の抽出
    4. レース結果テーブル（着順、馬名、タイムなど）の抽出
    """
    def __init__(self):
        pass

    def _load_html(self, filepath):
        """
        gzip圧縮されたHTMLファイルを読み込み、適切な文字コードでデコードします。
        netkeibaは古いページなどでエンコーディングが混在する場合があるため、
        UTF-8 -> EUC-JP -> CP932 の順でデコードを試行します。
        """
        try:
            with gzip.open(filepath, 'rb') as f:
                raw_content = f.read()
                
            # まず UTF-8 で試す
            try:
                content = raw_content.decode('utf-8')
                # 特定ファイルでのデバッグ用ログ
                if '202206020703' in filepath:
                    logger.info(f"Decoded {filepath} with UTF-8 successfully.")
                return content
            except UnicodeDecodeError:
                pass
                
            # 次に EUC-JP (netkeibaの伝統的なエンコーディング) で試す
            try:
                content = raw_content.decode('euc-jp')
                if '202206020703' in filepath:
                    logger.info(f"Decoded {filepath} with EUC-JP.")
                return content
            except UnicodeDecodeError:
                pass
                
            # 最後に CP932 (Windows-31J) で試す
            if '202206020703' in filepath:
                logger.info(f"Decoded {filepath} with CP932 (fallback).")
            return raw_content.decode('cp932', errors='replace')
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None

    def parse_file(self, filepath):
        """
        1つのHTMLファイルをパースして、レース情報と結果データの辞書を返します。
        
        戻り値の形式:
        {
            'race_info': dict,  # レース名、距離、天候などのメタデータ
            'results': DataFrame # 着順などのテーブルデータ
        }
        """
        content = self._load_html(filepath)
        if not content:
            return None

        soup = BeautifulSoup(content, 'lxml')
        
        # ファイル名からレースIDを取得 (例: 202106030211.html.gz -> 202106030211)
        race_id = os.path.basename(filepath).split('.')[0]
        
        try:
            race_info = self._parse_race_info(soup)
            race_info['race_id'] = race_id
            
            results = self._parse_race_results(soup)
            if results is not None:
                results['race_id'] = race_id
                
            return {
                'race_info': race_info,
                'results': results
            }
        except Exception as e:
            logger.warning(f"Parse error in {filepath}: {e}")
            return None

    def _parse_race_info(self, soup):
        """
        HTML内のレース基本情報セクションからデータを抽出します。
        対象: <div class="data_intro"> 内のタイトルや詳細テキスト
        """
        info = {}
        
        # タイトル要素の取得
        data_intro = soup.find('div', class_='data_intro')
        if not data_intro:
            return info
            
        h1 = data_intro.find('h1')
        if h1:
            info['race_name'] = h1.text.strip()
            
        # レース詳細行 (R数, 天候, コース種別, 距離, 発走時刻など)
        # 構造例: 
        # <div class="data_intro">
        #   <dt>11 R</dt>
        #   <p class="diary_snap_cut">
        #     <span>ダ右1800m / 天候 : 晴 / ダート : 良 / 発走 : 15:35</span>
        #   </p>
        # </div>
        
        # レース番号 (R)
        dt = data_intro.find('dt')
        if dt:
            info['race_round'] = dt.text.strip()

        # 詳細テキストの取得
        # diary_snap_cut クラス内に情報がある場合が多いですが、直書きの場合もあります
        details_text = ""
        diary_snap = data_intro.find('p', class_='diary_snap_cut') 
        if diary_snap:
            details_text = diary_snap.text
        else:
            # テキスト全体から推測
            details_text = data_intro.text
            
        # 距離 (m) の抽出: "1800m" のようなパターンを正規表現で探す
        match_dist = re.search(r'(\d+)m', details_text)
        if match_dist:
            info['distance'] = int(match_dist.group(1))
            
        # コース種別 (芝/ダート/障害) の判定
        # 文字化けのリスクも考慮しつつ、キーワードが含まれるかで判定
        if '芝' in details_text:
            info['surface'] = 'turf'
        elif 'ダ' in details_text:
            info['surface'] = 'dirt'
        elif '障' in details_text:
            info['surface'] = 'obstacle'
        else:
            info['surface'] = 'unknown'

        # 天候 (晴, 曇, 雨, 小雨, 雪, 小雪) の判定
        if '晴' in details_text:
            info['weather'] = 'sunny'
        elif '曇' in details_text:
            info['weather'] = 'cloudy'
        elif '雨' in details_text:
            info['weather'] = 'rainy'
        elif '雪' in details_text:
            info['weather'] = 'snowy'
            
        # 馬場状態 (良, 稍重, 重, 不良) の判定
        # 判定順序に注意（"稍重"に"重"が含まれるため、先に"稍重"などをチェックするか、elifで繋ぐ）
        if '不良' in details_text:
            info['track_condition'] = 'bad'
        elif '稍重' in details_text:
            info['track_condition'] = 'slightly_heavy'
        elif '重' in details_text: 
            info['track_condition'] = 'heavy'
        elif '良' in details_text:
            info['track_condition'] = 'good'
            
        return info

    def _parse_race_results(self, soup):
        """
        レース結果テーブル (class="race_table_01") からデータを抽出し、DataFrameとして返します。
        """
        table = soup.find('table', class_='race_table_01')
        if not table:
            return None
            
        # ヘッダー行の取得（参考用、実際は固定インデックスで取得することが多い）
        headers = [th.text.strip() for th in table.find('tr').find_all('th')]
        
        # データ行の取得
        data = []
        rows = table.find_all('tr')[1:] # 1行目はヘッダーなのでスキップ
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 3: # データ不正防止
                continue
                
            # 各カラムの抽出
            # netkeibaのテーブル構造は概ね固定されていますが、変更の可能性には留意が必要です
            # 典型的な並び: 着順, 枠番, 馬番, 馬名, 性齢, 斤量, 騎手, タイム, 着差...
            
            # 着順 (数値以外が入ることもある: 取消, 除外など)
            rank_text = cols[0].text.strip()
            
            # 枠番
            frame_number = cols[1].text.strip()
            
            # 馬番
            horse_number = cols[2].text.strip()
            
            # 馬名
            horse_name = cols[3].text.strip()
            
            # 性齢 (例: 牡3)
            sex_age = cols[4].text.strip()
            
            # 斤量
            weight = cols[5].text.strip()
            
            # 騎手
            jockey = cols[6].text.strip()
            
            # タイム
            time = cols[7].text.strip()
            
            # 単勝オッズと人気
            # カラム位置が変わる可能性があるため、lenチェックを入れています
            # 通常位置: 11:単勝, 12:人気
            odds = cols[11].text.strip() if len(cols) > 11 else ''
            popularity = cols[12].text.strip() if len(cols) > 12 else ''
            
            data.append({
                'rank': rank_text,
                'frame_number': frame_number,
                'horse_number': horse_number,
                'horse_name': horse_name,
                'sex_age': sex_age,
                'jockey_weight': weight,
                'jockey': jockey,
                'time': time,
                'odds': odds,
                'popularity': popularity
            })
            
        return pd.DataFrame(data)

def process_batch(data_dir, output_dir, limit=None):
    """
    指定ディレクトリ内のHTMLファイル（.html.gz）を一括処理して、Parquet形式で保存します。
    """
    parser = NetkeibaHtmlParser()
    
    # HTMLファイルリストの取得
    files = []
    for root, _, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith('.html.gz'):
                files.append(os.path.join(root, f))
                
    if not files:
        print("No html files found.")
        return

    if limit:
        files = files[:limit]

    all_race_infos = []
    all_results = []
    
    print(f"Processing {len(files)} files...")
    # tqdmで進捗バーを表示しながら処理
    for f in tqdm(files):
        parsed = parser.parse_file(f)
        if parsed:
            if parsed['race_info']:
                all_race_infos.append(parsed['race_info'])
            if parsed['results'] is not None and not parsed['results'].empty:
                all_results.append(parsed['results'])
                
    # 抽出データの結合と保存
    os.makedirs(output_dir, exist_ok=True)
    
    if all_race_infos:
        df_race = pd.DataFrame(all_race_infos)
        df_race.to_parquet(os.path.join(output_dir, 'race_info.parquet'), index=False)
        print(f"Saved race_info.parquet: {len(df_race)} records")
        
    if all_results:
        df_results = pd.concat(all_results, ignore_index=True)
        df_results.to_parquet(os.path.join(output_dir, 'race_results.parquet'), index=False)
        print(f"Saved race_results.parquet: {len(df_results)} records")

if __name__ == "__main__":
    # 入力元と出力先の設定
    DATA_DIR = 'data/html/race'
    OUTPUT_DIR = 'data/processed'
    
    # バッチ処理実行
    process_batch(DATA_DIR, OUTPUT_DIR)
