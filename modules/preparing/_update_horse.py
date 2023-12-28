import datetime
import os
import re
from urllib.request import urlopen
from bs4 import BeautifulSoup
from tqdm.notebook import tqdm

from modules.constants import UrlPaths, LocalDirs
from modules import preparing

class UpdateHorse:
    """
    出走するhorse_idは事前に分かるので、馬の過去成績テーブルと血統テーブルをアップデートしておく。
    """
    def __init__(self) -> None:
        today = datetime.date.today().strftime('%Y%m%d')
        self.horse_results_path = os.path.join(
            LocalDirs.RAW_HORSE_RESULTS_DIR, 'horse_results_'+today+'.pickle'
            )
        self.peds_path = os.path.join(
            LocalDirs.RAW_PEDS_DIR, 'peds_'+today+'.pickle'
            )
    
    def scrape_horse_id_list(self, race_id_list: list) -> list:
        """
        当日出走するhorse_id一覧を取得
        """
        print('sraping horse_id_list')
        horse_id_list = []
        for race_id in tqdm(race_id_list):
            query = '?race_id=' + race_id
            url = UrlPaths.SHUTUBA_TABLE + query
            html = urlopen(url)
            soup = BeautifulSoup(html, 'lxml', from_encoding='utf-8')
            horse_td_list = soup.find_all("td", attrs={'class': 'HorseInfo'})
            for td in horse_td_list:
                horse_id = re.findall(r'\d+', td.find('a')['href'])[0]
                horse_id_list.append(horse_id)
        return horse_id_list
    
    def update_horse_results(self, horse_id_list: list):
        """
        netkeiba.com/horse/をスクレイピングし、data/raw/horse_results_{today}.pickleとして保存
        """
        print('scraping horse_results')
        html_files = preparing.scrape_html_horse(horse_id_list, skip=False)
        horse_results = preparing.get_rawdata_horse_results(html_files)
        horse_results.to_pickle(self.horse_results_path)
        
    def update_peds(self, horse_id_list: list):
        """
        netkeiba.com/horse/ped/をスクレイピングし、data/raw/peds_{today}.pickleとして保存
        """
        print('scraping peds')
        html_files = preparing.scrape_html_ped(horse_id_list, skip=True)
        peds = preparing.get_rawdata_peds(html_files)
        peds.to_pickle(self.peds_path)
    