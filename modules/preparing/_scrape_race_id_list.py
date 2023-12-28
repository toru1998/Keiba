import pandas as pd
import time
import re
from tqdm.notebook import tqdm
from bs4 import BeautifulSoup
from urllib.request import urlopen
from selenium import webdriver
from selenium.webdriver.common.by import By

from modules.constants import UrlPaths


def scrape_kaisai_date(from_: str, to_: str):
    """
    yyyy-mmの形式でfrom_とto_を指定すると、間のレース開催日一覧が返ってくる関数。
    to_の月は含まないので注意。
    """
    print('getting race date from {} to {}'.format(from_, to_))
    date_range = pd.date_range(start=from_, end=to_, freq="M") #間の年月一覧を作成
    kaisai_date_list = [] #開催日一覧を入れるリスト
    for year, month in tqdm(zip(date_range.year, date_range.month), total=len(date_range)):
        #取得したdate_rangeから、スクレイピング対象urlを作成する。
        #urlは例えば、https://race.netkeiba.com/top/calendar.html?year=2022&month=7 のような構造になっている。
        query = [
            'year=' + str(year),
            'month=' + str(month),
        ]
        url = UrlPaths.CALENDAR_URL + '?' + '&'.join(query)
        html = urlopen(url).read()
        time.sleep(1)
        soup = BeautifulSoup(html, "html.parser")
        a_list = soup.find('table', class_='Calendar_Table').find_all('a')
        for a in a_list:
            kaisai_date_list.append(re.findall('(?<=kaisai_date=)\d+', a['href'])[0])
    return kaisai_date_list
    
def scrape_race_id_list(kaisai_date_list: list, from_shutuba=False, waiting_time=10):
    """
    開催日をyyyymmddの文字列形式でリストで入れると、レースid一覧が返ってくる関数。
    レース前日準備のためrace_idを取得する際には、from_shutuba=Trueにする。
    ChromeDriverは要素を取得し終わらないうちに先に進んでしまうことがあるので、その場合の待機時間をwaiting_timeで指定。
    """
    race_id_list = []
    driver = webdriver.Chrome()
    print('getting race_id_list')
    for kaisai_date in tqdm(kaisai_date_list):
        try:
            query = [
                'kaisai_date=' + str(kaisai_date)
            ]
            url = UrlPaths.RACE_LIST_URL + '?' + '&'.join(query)
            print('scraping: {}'.format(url))
            driver.get(url)
            try:
                time.sleep(1) #取得し終わらないうちに先に進んでしまうのを防ぐ
                a_list = driver.find_element(By.CLASS_NAME, 'RaceList_Box').find_elements(By.TAG_NAME, 'a')
            except: #それでも取得できなかったらもう10秒待つ
                print('waiting more {} seconds'.format(waiting_time))
                time.sleep(waiting_time)
                a_list = driver.find_element(By.CLASS_NAME, 'RaceList_Box').find_elements(By.TAG_NAME, 'a')
            for a in a_list:
                if from_shutuba:
                    race_id = re.findall('(?<=shutuba.html\?race_id=)\d+', a.get_attribute('href'))
                else:
                    race_id = re.findall('(?<=result.html\?race_id=)\d+', a.get_attribute('href'))
                if len(race_id) > 0:
                    race_id_list.append(race_id[0])
        except Exception as e:
            print(e)
            break
    driver.close()
    return race_id_list