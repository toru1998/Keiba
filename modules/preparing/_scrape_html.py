import time
import os
from tqdm.notebook import tqdm
from urllib.request import urlopen

from modules.constants import UrlPaths, LocalDirs

def scrape_html_race(race_id_list: list, skip: bool = True):
    """
    netkeiba.comのraceページのhtmlをスクレイピングしてdata/html/raceに保存する関数。
    skip=Trueにすると、すでにhtmlが存在する場合はスキップされ、Falseにすると上書きされる。
    """
    html_path_list = []
    for race_id in tqdm(race_id_list):
        url = UrlPaths.RACE_URL + race_id #race_idからurlを作る
        time.sleep(1) #相手サーバーに負担をかけないように1秒待機する
        html = urlopen(url).read() #スクレイピング実行
        filename = os.path.join(LocalDirs.HTML_RACE_DIR, race_id+'.bin')
        html_path_list.append(filename)
        if skip and os.path.isfile(filename): #skipがTrueで、かつbinファイルがすでに存在する場合は飛ばす
            print('race_id {} skipped'.format(race_id))
            continue
        with open(filename, 'wb') as f: #保存するファイルパスを指定
            f.write(html) #保存
    return html_path_list

def scrape_html_horse(horse_id_list: list, skip: bool = True):
    """
    netkeiba.comのhorseページのhtmlをスクレイピングしてdata/html/horseに保存する関数。
    skip=Trueにすると、すでにhtmlが存在する場合はスキップされ、Falseにすると上書きされる。
    """
    html_path_list = []
    for horse_id in tqdm(horse_id_list):
        url = UrlPaths.HORSE_URL + horse_id #horse_idからurlを作る
        time.sleep(1) #相手サーバーに負担をかけないように1秒待機する
        html = urlopen(url).read() #スクレイピング実行
        filename = os.path.join(LocalDirs.HTML_HORSE_DIR, horse_id+'.bin')
        html_path_list.append(filename)
        if skip and os.path.isfile(filename): #skipがTrueで、かつbinファイルがすでに存在する場合は飛ばす
            print('horse_id {} skipped'.format(horse_id))
            continue
        with open(filename, 'wb') as f: #保存するファイルパスを指定
            f.write(html) #保存
    return html_path_list

def scrape_html_ped(horse_id_list: list, skip: bool = True):
    """
    netkeiba.comのhorse/pedページのhtmlをスクレイピングしてdata/html/pedに保存する関数。
    skip=Trueにすると、すでにhtmlが存在する場合はスキップされ、Falseにすると上書きされる。
    """
    html_path_list = []
    for horse_id in tqdm(horse_id_list):
        url = UrlPaths.PED_URL + horse_id #horse_idからurlを作る
        time.sleep(1) #相手サーバーに負担をかけないように1秒待機する
        html = urlopen(url).read() #スクレイピング実行
        filename = os.path.join(LocalDirs.HTML_PED_DIR, horse_id+'.bin')
        html_path_list.append(filename)
        if skip and os.path.isfile(filename): #skipがTrueで、かつbinファイルがすでに存在する場合は飛ばす
            print('horse_id {} skipped'.format(horse_id))
            continue
        with open(filename, 'wb') as f: #保存するファイルパスを指定
            f.write(html) #保存
    return html_path_list
