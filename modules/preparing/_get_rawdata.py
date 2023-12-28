import pandas as pd
from tqdm.notebook import tqdm
from bs4 import BeautifulSoup
import re

def get_rawdata_results(html_path_list: list):
    """
    raceページのhtmlを受け取って、レース結果テーブルに変換する関数。
    """
    print('preparing raw results table')
    race_results = {}
    for html_path in tqdm(html_path_list):
        with open(html_path, 'rb') as f:
            try:
                html = f.read() #保存してあるbinファイルを読み込む
                df = pd.read_html(html)[0] #メインとなるレース結果テーブルデータを取得
                
                soup = BeautifulSoup(html, "html.parser") #htmlをsoupオブジェクトに変換

                #馬ID、騎手IDをスクレイピング
                horse_id_list = []
                horse_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/horse")}
                )
                for a in horse_a_list:
                    horse_id = re.findall(r"\d+", a["href"])
                    horse_id_list.append(horse_id[0])
                jockey_id_list = []
                jockey_a_list = soup.find("table", attrs={"summary": "レース結果"}).find_all(
                    "a", attrs={"href": re.compile("^/jockey")}
                )
                for a in jockey_a_list:
                    jockey_id = re.findall(r"\d+", a["href"])
                    jockey_id_list.append(jockey_id[0])
                df["horse_id"] = horse_id_list
                df["jockey_id"] = jockey_id_list

                #インデックスをrace_idにする
                race_id = re.findall('(?<=race/)\d+', html_path)[0]
                df.index = [race_id] * len(df)

                race_results[race_id] = df
            except Exception as e:
                print('error at {}'.format(html_path))
                print(e)
    #pd.DataFrame型にして一つのデータにまとめる
    race_results_df = pd.concat([race_results[key] for key in race_results])

    return race_results_df

def get_rawdata_info(html_path_list: list):
    """
    raceページのhtmlを受け取って、レース情報テーブルに変換する関数。
    """
    print('preparing raw race_info table')
    race_infos = {}
    for html_path in tqdm(html_path_list):
        with open(html_path, 'rb') as f:
            try:
                html = f.read() #保存してあるbinファイルを読み込む
                
                soup = BeautifulSoup(html, "html.parser") #htmlをsoupオブジェクトに変換

                #天候、レースの種類、コースの長さ、馬場の状態、日付をスクレイピング
                texts = (
                    soup.find("div", attrs={"class": "data_intro"}).find_all("p")[0].text
                    + soup.find("div", attrs={"class": "data_intro"}).find_all("p")[1].text
                )
                info = re.findall(r'\w+', texts)
                df = pd.DataFrame()
                for text in info:
                    if text in ["芝", "ダート"]:
                        df["race_type"] = [text]
                    if "障" in text:
                        df["race_type"] = ["障害"]
                    if "m" in text:
                        df["course_len"] = [int(re.findall(r"\d+", text)[-1])] #20211212：[0]→[-1]に修正
                    if text in ["良", "稍重", "重", "不良"]:
                        df["ground_state"] = [text]
                    if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                        df["weather"] = [text]
                    if "年" in text:
                        df["date"] = [text]
                
                #インデックスをrace_idにする
                race_id = re.findall('(?<=race/)\d+', html_path)[0]
                df.index = [race_id] * len(df)

                race_infos[race_id] = df
            except Exception as e:
                print('error at {}'.format(html_path))
                print(e)
    #pd.DataFrame型にして一つのデータにまとめる
    race_infos_df = pd.concat([race_infos[key] for key in race_infos])

    return race_infos_df

def get_rawdata_return(html_path_list: list):
    """
    raceページのhtmlを受け取って、払い戻しテーブルに変換する関数。
    """
    print('preparing raw return table')
    horse_results = {}
    for html_path in tqdm(html_path_list):
        with open(html_path, 'rb') as f:
            try: 
                html = f.read() #保存してあるbinファイルを読み込む
                
                html = html.replace(b'<br />', b'br')
                dfs = pd.read_html(html)

                #dfsの1番目に単勝〜馬連、2番目にワイド〜三連単がある
                df = pd.concat([dfs[1], dfs[2]])
                
                race_id = re.findall('(?<=race/)\d+', html_path)[0]
                df.index = [race_id] * len(df)
                horse_results[race_id] = df
            except Exception as e:
                print('error at {}'.format(html_path))
                print(e)
    #pd.DataFrame型にして一つのデータにまとめる
    horse_results_df = pd.concat([horse_results[key] for key in horse_results])
    return horse_results_df

def get_rawdata_horse_results(html_path_list: list):
    """
    horseページのhtmlを受け取って、馬の過去成績のDataFrameに変換する関数。
    """
    print('preparing raw horse_results table')
    horse_results = {}
    for html_path in tqdm(html_path_list):
        with open(html_path, 'rb') as f:
            html = f.read() #保存してあるbinファイルを読み込む
            
            df = pd.read_html(html)[3]
            #受賞歴がある馬の場合、3番目に受賞歴テーブルが来るため、4番目のデータを取得する
            if df.columns[0]=='受賞歴':
                df = pd.read_html(html)[4]
                
            horse_id = re.findall('(?<=horse/)\d+', html_path)[0]
            
            df.index = [horse_id] * len(df)
            horse_results[horse_id] = df
            
    #pd.DataFrame型にして一つのデータにまとめる
    horse_results_df = pd.concat([horse_results[key] for key in horse_results])
    return horse_results_df

def get_rawdata_peds(html_path_list: list):
    """
    horse/pedページのhtmlを受け取って、血統のDataFrameに変換する関数。
    """
    print('preparing raw peds table')
    peds = {}
    for html_path in tqdm(html_path_list):
        with open(html_path, 'rb') as f:
            html = f.read() #保存してあるbinファイルを読み込む
            
            df = pd.read_html(html)[0]

            #重複を削除して1列のSeries型データに直す
            generations = {}
            horse_id = re.findall('(?<=ped/)\d+', html_path)[0]
            for i in reversed(range(5)):
                generations[i] = df[i]
                df.drop([i], axis=1, inplace=True)
                df = df.drop_duplicates()
            ped = pd.concat([generations[i] for i in range(5)]).rename(horse_id)
            peds[horse_id] = ped.reset_index(drop=True)
    #pd.DataFrame型にして一つのデータにまとめる
    peds_df = pd.concat([peds[key] for key in peds], axis=1).T.add_prefix('peds_')
    return peds_df