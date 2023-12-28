import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By

from modules.constants import UrlPaths

def scrape_shutuba_table(race_id: str, date: str, file_path: str):
    """
    当日の出馬表をスクレイピング。
    dateはyyyy/mm/ddの形式。
    """
    driver = webdriver.Chrome()
    query = '?race_id=' + race_id
    url = UrlPaths.SHUTUBA_TABLE + query
    df = pd.DataFrame()
    try:
        driver.get(url)
        time.sleep(1)
        
        #メインのテーブルの取得
        for tr in driver.find_elements(By.CLASS_NAME, 'HorseList'):
            row = []
            for td in tr.find_elements(By.TAG_NAME, 'td'):
                if td.get_attribute('class') in ['HorseInfo', 'Jockey']:
                    href = td.find_element(By.TAG_NAME, 'a').get_attribute('href')
                    row.append(re.findall(r'\d+', href)[0])
                row.append(td.text)
            df = df.append(pd.Series(row), ignore_index=True)
            
        #レース結果テーブルと列を揃える
        df = df[[0, 1, 5, 6, 11, 12, 10, 3, 7]]
        df.columns = ['枠番', '馬番', '性齢', '斤量', '単勝', '人気', '馬体重', 'horse_id', 'jockey_id']
        df.index = [race_id] * len(df)
        
        #レース情報の取得
        texts = driver.find_element(By.CLASS_NAME, 'RaceList_Item02').text
        texts = re.findall(r'\w+', texts)
        for text in texts:
            if 'm' in text:
                df['course_len'] = [int(re.findall(r'\d+', text)[-1])] * len(df) #20211212：[0]→[-1]に修正
            if text in ["曇", "晴", "雨", "小雨", "小雪", "雪"]:
                df["weather"] = [text] * len(df)
            if text in ["良", "稍重", "重"]:
                df["ground_state"] = [text] * len(df)
            if '不' in text:
                df["ground_state"] = ['不良'] * len(df)
            # 2020/12/13追加
            if '稍' in text:
                df["ground_state"] = ['稍重'] * len(df)
            if '芝' in text:
                df['race_type'] = ['芝'] * len(df)
            if '障' in text:
                df['race_type'] = ['障害'] * len(df)
            if 'ダ' in text:
                df['race_type'] = ['ダート'] * len(df)
        df['date'] = [date] * len(df)
    except Exception as e:
        print(e)
        driver.close()
    df.to_pickle(file_path)

def preprocess(self):
    df = self.raw_data

    # 性齢を性と年齢に分ける
    df["性"] = df["性齢"].map(lambda x: str(x)[0])
    df["年齢"] = df["性齢"].map(lambda x: str(x)[1:]).astype(int)

    # 馬体重を体重と体重変化に分ける
    df["体重"] = df["馬体重"].str.split("(", expand=True)[0]
    df["体重変化"] = df["馬体重"].str.split("(", expand=True)[1].str[:-1]
    
    #errors='coerce'で、"計不"など変換できない時に欠損値にする
    df['体重'] = pd.to_numeric(df['体重'], errors='coerce')
    df['体重変化'] = pd.to_numeric(df['体重変化'], errors='coerce')

    # 単勝をfloatに変換
    df["単勝"] = df["単勝"].astype(float)

    # 不要な列を削除
    df.drop(["性齢", "馬体重"], axis=1, inplace=True)
    
    #6/6出走数追加
    df['n_horses'] = df.index.map(df.index.value_counts())
    
    # 距離は10の位を切り捨てる
    df["course_len"] = df["course_len"].astype(float) // 100
    # 日付型に変更
    df["date"] = pd.to_datetime(df["date"], format="%Y年%m月%d日")
    # 開催場所
    df['開催'] = df.index.map(lambda x:str(x)[4:6])
    
    self.preprocessed_data = df
    return self