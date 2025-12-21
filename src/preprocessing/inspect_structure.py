
import gzip
from bs4 import BeautifulSoup
import re
import pandas as pd
import os

def load_sample_html(filepath):
    """
    指定されたgzip圧縮HTMLファイルを読み込み、EUC-JPとしてデコードを試みます。
    構造確認用の簡易ローダーであるため、エラーは無視して読み込みます。
    """
    with gzip.open(filepath, 'rb') as f:
        content = f.read().decode('euc-jp', errors='ignore')
    return content

def inspect_html(filepath):
    """
    HTMLファイルの構造（DOM構造）を解析・表示し、スクレイピングロジックの検討を支援します。
    
    確認内容:
    - レース基本情報（タイトル、詳細テキストなど）のDOM構造
    - レース結果テーブルのヘッダーやデータ行の並び
    - 払い戻しテーブルの有無や構造
    """
    content = load_sample_html(filepath)
    soup = BeautifulSoup(content, 'html.parser')
    
    # --- レース基本情報の確認 ---
    # data_intro クラス内にレース名や条件が含まれていると想定
    data_intro = soup.find('div', class_='data_intro')
    if data_intro:
        print("--- Race Info (レース情報) ---")
        h1 = data_intro.find('h1')
        print(f"Title: {h1.text.strip() if h1 else 'Not Found'}")
        
        # レース詳細 (R, コース, 天候, 馬場など)
        r_info = data_intro.find('dt')
        print(f"Race R: {r_info.text.strip() if r_info else 'Not Found'}")
        
        # 詳細テキスト (diary_snap_cut 内にあることが多い)
        diary_snap = data_intro.find('p', class_='diary_snap_cut') # divではなくpタグの場合が多い
        if diary_snap:
            print(f"Details: {diary_snap.text.strip()}")
        else:
            # 構造が違う場合のフォールバック探索: テキスト全体を表示
            lines = data_intro.text.split('\n')
            clean_lines = [l.strip() for l in lines if l.strip()]
            print(f"Raw Intro Text: {clean_lines[:5]}")
            
    # --- レース結果テーブルの確認 ---
    # table class="race_table_01"
    table = soup.find('table', class_='race_table_01')
    if table:
        print("\n--- Result Table Headers (結果テーブルヘッダー) ---")
        headers = [th.text.strip() for th in table.find('tr').find_all('th')]
        print(headers)
        
        print("\n--- First Row (データ1行目) ---")
        rows = table.find_all('tr')
        if len(rows) > 1:
            first_row = rows[1] # 0はheader
            cols = [td.text.strip().replace('\n', ' ') for td in first_row.find_all('td')]
            print(cols)

    # --- 払い戻しテーブルの確認 ---
    # table class="pay_block" などの確認用
    pay_table = soup.find('table', class_='pay_table_01') # netkeibaの払い戻しテーブルクラス
    if pay_table:
        print("\n--- Pay Table Found (払い戻しテーブル) ---")
        # 簡易ダンプ
        rows = pay_table.find_all('tr')
        for r in rows[:3]:
            print([c.text.strip() for c in r.find_all(['th', 'td'])])

if __name__ == "__main__":
    # 動作確認用のターゲットファイル
    target_file = '/home/toru/code/Keiba/data/html/race/202106030211.html.gz'
    
    if os.path.exists(target_file):
        inspect_html(target_file)
    else:
        print(f"File not found: {target_file}")
        print("任意の .html.gz ファイルパスを指定してください。")
