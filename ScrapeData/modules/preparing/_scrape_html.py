# -*- coding: utf-8 -*-

import datetime
import re
import pandas as pd
import time
import os
from tqdm.auto import tqdm
from urllib.request import urlopen, Request
from contextlib import closing
import sqlite3

from modules.constants import UrlPaths, LocalPaths, Config
from modules.utils.logging_utils import get_logger

logger = get_logger(__name__)

def init_db():
    """データベースの初期化"""
    with closing(sqlite3.connect(LocalPaths.DB_PATH)) as conn:
        cursor = conn.cursor()
        
        # race_htmlテーブルの作成
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS race_html (
                race_id TEXT PRIMARY KEY,
                html TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # horse_htmlテーブルの作成
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS horse_html (
                horse_id TEXT PRIMARY KEY,
                html TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # ped_htmlテーブルの作成
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ped_html (
                horse_id TEXT PRIMARY KEY,
                html TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()

def get_html(url: str) -> str:
    """HTMLを取得する関数"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    request = Request(url, headers=headers)
    try:
        with urlopen(request) as response:
            content = response.read()
            # UTF-8でデコード
            html = content.decode('euc-jp')
            return html
    except UnicodeDecodeError as e:
        logger.error(f"Error decoding HTML: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None

def scrape_html_race(race_id_list: list, skip: bool = True):
    """
    netkeiba.comのraceページのhtmlをスクレイピングしてDBに保存する関数。
    skip=Trueにすると、すでにhtmlが存在する場合はスキップされ、Falseにすると上書きされる。
    返り値：新しくスクレイピングしたhtmlのレコード数
    """
    updated_count = 0
    
    # データベースの初期化
    init_db()
    
    with closing(sqlite3.connect(LocalPaths.DB_PATH)) as conn:
        cursor = conn.cursor()
        
        for race_id in tqdm(race_id_list):
            # skipがTrueで、かつ既存のデータが存在する場合は飛ばす
            if skip:
                cursor.execute('SELECT 1 FROM race_html WHERE race_id = ?', (race_id,))
                if cursor.fetchone():
                    logger.info(f'race_id {race_id} skipped')
                    continue

            # race_idからurlを作る
            url = UrlPaths.RACE_URL + race_id
            # 相手サーバーに負担をかけないように待機する
            time.sleep(Config.SCRAPING_INTERVAL)
            # スクレイピング実行
            html = get_html(url)
            if not html:
                continue

            # DBに保存
            cursor.execute('''
                INSERT OR REPLACE INTO race_html (race_id, html)
                VALUES (?, ?)
            ''', (race_id, html))
            
            updated_count += 1
            
        conn.commit()

    return updated_count

def scrape_html_horse(horse_id_list: list, skip: bool = True):
    """
    netkeiba.comのhorseページのhtmlをスクレイピングしてDBに保存する関数。
    skip=Trueにすると、すでにhtmlが存在する場合はスキップされ、Falseにすると上書きされる。
    返り値：新しくスクレイピングしたhtmlのレコード数
    """
    updated_count = 0
    
    # データベースの初期化
    init_db()
    
    with closing(sqlite3.connect(LocalPaths.DB_PATH)) as conn:
        cursor = conn.cursor()
        
        for horse_id in tqdm(horse_id_list):
            if skip:
                cursor.execute('SELECT 1 FROM horse_html WHERE horse_id = ?', (horse_id,))
                if cursor.fetchone():
                    logger.info(f'horse_id {horse_id} skipped')
                    continue

            url = UrlPaths.HORSE_URL + horse_id
            # 相手サーバーに負担をかけないように待機する
            time.sleep(Config.SCRAPING_INTERVAL)
            html = get_html(url)
            if not html:
                continue

            cursor.execute('''
                INSERT OR REPLACE INTO horse_html (horse_id, html)
                VALUES (?, ?)
            ''', (horse_id, html))
            
            updated_count += 1
            
        conn.commit()

    return updated_count

def scrape_html_ped(horse_id_list: list, skip: bool = True):
    """
    netkeiba.comのhorse/pedページのhtmlをスクレイピングしてDBに保存する関数。
    skip=Trueにすると、すでにhtmlが存在する場合はスキップされ、Falseにすると上書きされる。
    返り値：新しくスクレイピングしたhtmlのレコード数
    """
    updated_count = 0
    
    # データベースの初期化
    init_db()
    
    with closing(sqlite3.connect(LocalPaths.DB_PATH)) as conn:
        cursor = conn.cursor()
        
        for horse_id in tqdm(horse_id_list):
            if skip:
                cursor.execute('SELECT 1 FROM ped_html WHERE horse_id = ?', (horse_id,))
                if cursor.fetchone():
                    logger.info(f'horse_id {horse_id} skipped')
                    continue

            url = UrlPaths.PED_URL + horse_id
            # 相手サーバーに負担をかけないように待機する
            time.sleep(Config.SCRAPING_INTERVAL)
            html = get_html(url)
            if not html:
                continue

            cursor.execute('''
                INSERT OR REPLACE INTO ped_html (horse_id, html)
                VALUES (?, ?)
            ''', (horse_id, html))
            
            updated_count += 1
            
        conn.commit()

    return updated_count