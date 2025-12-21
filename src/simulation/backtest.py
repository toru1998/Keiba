
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.df = None
        self.model = None
        
    def load_resources(self):
        print("Loading resources...")
        self.df = pd.read_parquet(self.data_path)
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        # 必要な列の型変換 (学習時と同じ処理が必要)
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype('category')

    def predict(self):
        print("Predicting probabilities...")
        # 学習時と同じ特徴量を使用
        # TODO: 学習時のカラムリストを保存しておき、それを使ってフィルタリングするのが安全
        drop_cols = ['rank', 'time', 'time_seconds', 'target', 'race_id', 'date', 'race_name', 'race_round']
        X = self.df.drop(columns=[c for c in drop_cols if c in self.df.columns])
        
        # 確率予測
        self.df['prob'] = self.model.predict(X)
        
    def calculate_expected_value(self):
        print("Calculating Expected Value (EV)...")
        # 期待値 = 確率 * オッズ
        # 単勝オッズを使用
        self.df['expected_value'] = self.df['prob'] * self.df['odds']
        
    def simulate(self, threshold=1.0):
        print(f"Simulating bets with EV threshold > {threshold}...")
        
        # 条件に合う馬券を抽出
        bets = self.df[self.df['expected_value'] > threshold].copy()
        
        # 的中判定 (rank == 1 が単勝的中とする)
        # 複勝等の場合はtarget定義による
        bets['hit'] = (bets['rank'] == 1).astype(int)
        
        # 払い戻し
        # 的中ならオッズ * 100円 (1単位), 外れなら0
        bets['return'] = bets['hit'] * bets['odds'] * 100
        
        # 収支
        total_bet_count = len(bets)
        total_investment = total_bet_count * 100
        total_return = bets['return'].sum()
        profit = total_return - total_investment
        recovery_rate = (total_return / total_investment * 100) if total_investment > 0 else 0
        
        print(f"Total Bets: {total_bet_count}")
        print(f"Total Return: {total_return:.0f} JPY")
        print(f"Total Investment: {total_investment:.0f} JPY")
        print(f"Profit: {profit:.0f} JPY")
        print(f"Recovery Rate: {recovery_rate:.2f}%")
        
        return bets

if __name__ == "__main__":
    DATA_PATH = 'data/processed/features.parquet'
    MODEL_PATH = 'data/models/lgbm_model.pkl'
    
    if os.path.exists(DATA_PATH) and os.path.exists(MODEL_PATH):
        tester = Backtester(DATA_PATH, MODEL_PATH)
        tester.load_resources()
        tester.predict()
        tester.calculate_expected_value()
        
        # 閾値1.0でシミュレーション
        tester.simulate(threshold=1.0)
        
        # 閾値を振ってみる
        print("\n--- Sensitivity Analysis ---")
        for th in [0.8, 1.0, 1.2, 1.5, 2.0]:
            tester.simulate(threshold=th)
            print("-" * 20)
    else:
        print("Resources not found.")
