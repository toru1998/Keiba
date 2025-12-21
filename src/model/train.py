
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import pickle

class ModelTrainer:
    def __init__(self, data_path, model_dir):
        self.data_path = data_path
        self.model_dir = model_dir
        self.model = None
        self.df = None
        
    def load_data(self):
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        # 必要な列だけに絞る
        # targetはfeature_engineeringで作った 'target' (3着以内など)
        # 学習に使わない列を除外
        self.drop_cols = ['rank', 'time', 'time_seconds', 'target', 'race_id', 'date', 'race_name', 'race_round']
        
        # 実際にはfeature_engineeringでエンコード済みの列を使う
        # 文字列型が残っているとLightGBMは扱えるが、object型はcategory型にする必要がある
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col] = self.df[col].astype('category')

    def train(self):
        print("Training model...")
        
        # 特徴量とターゲット
        X = self.df.drop(columns=[c for c in self.drop_cols if c in self.df.columns])
        y = self.df['target']
        
        # データの分割 (時系列分割が望ましいが、簡易的にランダム分割)
        # TODO: 時系列を考慮した分割 (race_idなどでソートされている前提なら前半・後半で分ける)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # LightGBMデータセット
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # パラメータ
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # 学習
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, test_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(100)
            ]
        )
        
        # 評価
        y_pred = self.model.predict(X_test, num_iteration=self.model.best_iteration)
        auc = roc_auc_score(y_test, y_pred)
        print(f"Test AUC: {auc:.4f}")
        
        # 特徴量重要度
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(importance.head(10))

    def save_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, 'lgbm_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    DATA_PATH = 'data/processed/features.parquet'
    MODEL_DIR = 'data/models'
    
    if os.path.exists(DATA_PATH):
        trainer = ModelTrainer(DATA_PATH, MODEL_DIR)
        trainer.load_data()
        trainer.train()
        trainer.save_model()
    else:
        print("Features data not found. Run feature_engineering.py first.")
