
#!/bin/bash
set -e

# データ処理が完了している前提
# もし html_parser.py がまだなら:
# uv run src/preprocessing/html_parser.py

echo "=== Step 2: Feature Engineering ==="
uv run src/preprocessing/feature_engineering.py

echo -e "\n=== Step 3: Model Training ==="
uv run src/model/train.py

echo -e "\n=== Step 4: Backtesting ==="
uv run src/simulation/backtest.py
