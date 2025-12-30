#!/bin/bash

# 自動重訓 Pipeline
# 使用方法: ./scripts/auto_retrain.sh

set -e  # 遇到錯誤立即停止

echo "=========================================="
echo "自動重訓 Pipeline 啟動"
echo "時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# 切換到專案根目錄
cd "$(dirname "$0")/.."

# 啟動虛擬環境
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ 虛擬環境已啟動"
else
    echo "✗ 找不到虛擬環境，請先執行: python -m venv venv"
    exit 1
fi

# 1. 收集最新資料
echo ""
echo "[1/6] 收集最新資料..."
python scripts/collect_data.py --days 90

# 2. 執行回測
echo ""
echo "[2/6] 執行回測驗證..."
python scripts/backtest.py \
  --model models/ppo_trading_final \
  --symbol BTCUSDT \
  --save-json reports/latest_backtest_results.json

# 3. 檢查回測結果
echo ""
echo "[3/6] 檢查回測結果..."
if [ -f "reports/latest_backtest_results.json" ]; then
    echo "✓ 回測結果已儲存"
else
    echo "✗ 回測結果儲存失敗"
    exit 1
fi

# 4. 評估性能
echo ""
echo "[4/6] 評估模型性能..."
python scripts/evaluate_model_performance.py \
  --model models/ppo_trading_final \
  --results reports/latest_backtest_results.json

# 5. 讀取評估結果
NEED_RETRAIN=$(python -c "
import json
from pathlib import Path

eval_file = Path('reports/model_evaluation.json')
if eval_file.exists():
    with open(eval_file, 'r') as f:
        result = json.load(f)
    print('yes' if result['need_retrain'] else 'no')
else:
    print('yes')  # 如果沒有評估結果，預設需要重訓
")

# 6. 條件重訓
if [ "$NEED_RETRAIN" = "yes" ]; then
    echo ""
    echo "[5/6] 開始重新訓練..."
    
    # 備份舊模型
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mkdir -p models/archive
    
    if [ -f "models/ppo_trading_final.zip" ]; then
        cp models/ppo_trading_final.zip models/archive/ppo_trading_final_${TIMESTAMP}.zip
        echo "✓ 舊模型已備份至 models/archive/ppo_trading_final_${TIMESTAMP}.zip"
    fi
    
    # 執行訓練
    python scripts/train.py --steps 100000
    
    # 重新回測
    echo ""
    echo "[6/6] 重新回測新模型..."
    python scripts/backtest.py \
      --model models/ppo_trading_final \
      --symbol BTCUSDT \
      --save-json reports/latest_backtest_results.json

    
    echo ""
    echo "=========================================="
    echo "✅ 重訓完成！"
    echo "新模型: models/ppo_trading_final.zip"
    echo "舊模型備份: models/archive/ppo_trading_final_${TIMESTAMP}.zip"
    echo "=========================================="
else
    echo ""
    echo "[5/6] 跳過重訓"
    echo "[6/6] 無需執行"
    echo ""
    echo "=========================================="
    echo "✅ 模型性能良好，無需重訓"
    echo "=========================================="
fi

echo ""
echo "Pipeline 執行完畢 - $(date '+%Y-%m-%d %H:%M:%S')"
