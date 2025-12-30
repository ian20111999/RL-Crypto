"""評估模型性能並判斷是否需要重新訓練"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
from loguru import logger

# 性能閾值配置
PERFORMANCE_THRESHOLDS = {
    "min_sharpe_ratio": 1.0,        # 最低 Sharpe Ratio
    "min_total_return": 0.05,       # 最低總回報 5%
    "max_drawdown": 0.20,           # 最大回撤 20%
    "min_win_rate": 0.45,           # 最低勝率 45%
    "max_volatility": 0.30,         # 最大波動率 30%
}

# 重訓觸發條件
RETRAIN_CONDITIONS = {
    "performance_degradation": True,  # 性能低於閾值
    "data_drift": True,                # 資料分佈變化
    "time_based": True,                # 定期重訓（每 30 天）
    "max_days_since_training": 30,     # 最長訓練間隔
}


def load_backtest_results(report_path: str) -> dict:
    """載入回測結果"""
    results_file = Path(report_path)
    
    if not results_file.exists():
        logger.warning(f"找不到回測結果: {results_file}")
        return {}
    
    with open(results_file, 'r') as f:
        return json.load(f)


def check_performance_thresholds(metrics: dict) -> dict:
    """檢查性能是否達標"""
    checks = {}
    
    # Sharpe Ratio
    sharpe = metrics.get("sharpe_ratio", 0)
    checks["sharpe_pass"] = sharpe >= PERFORMANCE_THRESHOLDS["min_sharpe_ratio"]
    
    # Total Return
    total_return = metrics.get("total_return_pct", 0) / 100
    checks["return_pass"] = total_return >= PERFORMANCE_THRESHOLDS["min_total_return"]
    
    # Max Drawdown
    max_dd = abs(metrics.get("max_drawdown_pct", 100)) / 100
    checks["drawdown_pass"] = max_dd <= PERFORMANCE_THRESHOLDS["max_drawdown"]
    
    # Win Rate
    win_rate = metrics.get("win_rate", 0)
    checks["winrate_pass"] = win_rate >= PERFORMANCE_THRESHOLDS["min_win_rate"]
    
    # Volatility
    volatility = metrics.get("volatility", 1.0)
    checks["volatility_pass"] = volatility <= PERFORMANCE_THRESHOLDS["max_volatility"]
    
    checks["all_pass"] = all(checks.values())
    
    return checks


def check_data_drift() -> bool:
    """檢查資料分佈是否變化（簡化版）"""
    # 比較訓練資料和最新資料的統計特性
    try:
        train_data = pd.read_csv("data/BTCUSDT.csv")
        
        if len(train_data) < 2160:
            logger.warning("資料不足，跳過漂移檢查")
            return False
        
        # 計算最近 30 天和歷史資料的統計差異
        recent_data = train_data.tail(720)  # 30 days * 24 hours
        historical_data = train_data.iloc[-2160:-720]  # 60-30 days ago
        
        # 比較波動率
        recent_vol = recent_data['close'].pct_change().std()
        hist_vol = historical_data['close'].pct_change().std()
        
        vol_change = abs(recent_vol - hist_vol) / hist_vol
        
        # 如果波動率變化超過 50%，視為資料漂移
        drift_detected = vol_change > 0.5
        
        logger.info(f"資料漂移檢查: 波動率變化 {vol_change*100:.2f}%")
        
        return drift_detected
        
    except Exception as e:
        logger.error(f"資料漂移檢查失敗: {e}")
        return False


def check_time_based_retrain(model_path: str) -> bool:
    """檢查是否需要定期重訓"""
    model_file = Path(model_path + ".zip")
    
    if not model_file.exists():
        logger.warning(f"模型檔案不存在: {model_file}")
        return True
    
    # 獲取模型最後修改時間
    model_mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
    days_since_training = (datetime.now() - model_mtime).days
    
    logger.info(f"模型訓練於 {days_since_training} 天前")
    
    return days_since_training >= RETRAIN_CONDITIONS["max_days_since_training"]


def should_retrain(
    metrics: dict,
    model_path: str,
    check_drift: bool = True,
    check_time: bool = True,
) -> tuple[bool, list[str]]:
    """判斷是否需要重新訓練
    
    Returns:
        (should_retrain, reasons)
    """
    reasons = []
    
    # 1. 性能檢查
    if RETRAIN_CONDITIONS["performance_degradation"]:
        perf_checks = check_performance_thresholds(metrics)
        
        if not perf_checks["all_pass"]:
            failed_checks = [k for k, v in perf_checks.items() if not v and k != "all_pass"]
            reasons.append(f"性能未達標: {', '.join(failed_checks)}")
    
    # 2. 資料漂移檢查
    if check_drift and RETRAIN_CONDITIONS["data_drift"]:
        if check_data_drift():
            reasons.append("檢測到資料分佈變化")
    
    # 3. 時間檢查
    if check_time and RETRAIN_CONDITIONS["time_based"]:
        if check_time_based_retrain(model_path):
            reasons.append(f"超過 {RETRAIN_CONDITIONS['max_days_since_training']} 天未重訓")
    
    return len(reasons) > 0, reasons


def main():
    parser = argparse.ArgumentParser(description="評估模型性能並判斷是否重訓")
    parser.add_argument("--model", type=str, default="models/ppo_trading_final", help="模型路徑")
    parser.add_argument("--results", type=str, default="reports/latest_backtest_results.json", help="回測結果路徑")
    parser.add_argument("--no-drift-check", action="store_true", help="跳過資料漂移檢查")
    parser.add_argument("--no-time-check", action="store_true", help="跳過時間檢查")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("模型性能評估與重訓判斷")
    logger.info("=" * 60)
    
    # 載入回測結果
    metrics = load_backtest_results(args.results)
    
    if not metrics:
        logger.error("無法載入回測結果，建議重新訓練")
        return
    
    # 顯示當前性能
    logger.info("\n當前模型性能:")
    logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
    logger.info(f"  Total Return: {metrics.get('total_return_pct', 'N/A')}%")
    logger.info(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 'N/A')}%")
    logger.info(f"  Win Rate: {metrics.get('win_rate', 'N/A')}")
    
    # 判斷是否重訓
    need_retrain, reasons = should_retrain(
        metrics,
        args.model,
        check_drift=not args.no_drift_check,
        check_time=not args.no_time_check,
    )
    
    logger.info("\n" + "=" * 60)
    if need_retrain:
        logger.warning("⚠️  建議重新訓練模型")
        logger.warning("原因:")
        for reason in reasons:
            logger.warning(f"  - {reason}")
        logger.info("\n執行重訓指令:")
        logger.info("  python scripts/train.py --steps 100000")
    else:
        logger.info("✅ 模型性能良好，無需重訓")
    logger.info("=" * 60)
    
    # 儲存評估結果
    evaluation_result = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model,
        "metrics": metrics,
        "need_retrain": need_retrain,
        "reasons": reasons,
    }
    
    eval_file = Path("reports/model_evaluation.json")
    eval_file.parent.mkdir(exist_ok=True)
    
    with open(eval_file, 'w') as f:
        json.dump(evaluation_result, f, indent=2)
    
    logger.info(f"\n評估結果已儲存至: {eval_file}")


if __name__ == "__main__":
    main()
