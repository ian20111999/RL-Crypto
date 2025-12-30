"""
優化的Reward Functions

測試3種配置：
v1: 激進收益導向
v2: 平衡型
v3: Sharpe優化型
"""

import numpy as np


def calculate_reward_v1_aggressive(env, actions):
    """
    V1: 激進收益導向
    - 提高收益權重3x
    - 大幅降低交易懲罰
    - 目標：最大化絕對收益
    """
    # 收益（3x權重）
    returns = (env.portfolio_value - env.prev_portfolio_value) / env.prev_portfolio_value
    reward = returns * env.reward_scaling * 3.0  # 核心改變：3x收益
    
    # 輕微交易懲罰（降低10x）
    turnover = np.sum(np.abs(actions - env.prev_positions))
    reward -= env.transaction_penalty * turnover * 0.1  # 降低到1/10
    
    # 回撤懲罰（保持）
    if env.peak_value > 0:
        current_dd = (env.peak_value - env.portfolio_value) / env.peak_value
        reward -= env.drawdown_penalty * current_dd
    
    # 移除action change懲罰（太嚴格）
    # 允許更靈活的倉位調整
    
    # 增強持倉獎勵（10x）
    total_position = np.sum(np.abs(env.positions))
    if total_position > 0.1:
        holding_bonus = 0.01 * total_position  # 從0.001 → 0.01
        reward += holding_bonus
    
    # 增強Sharpe獎勵（10x）
    if len(env.returns_history) > 20:
        mean_return = np.mean(env.returns_history[-20:])
        std_return = np.std(env.returns_history[-20:]) + 1e-8
        sharpe = mean_return / std_return
        reward += 0.1 * sharpe  # 從0.01 → 0.1
    
    return float(reward)


def calculate_reward_v2_balanced(env, actions):
    """
    V2: 平衡型
    - 提高收益權重2x
    - 適度降低交易懲罰
    - 保留風控
    """
    # 收益（2x權重）
    returns = (env.portfolio_value - env.prev_portfolio_value) / env.prev_portfolio_value
    reward = returns * env.reward_scaling * 2.0
    
    # 適度交易懲罰（降低5x）
    turnover = np.sum(np.abs(actions - env.prev_positions))
    reward -= env.transaction_penalty * turnover * 0.2
    
    # 回撤懲罰（稍微降低）
    if env.peak_value > 0:
        current_dd = (env.peak_value - env.portfolio_value) / env.peak_value
        reward -= env.drawdown_penalty * current_dd * 0.8
    
    # 輕微action change懲罰（從5x → 2x）
    action_change = np.sum(np.abs(actions - env.prev_positions))
    if action_change > 0.15:  # 提高閾值 0.1 → 0.15
        reward -= env.transaction_penalty * action_change * 2  # 從5x → 2x
    
    # 持倉獎勵（5x）
    total_position = np.sum(np.abs(env.positions))
    if total_position > 0.1:
        holding_bonus = 0.005 * total_position
        reward += holding_bonus
    
    # Sharpe獎勵（5x）
    if len(env.returns_history) > 20:
        mean_return = np.mean(env.returns_history[-20:])
        std_return = np.std(env.returns_history[-20:]) + 1e-8
        sharpe = mean_return / std_return
        reward += 0.05 * sharpe
    
    # 保留diversification bonus
    position_concentration = np.std(np.abs(env.positions))
    reward += 0.002 * (1 - position_concentration)
    
    return float(reward)


def calculate_reward_v3_sharpe(env, actions):
    """
    V3: Sharpe優化型
    - 重視風險調整收益
    - 獎勵穩定性
    """
    # 收益（1.5x權重）
    returns = (env.portfolio_value - env.prev_portfolio_value) / env.prev_portfolio_value
    reward = returns * env.reward_scaling * 1.5
    
    # 低交易懲罰
    turnover = np.sum(np.abs(actions - env.prev_positions))
    reward -= env.transaction_penalty * turnover * 0.3
    
    # 強回撤懲罰（保持風控）
    if env.peak_value > 0:
        current_dd = (env.peak_value - env.portfolio_value) / env.peak_value
        reward -= env.drawdown_penalty * current_dd * 1.2  # 提高
    
    # 無action change懲罰
    
    # 持倉獎勵
    total_position = np.sum(np.abs(env.positions))
    if total_position > 0.1:
        holding_bonus = 0.005 * total_position
        reward += holding_bonus
    
    # 超強Sharpe獎勵（20x）
    if len(env.returns_history) > 20:
        mean_return = np.mean(env.returns_history[-20:])
        std_return = np.std(env.returns_history[-20:]) + 1e-8
        sharpe = mean_return / std_return
        reward += 0.2 * sharpe  # 從0.01 → 0.2 (20x)
    
    # 穩定性獎勵（獎勵低波動）
    if len(env.returns_history) > 10:
        recent_volatility = np.std(env.returns_history[-10:])
        stability_bonus = 0.01 / (recent_volatility + 1e-6)
        reward += stability_bonus
    
    return float(reward)


# 對比表
REWARD_CONFIGS = {
    "original": {
        "name": "原版",
        "return_weight": 1.0,
        "transaction_penalty": 1.0,
        "特點": "保守，過度懲罰交易"
    },
    "v1_aggressive": {
        "name": "激進型",
        "return_weight": 3.0,
        "transaction_penalty": 0.1,
        "特點": "最大化收益，允許頻繁交易"
    },
    "v2_balanced": {
        "name": "平衡型",
        "return_weight": 2.0,
        "transaction_penalty": 0.2,
        "特點": "收益與風控平衡"
    },
    "v3_sharpe": {
        "name": "Sharpe型",
        "return_weight": 1.5,
        "transaction_penalty": 0.3,
        "特點": "優化風險調整收益"
    },
}
