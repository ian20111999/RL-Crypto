"""穩定性優先的Reward Function"""

import numpy as np


def calculate_reward_stability(env, actions: np.ndarray) -> float:
    """
    穩定性優先的reward function
    
    目標：斜率接近1的穩定收益曲線
    - 使用log return (可加性)
    - 懲罰波動率
    - 非線性懲罰回撤
    - 懲罰交易成本
    """
    # 1. 基礎收益（使用log return）
    if env.prev_portfolio_value > 0:
        log_return = np.log(env.portfolio_value / env.prev_portfolio_value)
    else:
        log_return = 0.0
    
    reward = log_return * 100.0  # Scale up
    
    # 2. 波動率懲罰（核心：懲罰收益不穩定）
    if len(env.returns_history) > 10:
        recent_returns = env.returns_history[-10:]
        returns_std = np.std(recent_returns)
        volatility_penalty = returns_std * 50.0  # 強懲罰波動
        reward -= volatility_penalty
    
    # 3. 回撤懲罰（非線性，回撤越大懲罰越重）
    if env.peak_value > 0:
        current_dd = (env.peak_value - env.portfolio_value) / env.peak_value
        drawdown_penalty = (current_dd ** 2) * 1000.0  # 平方懲罰
        reward -= drawdown_penalty
    
    # 4. 交易成本懲罰
    turnover = np.sum(np.abs(actions - env.prev_positions))
    fee_penalty = turnover * env.transaction_penalty * 100.0
    reward -= fee_penalty
    
    # 5. Sharpe獎勵（鼓勵風險調整後收益）
    if len(env.returns_history) > 20:
        mean_return = np.mean(env.returns_history[-20:])
        std_return = np.std(env.returns_history[-20:]) + 1e-8
        sharpe = mean_return / std_return
        reward += sharpe * 10.0  # Sharpe獎勵
    
    # 6. 持倉穩定性獎勵（鼓勵持有而非頻繁調倉）
    position_change = np.sum(np.abs(actions - env.prev_positions))
    if position_change < 0.1:  # 幾乎沒變
        stability_bonus = 0.5
        reward += stability_bonus
    
    return float(reward)


def calculate_reward_moderate(env, actions: np.ndarray) -> float:
    """
    中等激進版本
    - 波動懲罰較輕
    - 允許適度交易
    """
    if env.prev_portfolio_value > 0:
        log_return = np.log(env.portfolio_value / env.prev_portfolio_value)
    else:
        log_return = 0.0
    
    reward = log_return * 100.0
    
    # 較輕的波動懲罰
    if len(env.returns_history) > 10:
        recent_returns = env.returns_history[-10:]
        returns_std = np.std(recent_returns)
        volatility_penalty = returns_std * 20.0  # 較輕
        reward -= volatility_penalty
    
    # 回撤懲罰
    if env.peak_value > 0:
        current_dd = (env.peak_value - env.portfolio_value) / env.peak_value
        drawdown_penalty = (current_dd ** 2) * 500.0
        reward -= drawdown_penalty
    
    # 交易成本
    turnover = np.sum(np.abs(actions - env.prev_positions))
    fee_penalty = turnover * env.transaction_penalty * 50.0
    reward -= fee_penalty
    
    # Sharpe獎勵
    if len(env.returns_history) > 20:
        mean_return = np.mean(env.returns_history[-20:])
        std_return = np.std(env.returns_history[-20:]) + 1e-8
        sharpe = mean_return / std_return
        reward += sharpe * 5.0
    
    return float(reward)
