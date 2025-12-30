"""Action Smoothing Wrapper for stable trading"""

import gymnasium as gym
import numpy as np
from typing import Any, SupportsFloat


class ActionSmoothingWrapper(gym.ActionWrapper):
    """
    平滑動作輸出，避免倉位劇烈跳動
    
    new_action = alpha * previous_action + (1-alpha) * model_action
    
    Args:
        env: 環境
        alpha: 平滑係數 (0-1)，越大越平滑
        volatility_scaling: 是否根據市場波動調整最大倉位
    """
    
    def __init__(
        self,
        env: gym.Env,
        alpha: float = 0.7,
        volatility_scaling: bool = True,
        max_volatility_threshold: float = 0.03,
    ):
        super().__init__(env)
        self.alpha = alpha
        self.previous_action = None
        self.volatility_scaling = volatility_scaling
        self.max_volatility_threshold = max_volatility_threshold
        
    def action(self, action: np.ndarray) -> np.ndarray:
        """平滑化action"""
        # 第一次調用
        if self.previous_action is None:
            self.previous_action = np.zeros_like(action)
        
        # 指數移動平均平滑
        smoothed_action = self.alpha * self.previous_action + (1 - self.alpha) * action
        
        # 波動率調整（如果啟用）
        if self.volatility_scaling:
            # 獲取當前市場波動率（從環境）
            try:
                # 假設環境有get_current_volatility方法
                current_volatility = self._get_market_volatility()
                
                # 如果波動率過高，強制降低倉位上限
                if current_volatility > self.max_volatility_threshold:
                    scale_factor = self.max_volatility_threshold / current_volatility
                    smoothed_action = smoothed_action * scale_factor
            except:
                pass  # 環境不支持則跳過
        
        # 限制在[-1, 1]
        smoothed_action = np.clip(smoothed_action, -1.0, 1.0)
        
        # 保存用於下次
        self.previous_action = smoothed_action.copy()
        
        return smoothed_action
    
    def _get_market_volatility(self) -> float:
        """獲取當前市場波動率"""
        # 嘗試從環境的returns_history計算
        if hasattr(self.env.unwrapped, 'returns_history'):
            returns = self.env.unwrapped.returns_history
            if len(returns) > 10:
                return np.std(returns[-10:])
        return 0.01  # 默認值
    
    def reset(self, **kwargs):
        """重置時清空previous_action"""
        self.previous_action = None
        return self.env.reset(**kwargs)


class VolatilityAwareWrapper(gym.Wrapper):
    """
    根據市場波動率動態調整風險參數
    
    - 高波動：降低最大倉位限制
    - 低波動：正常倉位
    """
    
    def __init__(
        self,
        env: gym.Env,
        volatility_window: int = 20,
        high_vol_threshold: float = 0.03,
        low_vol_threshold: float = 0.01,
    ):
        super().__init__(env)
        self.volatility_window = volatility_window
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        
    def step(self, action):
        """在step前調整action based on volatility"""
        # 獲取當前波動率
        volatility = self._calculate_volatility()
        
        # 根據波動率調整action幅度
        if volatility > self.high_vol_threshold:
            # 高波動：減半倉位
            action = action * 0.5
        elif volatility > self.low_vol_threshold:
            # 中等波動：70%倉位
            action = action * 0.7
        # 低波動：正常
        
        return self.env.step(action)
    
    def _calculate_volatility(self) -> float:
        """計算近期波動率"""
        if hasattr(self.env.unwrapped, 'returns_history'):
            returns = self.env.unwrapped.returns_history
            if len(returns) >= self.volatility_window:
                return np.std(returns[-self.volatility_window:])
        return 0.01  # 默認


def create_stable_env(base_env: gym.Env, config: dict = None) -> gym.Env:
    """
    創建穩定性優化的環境
    
    Args:
        base_env: 基礎環境
        config: 配置dict，包含:
            - action_smoothing_alpha: 平滑係數 (default: 0.7)
            - enable_volatility_scaling: 是否啟用波動率調整 (default: True)
    
    Returns:
        wrapped environment
    """
    if config is None:
        config = {}
    
    # 1. Action Smoothing
    env = ActionSmoothingWrapper(
        base_env,
        alpha=config.get('action_smoothing_alpha', 0.7),
        volatility_scaling=config.get('enable_volatility_scaling', True),
    )
    
    # 2. Volatility Aware (optional)
    if config.get('enable_volatility_wrapper', False):
        env = VolatilityAwareWrapper(env)
    
    return env
