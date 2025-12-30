"""Dynamic weight management for ensemble models."""

import numpy as np
from collections import deque
from typing import Dict, List
from loguru import logger


class DynamicWeightManager:
    """管理ensemble模型的動態權重"""
    
    def __init__(
        self,
        model_names: List[str],
        window_size: int = 100,
        method: str = 'performance',  # 'performance', 'sharpe', 'fixed'
        initial_weights: Dict[str, float] = None,
    ):
        """
        初始化動態權重管理器
        
        Args:
            model_names: 模型名稱列表
            window_size: 滾動窗口大小
            method: 權重計算方法
            initial_weights: 初始權重字典
        """
        self.model_names = model_names
        self.window_size = window_size
        self.method = method
        
        # 初始化權重
        if initial_weights:
            self.weights = initial_weights.copy()
        else:
            # 平均權重
            n = len(model_names)
            self.weights = {name: 1.0 / n for name in model_names}
        
        # 性能歷史記錄（滾動窗口）
        self.reward_history = {
            name: deque(maxlen=window_size) 
            for name in model_names
        }
        
        # 統計信息
        self.total_updates = 0
        
    def update(self, rewards: Dict[str, float]):
        """
        更新權重基於最新reward
        
        Args:
            rewards: {model_name: reward} 字典
        """
        # 記錄reward
        for name in self.model_names:
            if name in rewards:
                self.reward_history[name].append(rewards[name])
        
        self.total_updates += 1
        
        # 只有足夠數據後才更新權重 (至少10個樣本)
        if self.total_updates < 10:
            return
        
        # 根據方法計算權重
        if self.method == 'fixed':
            # 固定權重，不更新
            pass
        elif self.method == 'performance':
            self._update_by_performance()
        elif self.method == 'sharpe':
            self._update_by_sharpe()
    
    def _update_by_performance(self):
        """基於平均performance更新權重"""
        performances = {}
        
        for name in self.model_names:
            if len(self.reward_history[name]) > 0:
                # 計算滾動平均reward
                performances[name] = np.mean(self.reward_history[name])
            else:
                performances[name] = 0.0
        
        # Softmax歸一化（確保正值）
        # 添加offset避免負數
        min_perf = min(performances.values())
        if min_perf < 0:
            performances = {k: v - min_perf + 1.0 for k, v in performances.items()}
        
        # Softmax with temperature
        temperature = 2.0  # 較高temperature = 更平滑的權重
        exp_perfs = {k: np.exp(v / temperature) for k, v in performances.items()}
        total = sum(exp_perfs.values())
        
        if total > 0:
            self.weights = {k: v / total for k, v in exp_perfs.items()}
        
        logger.debug(f"Updated weights (performance): {self.weights}")
    
    def _update_by_sharpe(self):
        """基於Sharpe ratio更新權重"""
        sharpes = {}
        
        for name in self.model_names:
            if len(self.reward_history[name]) > 1:
                rewards = list(self.reward_history[name])
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                
                # Sharpe ratio
                if std_reward > 1e-8:
                    sharpes[name] = mean_reward / std_reward
                else:
                    sharpes[name] = mean_reward
            else:
                sharpes[name] = 0.0
        
        # Softmax歸一化
        min_sharpe = min(sharpes.values())
        if min_sharpe < 0:
            sharpes = {k: v - min_sharpe + 1.0 for k, v in sharpes.items()}
        
        temperature = 1.5
        exp_sharpes = {k: np.exp(v / temperature) for k, v in sharpes.items()}
        total = sum(exp_sharpes.values())
        
        if total > 0:
            self.weights = {k: v / total for k, v in exp_sharpes.items()}
        
        logger.debug(f"Updated weights (sharpe): {self.weights}")
    
    def get_weights(self) -> Dict[str, float]:
        """獲取當前權重"""
        return self.weights.copy()
    
    def get_statistics(self) -> Dict:
        """獲取統計信息"""
        stats = {
            'weights': self.weights.copy(),
            'total_updates': self.total_updates,
        }
        
        for name in self.model_names:
            if len(self.reward_history[name]) > 0:
                rewards = list(self.reward_history[name])
                stats[f'{name}_mean'] = np.mean(rewards)
                stats[f'{name}_std'] = np.std(rewards)
                stats[f'{name}_samples'] = len(rewards)
        
        return stats
