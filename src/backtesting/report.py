"""Enhanced backtesting visualization and reporting."""

import matplotlib
matplotlib.use('Agg')  # 非交互式後端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

# 配置中文字體
plt.rcParams['font.sans-serif'] = ['Heiti TC', 'Songti SC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class BacktestReport:
    """Generate detailed backtest reports with visualizations."""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def generate_full_report(
        self,
        results: Dict[str, Dict],
        timestamps: Optional[pd.DatetimeIndex] = None,
        save_html: bool = True,
    ) -> None:
        """Generate complete backtest report with all visualizations.
        
        Args:
            results: Dict of strategy results
            timestamps: Optional timestamps for x-axis
            save_html: Whether to save HTML report
        """
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"backtest_{report_time}"
        report_dir.mkdir(exist_ok=True)
        
        # Generate all charts
        self.plot_equity_curves(results, timestamps, report_dir)
        self.plot_returns_distribution(results, report_dir)
        self.plot_drawdown_curves(results, timestamps, report_dir)
        self.plot_strategy_comparison(results, report_dir)
        self.plot_monthly_returns(results, timestamps, report_dir)
        self.plot_cumulative_pnl(results, timestamps, report_dir)
        
        # Generate HTML report
        if save_html:
            self._generate_html_report(results, report_dir)
        
        print(f"\n✅ Report generated: {report_dir}")
    
    def plot_equity_curves(
        self,
        results: Dict[str, Dict],
        timestamps: Optional[pd.DatetimeIndex],
        save_dir: Path,
    ) -> None:
        """Plot equity curves for all strategies."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for strategy_name, result in results.items():
            equity = result.get("equity_curve", [])
            if len(equity) == 0:
                continue
                
            if timestamps is not None and len(timestamps) == len(equity):
                ax.plot(timestamps, equity, label=strategy_name, linewidth=2)
            else:
                ax.plot(equity, label=strategy_name, linewidth=2)
        
        ax.set_title("資金曲線對比", fontsize=16, fontweight='bold')
        ax.set_xlabel("時間")
        ax.set_ylabel("資金 (USDT)")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if timestamps is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / "equity_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_returns_distribution(
        self,
        results: Dict[str, Dict],
        save_dir: Path,
    ) -> None:
        """Plot returns distribution for all strategies."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("收益分布分析", fontsize=16, fontweight='bold')
        
        strategy_names = list(results.keys())
        
        for idx, strategy_name in enumerate(strategy_names):
            if idx >= 4:  # Max 4 subplots
                break
                
            ax = axes[idx // 2, idx % 2]
            result = results[strategy_name]
            
            # Get daily returns
            equity = result.get("equity_curve", [])
            if len(equity) > 1:
                returns = np.diff(equity) / equity[:-1] * 100  # Percentage returns
                
                # Histogram
                ax.hist(returns, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                ax.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'平均: {np.mean(returns):.2f}%')
                ax.axvline(np.median(returns), color='green', linestyle='--', linewidth=2, label=f'中位數: {np.median(returns):.2f}%')
                
                ax.set_title(strategy_name)
                ax.set_xlabel("收益率 (%)")
                ax.set_ylabel("頻率")
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "returns_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_drawdown_curves(
        self,
        results: Dict[str, Dict],
        timestamps: Optional[pd.DatetimeIndex],
        save_dir: Path,
    ) -> None:
        """Plot drawdown curves."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for strategy_name, result in results.items():
            equity = np.array(result.get("equity_curve", []))
            if len(equity) == 0:
                continue
            
            # Calculate drawdown
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max * 100
            
            if timestamps is not None and len(timestamps) == len(drawdown):
                ax.plot(timestamps, drawdown, label=strategy_name, linewidth=2)
            else:
                ax.plot(drawdown, label=strategy_name, linewidth=2)
        
        ax.set_title("回撤曲線", fontsize=16, fontweight='bold')
        ax.set_xlabel("時間")
        ax.set_ylabel("回撤 (%)")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        if timestamps is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / "drawdown_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_strategy_comparison(
        self,
        results: Dict[str, Dict],
        save_dir: Path,
    ) -> None:
        """Plot strategy comparison metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("策略對比", fontsize=16, fontweight='bold')
        
        strategies = list(results.keys())
        metrics = {
            'total_return_pct': [],
            'sharpe_ratio': [],
            'max_drawdown_pct': [],
            'n_trades': [],
        }
        
        for strategy in strategies:
            result = results[strategy]
            
            # Safely extract metrics (handle both dict and scalar values)
            total_return = result.get('total_return_pct', 0)
            if isinstance(total_return, dict):
                total_return = total_return.get('value', 0)
            metrics['total_return_pct'].append(total_return)
            
            sharpe = result.get('sharpe_ratio', 0)
            if isinstance(sharpe, dict):
                sharpe = sharpe.get('value', 0)
            metrics['sharpe_ratio'].append(sharpe)
            
            max_dd = result.get('max_drawdown_pct', 0)
            if isinstance(max_dd, dict):
                max_dd = max_dd.get('value', 0)
            metrics['max_drawdown_pct'].append(abs(max_dd) if max_dd else 0)
            
            n_trades = result.get('n_trades', 0)
            if isinstance(n_trades, dict):
                n_trades = n_trades.get('value', 0)
            metrics['n_trades'].append(n_trades)
        
        # Total Return
        ax = axes[0, 0]
        bars = ax.bar(strategies, metrics['total_return_pct'], color=['green' if x > 0 else 'red' for x in metrics['total_return_pct']])
        ax.set_title("總收益率 (%)")
        ax.set_ylabel("%")
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Sharpe Ratio
        ax = axes[0, 1]
        ax.bar(strategies, metrics['sharpe_ratio'], color='steelblue')
        ax.set_title("Sharpe Ratio")
        ax.set_ylabel("Ratio")
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Max Drawdown
        ax = axes[1, 0]
        ax.bar(strategies, metrics['max_drawdown_pct'], color='darkred')
        ax.set_title("最大回撤 (%)")
        ax.set_ylabel("%")
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Number of Trades
        ax = axes[1, 1]
        ax.bar(strategies, metrics['n_trades'], color='orange')
        ax.set_title("交易次數")
        ax.set_ylabel("次數")
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_dir / "strategy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_monthly_returns(
        self,
        results: Dict[str, Dict],
        timestamps: Optional[pd.DatetimeIndex],
        save_dir: Path,
    ) -> None:
        """Plot monthly returns heatmap."""
        if timestamps is None:
            return
        
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))
        if len(results) == 1:
            axes = [axes]
        
        fig.suptitle("每月收益率", fontsize=16, fontweight='bold')
        
        for idx, (strategy_name, result) in enumerate(results.items()):
            equity = np.array(result.get("equity_curve", []))
            if len(equity) == 0 or len(equity) != len(timestamps):
                continue
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': timestamps,
                'equity': equity
            })
            df.set_index('date', inplace=True)
            
            # Calculate monthly returns
            monthly = df.resample('M').last()
            monthly['return'] = monthly['equity'].pct_change() * 100
            
            # Create pivot table for heatmap
            monthly['year'] = monthly.index.year
            monthly['month'] = monthly.index.month
            pivot = monthly.pivot_table(values='return', index='year', columns='month')
            
            # Skip if pivot is empty
            if pivot.empty or pivot.size == 0:
                logger.warning(f"Skipping monthly returns for {strategy_name}: no data")
                continue
            
            # Plot heatmap
            ax = axes[idx]
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                       cbar_kws={'label': '收益率 (%)'}, ax=ax)
            ax.set_title(strategy_name)
            ax.set_xlabel("月份")
            ax.set_ylabel("年份")
        
        plt.tight_layout()
        plt.savefig(save_dir / "monthly_returns.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_cumulative_pnl(
        self,
        results: Dict[str, Dict],
        timestamps: Optional[pd.DatetimeIndex],
        save_dir: Path,
    ) -> None:
        """Plot cumulative PnL."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for strategy_name, result in results.items():
            equity = np.array(result.get("equity_curve", []))
            if len(equity) == 0:
                continue
            
            initial_capital = equity[0]
            pnl = equity - initial_capital
            
            if timestamps is not None and len(timestamps) == len(pnl):
                ax.plot(timestamps, pnl, label=strategy_name, linewidth=2)
            else:
                ax.plot(pnl, label=strategy_name, linewidth=2)
        
        ax.set_title("累積 PnL", fontsize=16, fontweight='bold')
        ax.set_xlabel("時間")
        ax.set_ylabel("PnL (USDT)")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        if timestamps is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / "cumulative_pnl.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(
        self,
        results: Dict[str, Dict],
        report_dir: Path,
    ) -> None:
        """Generate HTML report with all visualizations."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>回測報告</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-table th, .summary-table td {{
            padding: 12px;
            border: 1px solid #ddd;
            text-align: right;
        }}
        .summary-table th {{
            background-color: #4CAF50;
            color: white;
            text-align: left;
        }}
        .summary-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .positive {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .negative {{
            color: #f44336;
            font-weight: bold;
        }}
        .chart {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart img {{
            width: 100%;
            height: auto;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>📊 RL-Crypto 回測報告</h1>
    <p class="timestamp">生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>策略表現總結</h2>
    <table class="summary-table">
        <tr>
            <th>策略</th>
            <th>總收益率</th>
            <th>Sharpe Ratio</th>
            <th>最大回撤</th>
            <th>交易次數</th>
            <th>勝率</th>
        </tr>
"""
        
        for strategy_name, result in results.items():
            total_return = result.get('total_return_pct', 0)
            return_class = 'positive' if total_return > 0 else 'negative'
            
            html_content += f"""
        <tr>
            <td style="text-align: left;"><strong>{strategy_name}</strong></td>
            <td class="{return_class}">{total_return:.2f}%</td>
            <td>{result.get('sharpe_ratio', 0):.2f}</td>
            <td class="negative">{result.get('max_drawdown_pct', 0):.2f}%</td>
            <td>{result.get('n_trades', 0)}</td>
            <td>{result.get('win_rate', 0):.1f}%</td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>視覺化分析</h2>
    
    <div class="chart">
        <h3>資金曲線對比</h3>
        <img src="equity_curves.png" alt="資金曲線">
    </div>
    
    <div class="chart">
        <h3>回撤曲線</h3>
        <img src="drawdown_curves.png" alt="回撤曲線">
    </div>
    
    <div class="chart">
        <h3>策略對比</h3>
        <img src="strategy_comparison.png" alt="策略對比">
    </div>
    
    <div class="chart">
        <h3>收益分布</h3>
        <img src="returns_distribution.png" alt="收益分布">
    </div>
    
    <div class="chart">
        <h3>累積 PnL</h3>
        <img src="cumulative_pnl.png" alt="累積 PnL">
    </div>
    
    <div class="chart">
        <h3>每月收益率</h3>
        <img src="monthly_returns.png" alt="每月收益率">
    </div>
    
</body>
</html>
"""
        
        with open(report_dir / "report.html", "w", encoding="utf-8") as f:
            f.write(html_content)
