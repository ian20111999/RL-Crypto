# 🚀 RL-Crypto

基於 PPO 強化學習的加密貨幣合約交易系統。

## ✨ 功能特色

- **多幣種交易**：同時管理 BTC、ETH 等多個合約
- **PPO 訓練**：Stable-Baselines3 + 自定義交易環境
- **多策略集成**：PPO + SAC + TD3 投票機制
- **風險管理**：動態止損、回撤熔斷、倉位限制
- **市場狀態檢測**：牛熊識別、波動率 regime
- **智能倉位管理**：Kelly Criterion、Risk Parity
- **實盤接口**：WebSocket 實時數據 + 訂單執行
- **監控 Dashboard**：Streamlit 可視化界面

## 🚀 快速開始

```bash
# 1. 安裝
cd RL-Crypto
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. 配置 API (複製並編輯)
cp .env.example .env

# 3. 收集數據
python scripts/collect_data.py --days 7

# 4. 訓練
python scripts/train.py --dry-run  # 快速測試
python scripts/train.py --steps 100000  # 完整訓練

# 5. 超參數調優
python scripts/optimize.py --trials 50 --timesteps 50000

# 6. 回測
python scripts/backtest.py --symbol BTCUSDT

# 7. 啟動 Dashboard
streamlit run scripts/dashboard.py

# 8. 實盤交易 (Testnet)
python -m src.live.executor --model models/ppo_trading_final --live
```

## 🔔 Telegram 通知

系統支持 Telegram 即時通知，在交易時自動推送訊息：

```bash
# 設定 .env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 📁 項目結構

```
RL-Crypto/
├── config/config.yaml        # 主配置
├── src/
│   ├── data/                 # 數據層
│   ├── envs/                 # 交易環境
│   ├── agents/               # RL Agents (PPO/SAC/TD3)
│   ├── live/                 # 實盤模塊
│   ├── analysis/             # 分析模塊
│   └── backtesting/          # 回測
├── scripts/
│   ├── train.py              # 訓練
│   ├── backtest.py           # 回測
│   ├── optimize.py           # Optuna 調優
│   ├── collect_data.py       # 數據收集
│   └── dashboard.py          # Streamlit
└── tests/                    # 測試
```

## 🛡️ 風險管理

- **最大回撤**：20% 時暫停交易
- **單日虧損**：5% 限制
- **單倉止損**：2%
- **最大持倉**：每幣種 20%

## ⚠️ 風險警告

> **加密貨幣合約交易風險極高**，可能導致全部本金損失。
> 本專案僅供學習研究，請勿用於實盤交易。

## 📈 開發路線

- [x] 核心交易環境 + PPO
- [x] 回測引擎 + Optuna 調優
- [x] 風險管理 + 實盤接口
- [x] 多策略集成 + 市場檢測
- [x] Kelly/Risk Parity + Dashboard
- [x] 實盤測試 (Testnet)
