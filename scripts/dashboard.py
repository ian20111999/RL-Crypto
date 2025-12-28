"""Streamlit monitoring dashboard for RL-Crypto."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import time


# Page config
st.set_page_config(
    page_title="RL-Crypto Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_config():
    """Load configuration."""
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def load_training_logs():
    """Load training logs from tensorboard."""
    from tensorboard.backend.event_processing import event_accumulator
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    
    # Find latest log directory
    log_dirs = sorted(logs_dir.glob("ppo_trading_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not log_dirs:
        return None
    
    latest_dir = log_dirs[0]
    
    # Load events
    try:
        ea = event_accumulator.EventAccumulator(str(latest_dir))
        ea.Reload()
        
        data = {}
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            data[tag] = pd.DataFrame([
                {"step": e.step, "value": e.value, "wall_time": e.wall_time}
                for e in events
            ])
        
        return data
    except:
        return None


def load_data_files():
    """Load available data files."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    
    files = []
    for f in data_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
            files.append({
                "file": f.name,
                "symbol": f.stem.split("_")[0],
                "rows": len(df),
                "start": df["open_time"].min() if "open_time" in df else None,
                "end": df["open_time"].max() if "open_time" in df else None,
                "size_mb": f.stat().st_size / 1024 / 1024,
            })
        except:
            pass
    
    return files


def load_model_info():
    """Load model information."""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    models = []
    for f in models_dir.glob("*.zip"):
        models.append({
            "name": f.stem,
            "path": str(f),
            "size_mb": f.stat().st_size / 1024 / 1024,
            "modified": datetime.fromtimestamp(f.stat().st_mtime),
        })
    
    return sorted(models, key=lambda x: x["modified"], reverse=True)


def render_sidebar():
    """Render sidebar."""
    st.sidebar.title("🚀 RL-Crypto")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "📈 Training", "🔬 Backtest", "⚙️ Settings", "🔴 Live"]
    )
    
    st.sidebar.markdown("---")
    
    # Status
    config = load_config()
    if config:
        st.sidebar.markdown("### Status")
        st.sidebar.metric("Capital", f"${config.get('trading', {}).get('initial_capital', 0)}")
        st.sidebar.metric("Leverage", f"{config.get('trading', {}).get('leverage', 1)}x")
        st.sidebar.metric("Symbols", len(config.get("symbols", [])))
    
    return page


def render_overview():
    """Render overview page."""
    st.title("📊 Trading System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Load data
    data_files = load_data_files()
    models = load_model_info()
    
    with col1:
        st.metric("Data Files", len(data_files))
    with col2:
        total_rows = sum(f["rows"] for f in data_files) if data_files else 0
        st.metric("Total Data Points", f"{total_rows:,}")
    with col3:
        st.metric("Trained Models", len(models))
    with col4:
        config = load_config()
        st.metric("Symbols", len(config.get("symbols", [])))
    
    st.markdown("---")
    
    # Data files
    st.subheader("📁 Data Files")
    if data_files:
        df = pd.DataFrame(data_files)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No data files found. Run `python scripts/collect_data.py` to collect data.")
    
    st.markdown("---")
    
    # Models
    st.subheader("🤖 Models")
    if models:
        df = pd.DataFrame(models)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No trained models found. Run `python scripts/train.py` to train a model.")
    
    st.markdown("---")
    
    # Price chart
    st.subheader("💹 Latest Prices")
    if data_files:
        symbol = st.selectbox("Select Symbol", [f["symbol"] for f in data_files])
        data_path = Path("data") / f"{symbol}_1m.parquet"
        
        if data_path.exists():
            df = pd.read_parquet(data_path)
            
            # Last 24 hours
            df = df.tail(1440)
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df["open_time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=symbol,
            ))
            fig.update_layout(
                title=f"{symbol} Price (Last 24h)",
                xaxis_title="Time",
                yaxis_title="Price (USDT)",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


def render_training():
    """Render training page."""
    st.title("📈 Training Monitor")
    
    logs = load_training_logs()
    
    if not logs:
        st.warning("No training logs found. Start training to see metrics.")
        
        st.code("""
# Quick training test
python scripts/train.py --dry-run --steps 1000

# Full training
python scripts/train.py --steps 100000
        """)
        return
    
    st.success(f"Found {len(logs)} metrics")
    
    # Reward plot
    if any("reward" in k.lower() for k in logs.keys()):
        st.subheader("Reward Over Time")
        
        for key, data in logs.items():
            if "reward" in key.lower():
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data["step"],
                    y=data["value"],
                    name=key,
                    mode="lines",
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Trading metrics
    st.subheader("Trading Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for key, data in logs.items():
            if "return" in key.lower():
                if not data.empty:
                    latest = data["value"].iloc[-1]
                    st.metric(key, f"{latest*100:.2f}%")
    
    with col2:
        for key, data in logs.items():
            if "drawdown" in key.lower():
                if not data.empty:
                    latest = data["value"].iloc[-1]
                    st.metric(key, f"{latest*100:.2f}%")


def render_backtest():
    """Render backtest page."""
    st.title("🔬 Backtest Analysis")
    
    # Backtest parameters
    col1, col2 = st.columns(2)
    
    with col1:
        models = load_model_info()
        if models:
            model = st.selectbox("Select Model", [m["name"] for m in models])
    
    with col2:
        data_files = load_data_files()
        if data_files:
            symbol = st.selectbox("Select Symbol", [f["symbol"] for f in data_files])
    
    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            st.info("Running backtest... (check terminal for output)")
            
            import subprocess
            result = subprocess.run(
                ["python", "scripts/backtest.py", "--symbol", symbol, "--no-plot"],
                capture_output=True,
                text=True,
                cwd=str(Path.cwd()),
            )
            
            if result.returncode == 0:
                st.success("Backtest completed!")
                st.code(result.stdout)
            else:
                st.error("Backtest failed")
                st.code(result.stderr)


def render_settings():
    """Render settings page."""
    st.title("⚙️ Configuration")
    
    config = load_config()
    
    if not config:
        st.warning("No configuration found.")
        return
    
    # Trading settings
    st.subheader("Trading Settings")
    col1, col2, col3 = st.columns(3)
    
    trading = config.get("trading", {})
    with col1:
        st.number_input("Initial Capital ($)", value=trading.get("initial_capital", 100), disabled=True)
    with col2:
        st.number_input("Leverage", value=trading.get("leverage", 2), disabled=True)
    with col3:
        st.number_input("Max Position", value=trading.get("max_position_per_asset", 0.2), disabled=True)
    
    # Fees
    st.subheader("Fees")
    fees = trading.get("fees", {})
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Maker Fee (%)", value=fees.get("maker", 0.02) * 100, disabled=True)
    with col2:
        st.number_input("Taker Fee (%)", value=fees.get("taker", 0.04) * 100, disabled=True)
    
    # Symbols
    st.subheader("Trading Symbols")
    symbols = config.get("symbols", [])
    st.write(", ".join(symbols))
    
    # PPO settings
    st.subheader("PPO Hyperparameters")
    ppo = config.get("ppo", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input("Learning Rate", value=ppo.get("learning_rate", 3e-4), format="%.2e", disabled=True)
    with col2:
        st.number_input("Batch Size", value=ppo.get("batch_size", 64), disabled=True)
    with col3:
        st.number_input("Gamma", value=ppo.get("gamma", 0.99), disabled=True)


def render_live():
    """Render live trading page."""
    st.title("🔴 Live Trading")
    
    st.warning("⚠️ Live trading involves real money. Use with extreme caution!")
    
    # Status
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Status")
        st.info("🟢 Ready (not connected)")
    
    with col2:
        st.subheader("Quick Actions")
        
        if st.button("Start Dry Run"):
            st.info("Starting dry run mode...")
            st.code("python -m src.live.executor --model models/ppo_trading_final --config config/config.yaml")
    
    st.markdown("---")
    
    # Instructions
    st.subheader("How to Start Live Trading")
    
    st.markdown("""
    1. **Train a model** first:
       ```bash
       python scripts/train.py --steps 100000
       ```
    
    2. **Backtest** to validate:
       ```bash
       python scripts/backtest.py --symbol BTCUSDT
       ```
    
    3. **Dry run** (no real trades):
       ```bash
       python -m src.live.executor --model models/ppo_trading_final
       ```
    
    4. **Live trading** (⚠️ real money):
       ```bash
       python -m src.live.executor --model models/ppo_trading_final --live
       ```
    """)


def main():
    """Main dashboard function."""
    page = render_sidebar()
    
    if page == "📊 Overview":
        render_overview()
    elif page == "📈 Training":
        render_training()
    elif page == "🔬 Backtest":
        render_backtest()
    elif page == "⚙️ Settings":
        render_settings()
    elif page == "🔴 Live":
        render_live()


if __name__ == "__main__":
    main()
