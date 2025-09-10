# Forex-Quantitative-Trading-Strategy-Development
**A Machine Learning Approach to Low-Volatility FX Pairs**

**Author:** Aguru Venkata Saisantosh Patnaik  
**Contact:** [agurusantosh@gmail.com](mailto:agurusantosh@gmail.com)

## 🎯 Project Overview

This repository presents a comprehensive research pipeline for developing and validating quantitative trading strategies on low-volatility foreign exchange pairs using hourly OHLCV data. The project demonstrates how advanced feature engineering, machine learning, and rigorous backtesting can be combined to create profitable trading strategies while maintaining strict adherence to no-lookahead principles.

### Key Achievements
- **+87% returns** with realistic transaction costs (0.01% commission)
- **Sharpe ratio of 1.69** with maximum drawdown of 7.3%
- Robust validation using purged time-series cross-validation
- Comprehensive feature engineering incorporating market microstructure

## 🏗️ Repository Structure
.
├── full_developed_strategy.ipynb   # Main notebook: data processing, feature engineering, modeling, backtest
├── saisantosh_report.pdf           # Written project report (figures & analysis)
├── data.csv                        # Primary hourly OHLCV dataset
├── gold.csv                        # Daily gold close used as macro/commodity feature
├── oil.csv                         # Daily WTI crude close & volume used as macro/commodity feature
├── requirements.txt                # Python package requirements
└── README.md


## 🔬 Research Methodology

### 1. Feature Engineering Strategy

The project employs a multi-layered approach to feature creation, combining domain knowledge with data-driven insights:

#### **Temporal & Session Features**
- **Trading Session Labeling**: Asian, European, and American sessions with overlap periods
- **Cyclical Encoding**: Sine/cosine transformations for hours, days, months to preserve circular relationships
- **Seasonal Indicators**: Quarter, month-end, and day-of-week effects

#### **Macroeconomic Integration**  
- **Interest Rate Differentials**: US 10Y-2Y Treasury spread as currency strength indicator
- **Commodity Integration**: Daily gold and WTI crude oil prices merged to hourly frequency
- **Volume Analysis**: Oil trading volume as market activity proxy

#### **Technical Indicators Suite**
The project implements bounded technical indicators across three categories:

**Trend Indicators:**
- Average Directional Index (ADX) - multiple timeframes (2, 5, 14 periods)
- Commodity Channel Index (CCI) - various lookback periods (2, 8, 20 periods)

**Momentum Indicators:**
- Percentage Price Oscillator (PPO) - fast/slow combinations
- Chande Momentum Oscillator (CMO) - short and long-term versions

**Volatility Indicators:**
- Chaikin Volatility - intrabar range analysis
- Keltner Channel Width - interbar volatility measurement

#### **Derived Features**
- **Cross-timeframe differences**: Capturing momentum shifts
- **Ratio analysis**: Relative strength between different periods
- **Autocorrelation-guided selection**: Using ACF/PACF analysis for optimal lookback periods

### 2. Innovative Target Signal Creation

Rather than using traditional price-based targets, this project implements a **hindsight-adjusted signal approach**:

1. **Base Strategy Generation**: Rule-based EMA/SMA crossover system with take-profit/stop-loss
2. **Trade Labeling**: Each completed trade classified as Profit (>0.1%), Loss (<-0.1%), or Neutral
3. **Signal Adjustment**: 
   - Profitable trades: Keep original signal
   - Loss-making trades: Reverse the signal
   - Neutral trades: Set signal to 0 (no position)

This approach enables the ML model to learn from "ideal" hindsight decisions while maintaining realistic market conditions.

### 3. Robust Model Development

#### **Feature Selection Process**
- **Random Forest Importance**: Permutation-based feature ranking across time-series CV folds
- **Domain Knowledge Filtering**: Retention of economically meaningful indicators
- **Correlation Analysis**: Removal of redundant features to prevent overfitting

#### **XGBoost Implementation**
- **Hyperparameter Optimization**: 30-iteration randomized search with time-series CV
- **Class Balancing**: Weighted training to handle imbalanced signal classes
- **Purged Cross-Validation**: 5-fold time-series CV with gap periods to prevent leakage
- **Early Stopping**: Prevents overfitting with 150-round patience

### 4. Realistic Backtesting Framework

#### **Execution Assumptions**
- **Trade Timing**: Execution at bar close prices
- **Commission Structure**: Realistic FX broker fees (0.003-0.007% per side)
- **No Slippage**: Conservative assumption for liquid FX pairs
- **Position Management**: Exclusive long/short positions (no hedging)

#### **Performance Validation**
- **Out-of-sample testing**: 30% holdout period with no parameter adjustment
- **Transaction Cost Sensitivity**: Analysis across multiple commission scenarios
- **Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown analysis

## 📊 Key Results

### Performance Metrics (0.01% Commission)
- **Total Return**: +87%
- **Sharpe Ratio**: 1.69
- **Maximum Drawdown**: 7.3%
- **Win Rate**: Competitive risk-adjusted returns
- **Calmar Ratio**: Strong risk-adjusted performance

### Feature Importance Rankings
Top contributing features identified through permutation importance:
1. **Original Signal**: Base strategy signal
2. **CMO_14**: 14-period Chande Momentum Oscillator  
3. **ADX_5**: 5-period Average Directional Index
4. **PPO_2_6**: 2/6-period Percentage Price Oscillator
5. **CCI_ratio_2_20**: Cross-timeframe CCI ratio

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Lab/Notebook environment
- Minimum 8GB RAM for full dataset processing

### Data Requirements
Ensure the `data/` directory contains:
- `data.csv`: Hourly OHLCV forex data
- `gold.csv`: Daily gold close prices
- `oil.csv`: Daily WTI crude oil data with volume
- `interest_rates.csv`: US Treasury yield curve data

## 🔧 Configuration Options

### Strategy Parameters
#### Base strategy settings
- EMA_WIN = 500    # EMA lookback period
- SMA_WIN = 2000   # SMA lookback period
- TP_PCT = 0.005   # Take profit threshold (0.5%)
- MAX_HOLD = 240   # Maximum position holding period

#### Model settings
- N_ITER = 30      # Hyperparameter search iterations
- N_SPLITS = 5     # Time-series CV folds
- GAP_BARS = 5     # Purge gap between train/validation

  
## 🔮 Future Enhancements

### Model Improvements
- **Ensemble Methods**: Combine multiple model types (XGBoost, LSTM, Random Forest)
- **Alternative Targets**: Explore different labeling schemes and prediction horizons
- **Feature Engineering**: Add order book imbalance, news sentiment, and volatility surfaces

### Execution Optimization
- **Market Microstructure**: Incorporate bid-ask spreads and order book dynamics
- **Optimal Execution**: Implement VWAP, TWAP, and other execution algorithms
- **Multi-Asset**: Extend to currency baskets and cross-asset signals

### Risk Management
- **Dynamic Position Sizing**: Kelly criterion and risk parity approaches
- **Regime Detection**: Adapt strategy parameters to market conditions
- **Stress Testing**: Monte Carlo simulation and scenario analysis


