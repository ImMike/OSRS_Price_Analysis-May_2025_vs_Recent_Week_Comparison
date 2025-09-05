# 🏰 OSRS Price Analysis - May 2025 vs Recent Week Comparison

A sophisticated Python application that analyzes Old School RuneScape (OSRS) item prices to identify profitable trading opportunities by comparing **May 2025 average prices** with **recent week averages**, incorporating advanced economic viability metrics.

## 🎯 Key Features

### 📊 **Advanced Price Analysis**
- **May 2025 Baseline**: Uses May 1-31, 2025 average prices as historical reference
- **Recent Week Comparison**: Compares against last 7 days average prices
- **Smart Fallbacks**: Multiple calculation methods for comprehensive coverage
- **Quality Filtering**: Identifies genuine market declines vs. artificial outliers

### 💰 **Economic Viability Assessment**
- **Liquidity Analysis**: Trading volume, bid-ask spreads, market depth
- **Slippage Calculations**: Estimates market impact and execution costs
- **Risk Assessment**: Price stability, trend consistency, outlier detection
- **Investment Scoring**: Composite opportunity scores and recommendations

### 🎨 **Interactive Visualizations**
- **Sortable Data Tables**: Professional DataTables with numeric sorting
- **Interactive Charts**: Plotly-powered scatter plots and distributions
- **Comprehensive Dashboard**: Multi-chart overview with embedded analytics
- **Export Options**: CSV, HTML, and interactive table formats

### ⚡ **Performance & Reliability**
- **Intelligent Caching**: SQLite database with smart cache management
- **Rate Limiting**: Respectful API usage with concurrent request handling
- **Progress Tracking**: Verbose console output with batch processing
- **Error Handling**: Robust error recovery and detailed logging

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/OSRS_Price_Analysis-May_2025_vs_Recent_Week_Comparison.git
   cd OSRS_Price_Analysis-May_2025_vs_Recent_Week_Comparison
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**:
   ```bash
   python3 run_analysis.py
   ```

### First Run
The application will automatically:
- Create necessary directories (`data/`, `data/exports/`)
- Initialize SQLite cache database
- Fetch item mappings and current prices
- Generate comprehensive analysis reports

## 📈 How It Works

### 🔍 **Data Collection Process**
1. **Item Mapping**: Fetches all OSRS items from Wiki API
2. **Current Prices**: Gets latest high/low prices for all tradeable items
3. **Historical Data**: Retrieves daily price history in optimized batches
4. **Quality Assessment**: Analyzes price stability and trend consistency

### 🧮 **Analysis Methodology**

#### **Primary Method: May 2025 vs Recent Week**
```
Decline % = ((May_2025_Avg - Recent_Week_Avg) / May_2025_Avg) × 100
```

- **May 2025 Average**: Mean price from May 1-31, 2025 data points
- **Recent Week Average**: Mean price from last 7 days of available data
- **Minimum Data Points**: Requires ≥3 data points for each period

#### **Fallback Methods**
- **Historical Legacy**: Uses 365-day historical comparison
- **Current High/Low**: Uses current session high vs low price

### 💡 **Economic Viability Metrics**

#### **Liquidity Scoring** (0-100)
- Average daily/weekly trading volume
- Bid-ask spread percentage
- Volume volatility assessment
- Market cap estimation

#### **Slippage Estimation**
- Market impact percentage calculation
- Time-to-execute estimation
- Total slippage cost projection
- Net profit after execution costs

#### **Quality Assessment**
- Price volatility analysis (coefficient of variation)
- Outlier detection using IQR method
- Trend consistency scoring
- Composite stability rating

#### **Investment Recommendations**
- **Strong Buy**: High decline, excellent liquidity, stable market
- **Buy**: Good decline, adequate liquidity, acceptable stability
- **Consider**: Moderate opportunity with some risks
- **Avoid**: Poor liquidity, unstable market, or artificial decline

## 📊 Output Files

The analysis generates several files in `data/exports/`:

### **Primary Outputs**
- **`sortable_table_YYYYMMDD_HHMMSS.html`** - Interactive sortable data table
- **`osrs_dashboard_YYYYMMDD_HHMMSS.html`** - Comprehensive dashboard
- **`osrs_price_decline_YYYYMMDD_HHMMSS.csv`** - Complete dataset

### **Chart Components**
- **Decline Distribution** - Histogram of all price declines
- **Liquidity vs Opportunity** - Scatter plot analysis
- **Quality Assessment** - Market stability visualization
- **Method Breakdown** - Calculation method distribution

## 🎛️ Configuration

Edit `config/settings.py` to customize:

```python
# Trading thresholds
DEFAULT_TARGET_VOLUME = 1000      # Volume for slippage calculations
MIN_DAILY_VOLUME = 100           # Minimum daily volume threshold
MAX_SPREAD_PERCENTAGE = 10.0     # Maximum acceptable spread

# Analysis parameters
HISTORICAL_DAYS = 365            # Historical data range
MIN_PRICE_THRESHOLD = 1          # Minimum price to analyze
TOP_DECLINING_ITEMS = 50         # Items to highlight

# Performance settings
REQUEST_DELAY = 1.5              # API request delay
MAX_CONCURRENT_REQUESTS = 5      # Concurrent request limit
```

## 🗂️ Project Structure

```
OSRS_Price_Analysis-May_2025_vs_Recent_Week_Comparison/
├── 📁 src/
│   ├── 🐍 api_client.py         # API interaction & rate limiting
│   ├── 🐍 data_manager.py       # Data fetching & SQLite caching  
│   ├── 🐍 calculator.py         # Price analysis & economic metrics
│   ├── 🐍 visualizer.py         # Charts & interactive tables
│   └── 🐍 main.py              # Main application orchestrator
├── 📁 config/
│   └── ⚙️ settings.py          # Configuration parameters
├── 📄 requirements.txt          # Python dependencies
├── 🚀 run_analysis.py          # Entry point script
├── 📖 README.md               # This documentation
└── 🚫 .gitignore              # Git ignore rules
```

## 🔧 Advanced Usage

### **Command Line Options**
```bash
# Include all items (bypass quality filters)
python3 run_analysis.py --include-all

# Force refresh all cached data
python3 run_analysis.py --force-refresh

# Limit analysis scope (for testing)
python3 run_analysis.py --max-items 100

# Disable console output
python3 run_analysis.py --no-console
```

### **Interactive Table Features**
- **Numeric Sorting**: Click column headers for proper numerical sorting
- **Quick Sort Buttons**: One-click sorting by key metrics
- **Search & Filter**: Real-time filtering across all columns
- **Pagination**: Configurable items per page (25/50/100/All)
- **Responsive Design**: Mobile-friendly table layout

### **Data Export Options**
- **CSV**: Complete dataset with all calculated metrics
- **HTML Tables**: Interactive sortable tables with styling
- **Dashboard**: Multi-chart comprehensive analysis view
- **Individual Charts**: Separate HTML files for each visualization

## 🎯 Use Cases

### **Trading Opportunities**
- Identify items with significant May 2025 → Recent declines
- Assess liquidity and execution feasibility
- Calculate expected slippage and net profits
- Filter by investment recommendation levels

### **Market Analysis**
- Analyze price stability and trend consistency
- Compare different calculation methodologies
- Identify artificial vs. genuine market declines
- Track volume patterns and market depth

### **Risk Assessment**
- Evaluate price volatility and outlier frequency
- Assess bid-ask spreads and execution risks
- Compare opportunity scores across items
- Filter by quality and stability metrics

## 🛠️ Troubleshooting

### **Common Issues**

**Import Errors**
```bash
pip install -r requirements.txt
```

**API Timeouts**
- Increase `TIMEOUT` in `config/settings.py`
- Reduce `MAX_CONCURRENT_REQUESTS` for slower connections

**Memory Issues**
- Use `--max-items` parameter to limit scope
- Clear `data/cache.db` to reset cache

**Missing Data**
- Some items may lack May 2025 data (uses fallback methods)
- Check `price_tracker.log` for detailed error information

### **Performance Optimization**
- **Cache Management**: Historical data cached permanently
- **Batch Processing**: Items processed in optimized batches
- **Rate Limiting**: Configurable delays prevent API restrictions
- **Progress Tracking**: Real-time console feedback

## 📋 Data Sources

- **OSRS Wiki Prices API**: Real-time and historical price data
- **Item Mapping**: Complete OSRS item database
- **Trading Data**: Volume, bid/ask spreads, market depth

## ⚠️ Important Notes

### **Data Accuracy**
- Prices sourced from OSRS Wiki (community-maintained)
- May not reflect exact in-game prices at all times
- Historical data availability varies by item

### **Trading Disclaimer**
- **Educational Purpose Only**: This tool is for analysis and learning
- **Verify In-Game**: Always confirm prices before trading
- **Market Risks**: Past performance doesn't guarantee future results
- **No Financial Advice**: Use analysis at your own discretion

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **OSRS Wiki**: For providing comprehensive price data API
- **RuneScape Community**: For maintaining accurate price information
- **Python Libraries**: pandas, plotly, aiohttp, and other excellent tools

---

**Happy Trading!** 🎯📈💰