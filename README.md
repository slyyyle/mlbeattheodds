# MLB Betting Analytics Platform

A comprehensive, end-to-end MLB betting research application that combines advanced data engineering, machine learning, and modern web technologies to provide intelligent betting analysis and recommendations.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Data Pipeline   │───▶│   DuckDB        │
│                 │    │                  │    │   Warehouse     │
│ • pybaseball    │    │ • Ingestion      │    │                 │
│ • The Odds API  │    │ • Cleaning       │    │ • Games         │
│ • Sports data   │    │ • Validation     │    │ • Standings     │
└─────────────────┘    └──────────────────┘    │ • Odds          │
                                               │ • Props         │
                                               └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  React Frontend │◀───│   FastAPI        │◀───│ Feature Engine  │
│                 │    │   Backend        │    │                 │
│ • Dashboard     │    │                  │    │ • Moneyline     │
│ • Odds Compare  │    │ • REST API       │    │ • Strikeout     │
│ • Props         │    │ • Real-time      │    │ • Hits/TB       │
│ • Backtests     │    │ • Caching        │    │ Features        │
│ • Recommendations │  └──────────────────┘    └─────────────────┘
└─────────────────┘                                   │
                                                      ▼
                                            ┌─────────────────┐
                                            │  ML Backtesting │
                                            │                 │
                                            │ • Sports-betting│
                                            │ • Scikit-learn  │
                                            │ • Model eval    │
                                            │ • Performance   │
                                            └─────────────────┘
```

## 🚀 Features

### Advanced Machine Learning & AI
- **Ensemble Models**: Combines Logistic Regression, Random Forest, XGBoost, and Gradient Boosting
- **Automated Hyperparameter Tuning**: Grid search and Bayesian optimization
- **Model Versioning & Deployment**: Automated model persistence and A/B testing
- **Prediction Confidence**: Model agreement-based confidence scoring
- **Feature Importance Analysis**: SHAP values and permutation importance

### Comprehensive Data Pipeline
- **Multi-source ingestion**: pybaseball for historical data, The Odds API for live betting markets
- **Advanced feature engineering**: 200+ features including sabermetrics, park factors, weather impact
- **Real-time processing**: Live odds updates, automated data quality monitoring
- **Data Validation**: Great Expectations integration for data quality assurance
- **Automated Pipeline**: Scheduled data ingestion, feature engineering, and model retraining

### Sophisticated Betting Analytics
- **Moneyline modeling**: Team performance, head-to-head records, Pythagorean expectation
- **Comprehensive prop analysis**: 15+ prop types including NRFI/YRFI, player props, team totals
- **Market efficiency detection**: Vig analysis, line shopping, arbitrage opportunities
- **Advanced sabermetrics**: wOBA, wRC+, FIP, xFIP, SIERA integration
- **Situational analysis**: Clutch performance, platoon splits, weather adjustments

### Professional Risk Management
- **Kelly Criterion Sizing**: Optimal bet sizing with fractional Kelly for risk reduction
- **Portfolio Optimization**: Multi-bet optimization using mean-variance optimization
- **Dynamic Risk Controls**: Real-time risk exposure monitoring and drawdown protection
- **Bankroll Management**: Comprehensive P&L tracking with performance attribution
- **Risk Assessment**: Automated risk level classification and recommendations

### Production-Ready Infrastructure
- **Automated Pipeline**: Continuous data ingestion, feature engineering, and model updates
- **Performance Monitoring**: Real-time model performance tracking and drift detection
- **Alert System**: Multi-channel notifications (Email, Slack, Webhook, SMS)
- **System Health Monitoring**: Database, API, and resource usage monitoring
- **Comprehensive Logging**: Structured logging with performance metrics

### Advanced User Interface
- **Modern React dashboard**: Real-time data visualization with Chart.js and Recharts
- **AI Recommendations**: Confidence-scored betting recommendations with reasoning
- **Interactive analysis**: Dynamic filtering, drill-down capabilities, responsive design
- **Risk Dashboard**: Real-time risk metrics and portfolio performance
- **Alert Management**: Centralized alert viewing and management
- **AI recommendations**: Confidence-scored betting suggestions with detailed reasoning

## 📋 Prerequisites

- **Python 3.8+**
- **Node.js 16+** (for frontend)
- **The Odds API key** (optional, for live odds)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd mlbetting
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix/macOS:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
# Copy environment template
cp config/env_template.txt .env

# Edit .env file with your configuration
# Most importantly, add your ODDS_API_KEY if you have one
```

### 4. Set Up Frontend (Optional)
```bash
cd ui
npm install
cd ..
```

## 🎯 Quick Start

### Option 1: Full Pipeline (Recommended)
```bash
# Run complete pipeline with frontend
python scripts/run_pipeline.py --full --with-frontend

# This will:
# 1. Ingest MLB data
# 2. Initialize DuckDB warehouse
# 3. Build features
# 4. Run backtests
# 5. Start API server (localhost:8000)
# 6. Start React frontend (localhost:3000)
```

### Option 2: API Server Only
```bash
# Start just the API server
python scripts/run_pipeline.py --serve-only

# Access API documentation at http://localhost:8000/docs
```

### Option 3: Manual Step-by-Step
```bash
# 1. Ingest data
python src/ingest/ingest_games.py
python src/ingest/ingest_odds.py

# 2. Initialize database
python src/storage/init_db.py

# 3. Build features
python src/features/build_features.py

# 4. Run backtests
python src/backtest/run_backtests.py

# 5. Start API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

## 📊 Data Sources

### Primary Data (pybaseball)
- **Game results**: Scores, dates, teams, win/loss records
- **Team standings**: Wins, losses, win percentage, run differential
- **Player statistics**: Batting and pitching stats (when available)

### Odds Data (The Odds API)
- **Moneyline odds**: Home/away odds across multiple bookmakers
- **Player props**: Strikeout, hits, and total bases betting lines
- **Market data**: Implied probabilities, vig calculations, line movement

## 🧮 Feature Engineering

### Moneyline Features
```python
# Team performance metrics
- win_pct_diff: Home vs away win percentage differential
- recent_form: Rolling 10-game win rate vs season average
- run_differential: Offensive and defensive strength indicators
- home_field_advantage: Historical home team performance

# Market efficiency features
- vig_analysis: Bookmaker margin calculation
- line_shopping: Best available odds identification
- market_consensus: Aggregate market sentiment
```

### Prop Bet Features
```python
# Player-specific metrics
- pitcher_k_rate: Strikeouts per 9 innings
- batter_contact_rate: Historical contact vs specific pitcher types
- park_factors: Venue-specific adjustments
- recent_performance: Last 10 games trending

# Situational analysis
- weather_impact: Temperature, wind, humidity effects
- umpire_tendencies: Strike zone impact on props
- bullpen_usage: Late-game pitching changes
```

## 🏃‍♂️ Backtesting Framework

### Strategy Implementation
```python
# Example: Moneyline value betting
from src.backtest.run_backtests import MLBBacktester

backtester = MLBBacktester()
results = backtester.backtest_moneyline()

# Key metrics returned:
# - ROI (Return on Investment)
# - Win Rate
# - Sharpe Ratio
# - Maximum Drawdown
# - Total Bets Placed
```

### Model Evaluation
- **Time series cross-validation**: Prevents data leakage
- **Sports-betting integration**: Realistic transaction costs and constraints
- **Risk-adjusted returns**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Bankroll simulation**: Kelly criterion optimal sizing

## 🌐 API Endpoints

### Core Endpoints
```bash
# Get backtest metrics
GET /metrics/{strategy}

# Current odds comparison
GET /odds?days_ahead=3

# Team performance data
GET /teams?season=2024

# Player prop bets
GET /props?prop_type=strikeouts&days_ahead=3

# AI recommendations
GET /recommendations?bet_type=all&min_edge=0.02
```

### Advanced Analytics
```bash
# Best odds across bookmakers
GET /analytics/best-odds

# Team trend analysis
GET /analytics/team-trends/{team}?games=20

# Market efficiency metrics
GET /analytics/market-efficiency
```

## 🎨 Frontend Features

### Dashboard Components
- **Performance Overview**: Key metrics, recent results, system status
- **Interactive Charts**: ROI trends, win rate analysis, risk metrics
- **Real-time Updates**: Live odds feeds, recommendation alerts

### Specialized Views
- **Odds Comparison**: Multi-bookmaker odds with vig analysis
- **Props Analysis**: Player prop bets with edge detection
- **Team Analytics**: Performance trends, head-to-head records
- **Backtest Results**: Historical strategy performance with detailed breakdowns

## 🧪 Testing Strategy

### Unit Tests
```bash
# Run Python tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_features.py -v
```

### Integration Tests
```bash
# Test full pipeline
python scripts/run_pipeline.py --test-mode

# API endpoint testing
python -m pytest tests/test_api.py -v
```

## 📈 Performance Optimization

### Database Performance
- **Indexed queries**: Strategic indexing on date, team, and player columns
- **Columnar storage**: Parquet files for analytical workloads
- **Query optimization**: Efficient joins and aggregations

### API Performance
- **Response caching**: TTL-based caching for frequently accessed data
- **Async processing**: Non-blocking I/O for concurrent requests
- **Rate limiting**: Protection against API abuse

### Frontend Performance
- **React Query**: Intelligent data fetching and caching
- **Component optimization**: Memoization and lazy loading
- **Bundle optimization**: Code splitting and compression

## 🔒 Security Considerations

### API Security
- **CORS configuration**: Controlled cross-origin requests
- **Input validation**: Pydantic models for request validation
- **Rate limiting**: Protection against excessive requests

### Data Security
- **Environment variables**: Sensitive configuration management
- **API key protection**: Secure credential storage
- **Local data**: No sensitive data transmission to external services

## 🐛 Troubleshooting

### Common Issues

**1. Missing Dependencies**
```bash
# Install missing packages
pip install -r requirements.txt

# For React issues
cd ui && npm install
```

**2. Database Connection Issues**
```bash
# Reset database
python src/storage/init_db.py --force-recreate
```

**3. API Key Issues**
```bash
# Check .env file
cat .env | grep ODDS_API_KEY

# Test API connection
python src/ingest/ingest_odds.py
```

**4. Port Conflicts**
```bash
# Check if ports are in use
lsof -i :8000  # API server
lsof -i :3000  # React frontend

# Use different ports
uvicorn src.api.app:app --port 8001
```

## 🚀 Deployment

### Production Deployment
```bash
# Build React app
cd ui && npm run build

# Start production API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4

# Use reverse proxy (nginx/apache) for static files
```

### Docker Deployment (Optional)
```dockerfile
# Example Dockerfile for API
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Code Standards
- **Python**: PEP 8 compliance, type hints encouraged
- **JavaScript**: ESLint configuration, Prettier formatting
- **Testing**: Comprehensive test coverage for new features

## 📚 Additional Resources

### Documentation
- [The Odds API Documentation](https://the-odds-api.com/liveapi/guides/v4/)
- [pybaseball Documentation](https://github.com/jldbc/pybaseball)
- [Sports-betting Package](https://github.com/AlgoWit/sports-betting)

### Learning Resources
- Sports betting mathematics and Kelly criterion
- Machine learning for sports analytics
- Time series analysis for betting strategies

## ⚠️ Disclaimer

This software is for educational and research purposes only. Sports betting involves risk and may not be legal in all jurisdictions. Always:

- Gamble responsibly
- Never bet more than you can afford to lose
- Understand that past performance doesn't guarantee future results
- Check local laws regarding sports betting
- Seek help if gambling becomes a problem

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **pybaseball** community for excellent MLB data access
- **The Odds API** for comprehensive betting market data
- **Sports-betting** package authors for backtesting framework
- **FastAPI** and **React** communities for amazing frameworks 