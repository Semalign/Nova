📈 Stock Sentiment Correlation Dashboard

This project investigates the relationship between financial news sentiment and stock price movements for top tech companies. The analysis includes exploratory data analysis, technical indicators, sentiment scoring, correlation studies, and an interactive dashboard (ongoing).

🔍 Project Structure
├── data/
│   ├── raw_analyst_ratings.csv       # News data with analyst ratings
│   └── yfinance_data/                # Stock price data for each company
├── notebooks/
│   ├── task1_eda.ipynb               # EDA and publisher analysis
│   ├── task2_technical_analysis.ipynb# Technical indicators & trends
│   └── task3_sentiment_correlation.ipynb  # Sentiment vs stock return correlation
├── scripts/
│   └── sentiment_correlation.py      # Automated script for Task 3
├── README.md
├── requirements.txt
└── .gitignore

✅ Tasks Overview
Task 1: Clean & explore news data.
Task 2: Compute stock indicators (SMA, EMA, RSI).
Task 3: Run sentiment analysis, align dates, compute daily returns, and test correlation.

📊 Tools & Libraries
- pandas, numpy
- matplotlib, seaborn
- textblob (sentiment)
- pynance (indicators)
- scipy.stats (correlation)
- jupyterlab

⚙️ Setup
```bash
git clone https://github.com/Semalign/Nova.git
cd stock-news-sentiment
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
jupyter lab  # or python scripts/sentiment_correlation.py
