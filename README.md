# Employee Sentiment Analysis

## 📊 Project Overview
This project analyzes employee email sentiment data from the Enron dataset using Natural Language Processing (NLP) techniques.

## 🎯 Objectives
- Analyze email sentiment (Positive/Negative/Neutral)
- Calculate monthly sentiment scores for employees
- Identify flight risk employees based on negative sentiment patterns
- Generate visualizations and reports

## 📁 Project Structure
```
Employee_Sentiment_Analysis/
│
├── 📄 test.xlsx                          # Input data (Enron emails)
├── 🐍 sentiment_analysis.py              # Main analysis script
│
├── 📊 visualizations/                    # Generated charts
│   ├── sentiment_overview.png
│   ├── flight_risk_analysis.png
│   └── negative_wordcloud.png
│
└── 📂 outputs/                           # Analysis results
    ├── full_sentiment_analysis.csv
    ├── monthly_sentiment_scores.csv
    ├── flight_risk_employees.csv
    └── summary_statistics.csv
```

## 🔍 Key Findings

## 🔍 Key Findings
- **Total Emails Analyzed**: 2,191
- **Total Employees**: 10
- **Sentiment Distribution**:
  - Positive: 979 (44.7%)
  - Neutral: 1,016 (46.4%)
  - Negative: 196 (8.9%)
- **Flight Risk Employees**: 0 detected

## 🛠️ Technologies Used
- **Python 3.x**
- **Libraries**:
  - pandas - Data manipulation
  - TextBlob - Sentiment analysis
  - matplotlib & seaborn - Visualizations
  - wordcloud - Word cloud generation
  - openpyxl - Excel file handling

## 🚀 How to Run

### Prerequisites
```bashpip install pandas openpyxl textblob matplotlib seaborn wordcloud
python -m textblob.download_corpora

### Execution
```bashpython sentiment_analysis.py

## 📊 Methodology

### 1. Sentiment Analysis
- Uses TextBlob's polarity score (-1 to +1)
- Classification:
  - **Positive**: polarity > 0.1 → Score: +1
  - **Negative**: polarity < -0.1 → Score: -1
  - **Neutral**: -0.1 ≤ polarity ≤ 0.1 → Score: 0

### 2. Monthly Sentiment Scoring
- Aggregates sentiment scores by employee per month
- Formula: Monthly Score = Sum of all sentiment scores in that month

### 3. Flight Risk Detection
- Analyzes 30-day rolling windows
- **Flight Risk Criteria**: Total sentiment score ≤ -4 in any 30-day period
- Indicates potential employee dissatisfaction/turnover risk

## 📈 Visualizations

### 1. Sentiment Overview (sentiment_overview.png)
- Pie chart: Overall sentiment distribution
- Bar chart: Sentiment category counts
- Line chart: Average sentiment over time
- Horizontal bar: Top 10 employees with negative emails

### 2. Flight Risk Analysis (flight_risk_analysis.png)
- Identifies employees at risk of leaving
- Shows maximum negative scores in 30-day windows

### 3. Word Cloud (negative_wordcloud.png)
- Most common words in negative emails
- Helps identify recurring issues/concerns

## 📝 Output Files

### 1. full_sentiment_analysis.csv
Complete dataset with sentiment scores and categories for each email

### 2. monthly_sentiment_scores.csv
Employee-wise monthly aggregated sentiment scores

### 3. flight_risk_employees.csv
List of employees identified as flight risks (if any)

### 4. summary_statistics.csv
High-level statistics of the entire analysis

## 🎓 Business Insights
- Overall sentiment is predominantly neutral-to-positive (91.1%)
- Low negative sentiment (8.9%) indicates healthy workplace communication
- No flight risk employees detected - positive sign for retention
- Can be used for:
  - Early warning system for employee dissatisfaction
  - Department-wise sentiment tracking
  - Intervention planning for at-risk employees

## 📧 Contact
For questions or collaboration:
- Email: jbirch@glynac.ai

## 📜 License
This project is for educational/research purposes.

## 🙏 Acknowledgments
- Enron email dataset
- TextBlob NLP library
- Python data science community
