#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Employee Sentiment Analysis - Direct Python Script
No Jupyter needed! Just run: python sentiment_analysis.py
"""

import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("   EMPLOYEE SENTIMENT ANALYSIS")
print("="*60)

# Import Libraries
print("\n[1/9] Importing libraries...")
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # For non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from textblob import TextBlob
    from datetime import datetime, timedelta
    from collections import Counter
    import re
    from wordcloud import WordCloud
    import os
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    print("✅ All libraries imported successfully!")
except Exception as e:
    print(f"❌ Error importing libraries: {e}")
    print("\nRun this command:")
    print("pip install pandas openpyxl textblob matplotlib seaborn wordcloud")
    exit(1)

# Load Excel Data
print("\n[2/9] Loading Excel file...")
try:
    df = pd.read_excel('test.xlsx')
    print(f"✅ Data loaded successfully!")
    print(f"📊 Total emails: {len(df)}")
    print(f"📋 Columns: {list(df.columns)}")
except Exception as e:
    print(f"❌ Error loading Excel: {e}")
    print("\nMake sure 'test.xlsx' is in the same folder!")
    exit(1)

# Data Cleaning
print("\n[3/9] Cleaning data...")
try:
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Check required columns
    required_cols = ['subject', 'body', 'date', 'from']
    
    # Auto-detect column names
    for col in df.columns:
        if 'subject' in col.lower():
            df.rename(columns={col: 'subject'}, inplace=True)
        elif 'body' in col.lower() or 'message' in col.lower():
            df.rename(columns={col: 'body'}, inplace=True)
        elif 'date' in col.lower():
            df.rename(columns={col: 'date'}, inplace=True)
        elif 'from' in col.lower() or 'sender' in col.lower():
            df.rename(columns={col: 'from'}, inplace=True)
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Fill NaN
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    df['combined_text'] = df['subject'].astype(str) + ' ' + df['body'].astype(str)
    
    print(f"✅ Data cleaned! Valid emails: {len(df)}")
except Exception as e:
    print(f"❌ Error cleaning data: {e}")
    exit(1)

# Sentiment Analysis
print("\n[4/9] Analyzing sentiments...")
def analyze_sentiment(text):
    """Analyze sentiment of text"""
    if not text or pd.isna(text):
        return 0, 'Neutral'
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return 1, 'Positive'
        elif polarity < -0.1:
            return -1, 'Negative'
        else:
            return 0, 'Neutral'
    except:
        return 0, 'Neutral'

try:
    results = df['combined_text'].apply(analyze_sentiment)
    df['sentiment_score'] = results.apply(lambda x: x[0])
    df['sentiment_category'] = results.apply(lambda x: x[1])
    
    print("✅ Sentiment analysis completed!")
    print("\n📊 Sentiment Distribution:")
    print(df['sentiment_category'].value_counts())
except Exception as e:
    print(f"❌ Error in sentiment analysis: {e}")
    exit(1)

# Monthly Scores
print("\n[5/9] Calculating monthly sentiment scores...")
try:
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_scores = df.groupby(['from', 'year_month'])['sentiment_score'].sum().reset_index()
    monthly_scores.columns = ['employee', 'month', 'monthly_score']
    print(f"✅ Monthly scores calculated for {monthly_scores['employee'].nunique()} employees")
except Exception as e:
    print(f"❌ Error calculating monthly scores: {e}")

# Flight Risk Detection
print("\n[6/9] Detecting flight risk employees...")
def detect_flight_risk(employee_df, window_days=30, threshold=-4):
    """Detect flight risk based on 30-day window"""
    flight_risk_periods = []
    for i in range(len(employee_df)):
        current_date = employee_df.iloc[i]['date']
        window_start = current_date - timedelta(days=window_days)
        window_emails = employee_df[
            (employee_df['date'] >= window_start) & 
            (employee_df['date'] <= current_date)
        ]
        total_score = window_emails['sentiment_score'].sum()
        if total_score <= threshold:
            flight_risk_periods.append({
                'date': current_date,
                'score': total_score,
                'negative_count': (window_emails['sentiment_score'] == -1).sum()
            })
    return flight_risk_periods

try:
    flight_risk_data = []
    for employee in df['from'].unique():
        emp_df = df[df['from'] == employee].sort_values('date')
        risk_periods = detect_flight_risk(emp_df)
        if risk_periods:
            flight_risk_data.append({
                'employee': employee,
                'risk_periods': len(risk_periods),
                'max_negative_score': min([p['score'] for p in risk_periods]),
                'first_risk_date': min([p['date'] for p in risk_periods])
            })
    
    flight_risk_df = pd.DataFrame(flight_risk_data)
    if len(flight_risk_df) > 0:
        print(f"🚨 Flight Risk Employees Found: {len(flight_risk_df)}")
    else:
        print("✅ No flight risk employees detected")
except Exception as e:
    print(f"⚠️ Warning in flight risk detection: {e}")
    flight_risk_df = pd.DataFrame()

# Create Visualizations
print("\n[7/9] Creating visualizations...")
try:
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Overall Sentiment
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sentiment_counts = df['sentiment_category'].value_counts()
    axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')
    
    sentiment_counts.plot(kind='bar', ax=axes[0, 1], color=['green', 'red', 'gray'])
    axes[0, 1].set_title('Sentiment Category Counts', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Sentiment')
    axes[0, 1].set_ylabel('Count')
    
    monthly_sentiment = df.groupby(df['date'].dt.to_period('M'))['sentiment_score'].mean()
    monthly_sentiment.plot(ax=axes[1, 0], marker='o', color='blue')
    axes[1, 0].set_title('Average Sentiment Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Average Sentiment')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    top_negative = df[df['sentiment_score'] == -1].groupby('from').size().nlargest(10)
    top_negative.plot(kind='barh', ax=axes[1, 1], color='red')
    axes[1, 1].set_title('Top 10 Employees with Most Negative Emails', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Negative Email Count')
    
    plt.tight_layout()
    plt.savefig('visualizations/sentiment_overview.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: sentiment_overview.png")
    plt.close()
    
    # 2. Flight Risk
    if len(flight_risk_df) > 0:
        plt.figure(figsize=(12, 6))
        flight_risk_sorted = flight_risk_df.sort_values('max_negative_score')
        plt.barh(flight_risk_sorted['employee'], flight_risk_sorted['max_negative_score'], color='darkred')
        plt.xlabel('Maximum Negative Score (30-day window)', fontsize=12)
        plt.ylabel('Employee Email', fontsize=12)
        plt.title('Flight Risk Employees - Maximum Negative Scores', fontsize=14, fontweight='bold')
        plt.axvline(x=-4, color='orange', linestyle='--', label='Risk Threshold (-4)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/flight_risk_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: flight_risk_analysis.png")
        plt.close()
    
    # 3. Word Cloud
    negative_text = ' '.join(df[df['sentiment_score'] == -1]['combined_text'].astype(str))
    if negative_text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Negative Emails', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/negative_wordcloud.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: negative_wordcloud.png")
        plt.close()
    
    print("\n✅ All visualizations created in 'visualizations/' folder!")
except Exception as e:
    print(f"⚠️ Warning in visualization: {e}")

# Save Results
print("\n[8/9] Saving results...")
try:
    os.makedirs('outputs', exist_ok=True)
    
    df.to_csv('outputs/full_sentiment_analysis.csv', index=False)
    print("✅ Saved: full_sentiment_analysis.csv")
    
    monthly_scores.to_csv('outputs/monthly_sentiment_scores.csv', index=False)
    print("✅ Saved: monthly_sentiment_scores.csv")
    
    if len(flight_risk_df) > 0:
        flight_risk_df.to_csv('outputs/flight_risk_employees.csv', index=False)
        print("✅ Saved: flight_risk_employees.csv")
    
    summary = {
        'Total Emails': len(df),
        'Total Employees': df['from'].nunique(),
        'Positive Emails': len(df[df['sentiment_score'] == 1]),
        'Negative Emails': len(df[df['sentiment_score'] == -1]),
        'Neutral Emails': len(df[df['sentiment_score'] == 0]),
        'Flight Risk Employees': len(flight_risk_df) if len(flight_risk_df) > 0 else 0,
        'Analysis Date Range': f"{df['date'].min()} to {df['date'].max()}"
    }
    
    summary_df = pd.DataFrame([summary]).T
    summary_df.columns = ['Value']
    summary_df.to_csv('outputs/summary_statistics.csv')
    print("✅ Saved: summary_statistics.csv")
except Exception as e:
    print(f"⚠️ Warning saving results: {e}")

# Final Summary
print("\n" + "="*60)
print("   🎉 ANALYSIS COMPLETE!")
print("="*60)
print("\n📁 Results saved in:")
print("   ├── visualizations/")
print("   │   ├── sentiment_overview.png")
print("   │   ├── flight_risk_analysis.png")
print("   │   └── negative_wordcloud.png")
print("   └── outputs/")
print("       ├── full_sentiment_analysis.csv")
print("       ├── monthly_sentiment_scores.csv")
print("       ├── flight_risk_employees.csv")
print("       └── summary_statistics.csv")
print("\n✅ Ready for submission!")
print("="*60)