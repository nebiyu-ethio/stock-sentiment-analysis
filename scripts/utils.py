import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
import  matplotlib.dates as mdates
import  matplotlib.dates as mdates
#import TextBlob
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation





def plot_top_publishers(df):
    # Get the top 10 publishers
    top_publishers = df['publisher'].value_counts().nlargest(10)

    top_publishers.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Publishers by Article Count')
    plt.xlabel('Publishers')
    plt.ylabel('Number of Articles')
    
    # Display the plot
    plt.xticks(rotation=45, ha='right')
    plt.show()

def publication_dates(df):
    # Ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'], format='ISO8601')

    # Group by date and count articles
    daily_counts = df.groupby(df['date'].dt.date).size()
    
    # Find days with highest article counts
    top_days = daily_counts.nlargest(5)
    
    # Analyze weekday distribution
    weekday_counts = df['date'].dt.day_name().value_counts()
    
    # # Monthly trend
    # df['month_start'] = df['date'].dt.floor('D') + MonthEnd(0) - MonthEnd(1)
    # #monthly_counts = df.groupby('month_start').size()
    # monthly_counts = df.groupby(pd.Grouper(key='date', freq='M')).size()

    # Monthly trend
    df['month_start'] = df['date'].dt.floor('D') + MonthEnd(0) - MonthEnd(1)
    #monthly_counts = df.groupby('month_start').size()
    monthly_counts = df.groupby(df['date'].dt.to_period('M').dt.to_timestamp()).size()

    
    return {
        'daily_counts': daily_counts,
        'top_days': top_days,
        'weekday_counts': weekday_counts,
        'monthly_counts': monthly_counts
    }


# Plot the publication trends
def plot_publication_trends(date_analysis):

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Daily trend
    date_analysis['daily_counts'].plot(ax=axes[0, 0])
    axes[0, 0].set_title('Daily Article Count')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Number of Articles')
    
    # Top days
    date_analysis['top_days'].plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Top 5 Days with Most Articles')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Number of Articles')
    
    # Weekday distribution
    date_analysis['weekday_counts'].plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Article Distribution by Weekday')
    axes[1, 0].set_xlabel('Weekday')
    axes[1, 0].set_ylabel('Number of Articles')
    
    # Monthly trend
    monthly_counts = date_analysis['monthly_counts']
    monthly_counts.plot(ax=axes[1, 1])
    axes[1, 1].set_title('Monthly Article Count')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Number of Articles')
    
    # Format x-axis to show months
    axes[1, 1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    
    plt.tight_layout()
    return fig


def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


def categorize_sentiment(value):
    if value > 0:
        return 'positive'
    elif value < 0:
        return 'negative'
    else:
        return 'neutral'
    


def get_subjectivity(text):
    blob = TextBlob(text)
    return blob.sentiment.subjectivity
    


# Time Series Analysis

# Examine how article publication times are distributed across different hours of the day
def perform_topic_modeling(df, column='headline', topics_count=5, words_count=10):

    # Tokenize and vectorize the text in the specified column
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(df[column])

    # Apply Latent Dirichlet Allocation for topic modeling
    lda_model = LatentDirichletAllocation(n_components=topics_count, random_state=42)
    lda_model.fit(doc_term_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    extracted_topics = []
    for idx, topic in enumerate(lda_model.components_):
        top_words = [(feature_names[i], topic[i]) for i in topic.argsort()[:-words_count - 1:-1]]
        extracted_topics.append(top_words)
    
    return extracted_topics

# Time Series Analysis

# Examine how article publication times are distributed across different hours of the day
def publication_time_distribution(df):
    df['date'] = pd.to_datetime(df['date'],format="ISO8601")
    df['hour'] = df['date'].dt.hour
    hourly_distribution = df['hour'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    hourly_distribution.plot(kind='bar')
    plt.title('Hourly Distribution of Article Publications')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Articles Published')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    peak_hour = hourly_distribution.idxmax()
    return f"The highest publication activity occurs at {peak_hour}:00"


# Detect days with significantly higher publication activity
def detect_publication_anomalies(df, threshold=2):
    daily_counts = df.groupby(df['date'].dt.date).size()
    mean_publications = daily_counts.mean()
    std_publications = daily_counts.std()
    
    anomalies = daily_counts[daily_counts > mean_publications + threshold * std_publications]
    return anomalies

def analyze_publication_trends(df, date_column='date'):
    # Convert the date column to datetime
    df['publication_date'] = pd.to_datetime(df[date_column], format='ISO8601')
    
    # Extract day names for trend analysis
    df['publication_day'] = df['publication_date'].dt.day_name()
    publication_trends = df.groupby('publication_day').size()
    
    # Extract time for time series analysis
    df['publication_time'] = df['publication_date'].dt.time
    
    # Plot publication frequency over time
    df.set_index('publication_date').resample('D').size().plot()
    plt.title('Publication Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.show()
    
    return publication_trends

def plot_publication_frequency_by_day(df, date_column='date'):
    # Convert the 'date' column to datetime format
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Set the 'date' column as the index
    df.set_index(date_column, inplace=True)
    
    # Group by day of the week
    weekly_publications = df.groupby(df.index.dayofweek).size()
    
    # Set the labels for the x-axis
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Plot the bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(days_of_week, weekly_publications)
    plt.title('Publication Frequency by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Publications')
    plt.show()


def get_top_publisher_domains(df, publisher_column='publisher', top_n=10):
    def extract_domain(email):
        match = re.search(r"@[\w.]+", email)
        if match:
            return match.group()[1:]
        return email

    # Apply the extract_domain function to the specified publisher column
    df["publisher_domain"] = df[publisher_column].apply(extract_domain)

    # Count the occurrences of each publisher domain
    domain_counts = df["publisher_domain"].value_counts()

    # Return the top N publisher domains
    return domain_counts.head(top_n)