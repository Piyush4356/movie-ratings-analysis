# src/model.py
"""
Movie Ratings Analysis - Core Script
Includes: Data Loading, EDA, ML Model, Sentiment Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from wordcloud import WordCloud
from textblob import TextBlob

# -------------------------------
# Load Dataset
# -------------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df

# -------------------------------
# Train ML Model
# -------------------------------
def train_model(df):
    df = df[['budget', 'revenue', 'vote_average']].dropna()
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)]

    X = df[['budget', 'vote_average']]
    y = df['revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Model Performance:")
    print("R² Score:", r2_score(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    return model

# -------------------------------
# Sentiment Analysis
# -------------------------------
def sentiment_analysis(df):
    df['sentiment'] = df['overview'].fillna("").apply(lambda x: TextBlob(x).sentiment.polarity)
    print("Average sentiment:", df['sentiment'].mean())

    # Plot sentiment vs rating
    sns.scatterplot(x="sentiment", y="vote_average", data=df)
    plt.title("Sentiment vs Rating")
    plt.show()

    # Wordcloud for positive movies
    text = " ".join(df[df['sentiment'] > 0.2]['overview'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# -------------------------------
# Main Script
# -------------------------------
if __name__ == "__main__":
    path = "../data/tmdb_5000_movies.csv"   # adjust path if needed
    df = load_data(path)
    model = train_model(df)
    sentiment_analysis(df)
