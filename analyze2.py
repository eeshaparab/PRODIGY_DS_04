import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the Data
file_path = r"C:\Users\eesha\OneDrive\Desktop\prodigy\Task4\twitter_training.csv"
df = pd.read_csv(file_path)

# Correct column names to be more usable
df.columns = ['index', 'topic', 'sentiment', 'text']

# Display the first few rows to verify changes
print("First few rows of the dataset:")
print(df.head())

# Check for missing values in the 'text' column
if df['text'].isnull().sum() > 0:
    print("\nWarning: Missing values found in the 'text' column. They will be dropped.")
    df = df.dropna(subset=['text'])

# 2. Preprocess the Data
def clean_text(text):
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    # Remove hashtags
    text = re.sub(r"#\w+", "", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower().strip()
    return text

# Apply the cleaning function to the text column
df['clean_text'] = df['text'].apply(lambda x: clean_text(x) if isinstance(x, str) else '')

# Check the cleaned text
print("\nOriginal and cleaned text examples:")
print(df[['text', 'clean_text']].head())

# 3. Sentiment Analysis
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

# Apply the polarity function to the clean text column
df['polarity'] = df['clean_text'].apply(lambda x: get_polarity(x) if x else 0)

def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# We will reclassify sentiment based on polarity for comparison
df['analyzed_sentiment'] = df['polarity'].apply(classify_sentiment)

# Check the sentiment distribution
print("\nSentiment distribution (original vs analyzed):")
print(df[['sentiment', 'analyzed_sentiment']].value_counts())

# 4. Visualization
plt.figure(figsize=(10, 6))
sns.countplot(x='analyzed_sentiment', data=df, palette='viridis')
plt.title('Analyzed Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# Plot polarity distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['polarity'], bins=50, kde=True, color='purple')
plt.title('Polarity Distribution')
plt.xlabel('Polarity')
plt.ylabel('Frequency')
plt.show()

# 5. Analyzing Sentiment Over Time (Optional)
# Convert the timestamp column to datetime if exists (assuming there is a timestamp)
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Group by date and sentiment
    sentiment_over_time = df.groupby([df['timestamp'].dt.date, 'analyzed_sentiment']).size().unstack()

    # Plot sentiment over time
    plt.figure(figsize=(12, 8))
    sentiment_over_time.plot(ax=plt.gca())
    plt.title('Sentiment Over Time')
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.show()
