Sentiment Analysis of Social Media Data
Project Overview
This project focuses on analyzing sentiment patterns in social media data to understand public opinion and attitudes towards specific topics or brands. The analysis is done using a dataset containing social media posts (e.g., tweets), and the sentiment is classified as Positive, Negative, or Neutral. The project includes data preprocessing, sentiment analysis using TextBlob, and visualizations of the results.

Dataset
The dataset used in this project is a CSV file containing the following columns:

index: An identifier for each entry.
topic: The topic or keyword associated with the social media post.
sentiment: The original sentiment label (e.g., Positive, Negative, Neutral).
text: The content of the social media post.

Features
Data Preprocessing: Cleans the text data by removing URLs, mentions, hashtags, and special characters, and converts text to lowercase.
Sentiment Analysis: Uses TextBlob to calculate the polarity of the text and classify it as Positive, Negative, or Neutral.
Visualization: Provides visualizations of the sentiment distribution and polarity distribution.

Requirements
Python 3.6+
Required libraries:
pandas
re (regular expressions)
textblob
matplotlib
seaborn

Example Output
After running the script, you will see:

A count plot showing the distribution of analyzed sentiments.
A histogram showing the distribution of polarity scores
