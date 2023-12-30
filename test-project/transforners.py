import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Sample sentence
sentence = "I love sunny days but hate the rain."

# Downloading VADER lexicon
nltk.download('vader_lexicon')

# Initializing Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Analyzing the sentiment
sentiment = sia.polarity_scores(sentence)
print("NLTK Sentiment Analysis:", sentiment)
