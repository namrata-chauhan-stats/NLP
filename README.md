#Sentiment_Classification_of_Social_Media_Reviews
Project Overview

In this project, I worked on sentiment analysis of social media reviews to understand how customers express opinions online. The goal was to automatically classify user reviews into positive or negative sentiment, helping businesses track customer satisfaction and feedback at scale.

The project involved working with a large, real-world text dataset of 50,000+ social media reviews, applying Natural Language Processing (NLP) techniques, and building machine learning models to accurately capture sentiment patterns hidden in unstructured text.

Dataset Description

Size: 50,000+ social media reviews

Type: Unstructured text data

Target Variable: Sentiment label (Positive / Negative)

Data Preprocessing & Text Cleaning

Raw social media data is often noisy and inconsistent. To ensure high-quality inputs for modeling, several preprocessing steps were performed:

Removed HTML tags, URLs, special characters, and punctuation

Converted text to lowercase for standardization

Eliminated stopwords to reduce noise

Applied tokenization to split text into meaningful units

Used stemming to reduce words to their root forms

These steps helped improve model performance by focusing only on meaningful textual information.

Exploratory Text Analysis

Before modeling, exploratory analysis was conducted to better understand the data:

Analyzed word frequency distributions for positive and negative reviews

Identified commonly occurring terms and sentiment-specific patterns

Observed differences in language usage between positive and negative sentiments

This analysis helped validate assumptions and guide feature representation choices.

Feature Engineering

To convert text into numerical form suitable for machine learning models, the following techniques were used:

Bag of Words (BoW): Captured word occurrence patterns

TF-IDF (Term Frequency–Inverse Document Frequency): Highlighted important and sentiment-relevant words while reducing the impact of frequently occurring but less informative terms

Given the high dimensionality of text data, Principal Component Analysis (PCA) was applied to reduce feature space while retaining essential information.

Model Development

Multiple machine learning models were trained and evaluated:

Naive Bayes: Leveraged probabilistic assumptions well-suited for text classification

Random Forest: Captured non-linear patterns and interactions between features

Dimensionality reduction using PCA helped improve computational efficiency without significantly impacting accuracy.

Model Performance

Accuracy: ~84%

Balanced performance across positive and negative classes

Reduced dimensionality led to faster training and better generalization

Tools & Technologies

Python

Pandas, NumPy

Scikit-learn

NLTK

Matplotlib, Seaborn

Key Takeaways

NLP preprocessing plays a critical role in sentiment classification performance

TF-IDF provides richer contextual representation compared to simple word counts

Dimensionality reduction is essential when working with large text feature spaces

Even simple models like Naive Bayes can perform well with properly engineered features
