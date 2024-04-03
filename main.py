import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import string
from nltk.util import ngrams
import re
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score



train_df = pd.read_csv('train.csv')


disasterTweets = train_df[train_df['sentiment'] == 'negative']['review']
normalTweets = train_df[train_df['sentiment'] == 'positive']['review']

#all_tweets = pd.concat([train_df['review'], test_df['review']])
all_tweets =train_df['review']

lemmatizer = WordNetLemmatizer()

all_words = []
for tweet in all_tweets:
    tweet = tweet.replace('br', '')
    tokens = word_tokenize(tweet)
    lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens if token not in string.punctuation]

    all_words.extend(lemmatized_words)


disaster_word_counts = Counter()
normal_word_counts = Counter()

for tweet in disasterTweets:
    tweet = tweet.replace('br', '')
    tokens = word_tokenize(tweet)
    lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens if token not in string.punctuation]
    disaster_word_counts.update(lemmatized_words)

for tweet in normalTweets:
    tweet = tweet.replace('br', '')
    tokens = word_tokenize(tweet)
    lemmatized_words = [lemmatizer.lemmatize(token) for token in tokens if token not in string.punctuation]
    normal_word_counts.update(lemmatized_words)

print("Most common words in disaster tweets:")
print(disaster_word_counts.most_common(10))

print("\nMost common words in normal tweets:")
print(normal_word_counts.most_common(10))





bigrams_list = list(ngrams(all_words, 2))
trigrams_list = list(ngrams(all_words, 3))

bigrams_count = Counter(bigrams_list)
trigrams_count = Counter(trigrams_list)

print("Top 20 Bigrams:")
for bigram, count in bigrams_count.most_common(20):
    print(' '.join(bigram), count)

print("\nTop 20 Trigrams:")
for trigram, count in trigrams_count.most_common(20):
    print(' '.join(trigram), count)






#part b


stop_words = set(stopwords.words('english'))

preprocessed_tweets = []
for tweet in all_tweets:
    tweet = re.sub(r'@\w+', '', tweet)

    tokens = word_tokenize(tweet)

    filtered_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens
                       if token.lower() not in stop_words and token.lower() not in string.punctuation]

    preprocessed_tweet = ' '.join(filtered_tokens)

    preprocessed_tweets.append(preprocessed_tweet)



X_train, X_test, y_train, y_test = train_test_split(preprocessed_tweets[:len(train_df)], train_df['sentiment'],
                                                    test_size=0.2, random_state=42)

for max_feat in [100, 1000]:
    print(f"\nMax Features: {max_feat}")
    vectorizer = CountVectorizer(max_features=max_feat)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")


























