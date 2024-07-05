import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = "C:\\Users\\sujju\\Desktop\\ML\\245_1_part3.csv"
df = pd.read_csv(file_path)

# Preprocess the data
X = df['reviews.text_reduce']
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Create a DataFrame with reviews, actual sentiments, and predicted sentiments
results_df = pd.DataFrame({'Review': X_test, 'Actual Sentiment': y_test, 'Predicted Sentiment': y_pred})

# Separate sentiments
positive_reviews = results_df[results_df['Predicted Sentiment'] == 'positive']
negative_reviews = results_df[results_df['Predicted Sentiment'] == 'negative']
competition_reviews = results_df[results_df['Predicted Sentiment'] == 'competition']
future_reviews = results_df[results_df['Predicted Sentiment'] == 'future']

# Print structured output
print("All Reviews:\n")
for index, row in results_df.iterrows():
    print(f"Review: {row['Review']}\nActual Sentiment: {row['Actual Sentiment']}\nPredicted Sentiment: {row['Predicted Sentiment']}\n")

print("\nPositive Reviews:\n")
for index, row in positive_reviews.iterrows():
    print(f"Review: {row['Review']}\nActual Sentiment: {row['Actual Sentiment']}\nPredicted Sentiment: {row['Predicted Sentiment']}\n")

print("\nNegative Reviews:\n")
for index, row in negative_reviews.iterrows():
    print(f"Review: {row['Review']}\nActual Sentiment: {row['Actual Sentiment']}\nPredicted Sentiment: {row['Predicted Sentiment']}\n")

print("\nCompetition Reviews:\n")
for index, row in competition_reviews.iterrows():
    print(f"Review: {row['Review']}\nActual Sentiment: {row['Actual Sentiment']}\nPredicted Sentiment: {row['Predicted Sentiment']}\n")

print("\nFuture Reviews:\n")
for index, row in future_reviews.iterrows():
    print(f"Review: {row['Review']}\nActual Sentiment: {row['Actual Sentiment']}\nPredicted Sentiment: {row['Predicted Sentiment']}\n")

# Print counts of each sentiment with one-line space before
print("\n")
print("\nSentiment Counts:")
print(f"\n\tTotal Reviews: {len(results_df)}")
print(f"\n\tPositive Reviews: {len(positive_reviews)}")
print(f"\n\tNegative Reviews: {len(negative_reviews)}")
print(f"\n\tCompetition Reviews: {len(competition_reviews)}")
print(f"\n\tFuture Reviews: {len(future_reviews)}")
print("\n")
print("\n")

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\n")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Calculate and print correlation matrix (cross-tabulation)
correlation_matrix = pd.crosstab(results_df['Actual Sentiment'], results_df['Predicted Sentiment'], rownames=['Actual Sentiment'], colnames=['Predicted Sentiment'])
print("\nCorrelation Matrix:\n")
print(correlation_matrix)