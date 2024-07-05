# Intel-Product-Sentimental-analysis-2024
A python tool to classify the reviews from E-commerce sites as POSITIVE, NEGATIVE & COMPARATIVE
This project performs sentiment analysis on a dataset of textual reviews. The sentiment of each review is predicted using a Naive Bayes classifier trained on TF-IDF vectorized text data.

Project Structure
sentiment_analysis.py: Main Python script to perform sentiment analysis.
sentimental_analysis_updated.csv: Input dataset containing reviews and their actual sentiment labels.
Requirements
Ensure you have the following Python packages installed:
pandas
scikit-learn

You can install the required packages using pip: pip install pandas scikit-learn
Dataset
The dataset sentimental_analysis_updated.csv should be placed in the C:\\Users\\User\\Desktop\\ML\\ directory. It contains two columns:

reviews.text_reduce: The text of the reviews.
label: The actual sentiment label of each review.

Running the Code

To run the sentiment analysis script, execute the following command in your terminal or command prompt: python sentiment_analysis.py
Code Overview
1. Load the Dataset: The dataset is loaded using pandas.
2. Preprocess the Data: The reviews and their corresponding labels are extracted from the dataset.
3. Split the Dataset: The dataset is split into training and testing sets using an 80-20 split.
4. Vectorize the Text Data: The text data is converted into TF-IDF vectors using TfidfVectorizer from scikit-learn.
5. Train the Naive Bayes Classifier: The model is trained using the training data.
6. Make Predictions: Predictions are made on the test data.
7. Create Results DataFrame: A DataFrame is created to hold the reviews, their actual sentiments, and the predicted sentiments.
8. Separate Sentiments: Reviews are separated based on their predicted sentiments.
9. Print Structured Output: Reviews and their sentiments are printed in a structured manner.
10. Evaluate the Model: The accuracy of the model and a classification report are printed.

Output:
The script prints the following information:

All reviews with their actual and predicted sentiments.
Positive reviews with their actual and predicted sentiments.
Negative reviews with their actual and predicted sentiments.
Competition reviews with their actual and predicted sentiments.
Future reviews with their actual and predicted sentiments.
Model evaluation, including accuracy and a detailed classification report.

License
This project is licensed under the MIT License. See the LICENSE file for details.
