# Intel-Product-Sentimental-analysis-2024
A python tool to classify the reviews from E-commerce sites as POSITIVE, NEGATIVE & COMPARATIVE
This project performs sentiment analysis on a dataset of textual reviews. The sentiment of each review is predicted using a Naive Bayes classifier trained on TF-IDF vectorized text data.

About our files:
Project Structure:-

intel - supervised approach - RANDOMFOREST APPROACH
Untrained(optional) - unsupervised approach - KMEANS APPROACH

Ensure you have the following Python packages installed:
pandas
scikit-learn
seaborn

You can install the required packages using pip: pip install pandas scikit-learn
Dataset
The dataset sentimental_analysis_updated.csv should be placed in the #YOUR_PATH. It contains two columns:

reviews.text_reduce: The text of the reviews.
label: The actual sentiment label of each review.


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
