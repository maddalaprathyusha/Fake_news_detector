# Fake-News-Detection-Using-PassiveAggressiveClassifier-and-TfidfVectorizer
This Project combat the spread of fake news using Python, TfidfVectorizer, and PassiveAggressiveClassifier. This project helps verify the legitimacy of information and prevents the harmful effects of misinformation in the digital age.

**Overview:**

With the rapid growing rate of information over the internet, the topic of fake news detection on social media has recently attracted tremendous attention. The basic countermeasure of comparing websites against a list of labeled fake news sources is inflexible, and so a machine learning approach is desirable. This project aims to use Natural Language Processing (NLP) and Passive Aggressive Classifier to detect fake news directly, based on the text content of news articles.

**Objectives**
1. To investigate and identify the key features of fake news
2. To design and develop a machine learning based fake news detection system
3. To validate the effectiveness of a machine learning based fake news detection system
This project implements a Fake News Detection system using a Passive Aggressive Classifier. The model is trained to classify news articles as either real or fake based on their textual content. TF-IDF vectorization is applied to transform text into a numerical format suitable for machine learning.

**Dataset**
The dataset used (news.csv) contains two columns:

text: The news article content.

label: The classification of the news (either FAKE or REAL).

**Project Structure**
Loading Data: Read the dataset using pandas.

Preprocessing: Apply TF-IDF vectorization to convert text data into numerical features.

Model Training: Train a Passive Aggressive Classifier with a maximum of 50 iterations.

Evaluation: Measure the model's accuracy and generate a confusion matrix.

**How it Works**
Load and explore the dataset.

Split the dataset into training and testing sets (80%-20% split).

Transform the text data into TF-IDF features.

Train a Passive Aggressive Classifier on the training data.

Predict and evaluate on the test data.
**
Requirements**
Make sure you have the following libraries installed:

pip install numpy pandas scikit-learn

**Running the Project**

Clone this repository or download the code.
Place the news.csv dataset in the project directory.
Run the Python script:
python fake_news_detection.py

**Output**
Accuracy: Displays the model accuracy on the test set.

Confusion Matrix: Shows true positives, true negatives, false positives, and false negatives.

**Example Output:**

Accuracy: 93.5%
Confusion Matrix:
[[588  50]
 [ 43 584]]
 
**Model Used**

TF-IDF Vectorizer: Converts raw text into TF-IDF features.
Passive Aggressive Classifier: A linear model that remains largely passive for correct classifications and aggressively updates for misclassified samples.

Dataset: Kaggle Fake News Dataset 

Libraries: Numpy, Pandas, Scikit-learn



