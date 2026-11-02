# Standard imports and settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
nltk.download('punkt') # for tokenization if needed
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

#Our eventual goal is to build a machine learning model that can read a movie review and predict whether it is positive or negative.
#Load and exploration, shape and class balance
imdb = pd.read_csv('IMDB Dataset.csv') # Colab path; adjust if needed
print("Shape:", imdb.shape) # how many rows and columns in the dataset
print(imdb['sentiment'].value_counts())  # selects the sentiment column and counts how many times each unique value appears, we found that we had a balanced dataset, makes it easier to train and evaluate
imdb.head(6) # displays the first few rows of the dataset

# now we are asked to display three random reviews:
for i in imdb.sample(3).index:
    print("=== SAMPLE ===")
    print(imdb.loc[i, 'review'][:500], "\n---label:", imdb.loc[i, 'sentiment'])
# .sample() randomly selects rows from a DataFrame. The 3 means we want three random rows. For example, it might select rows 125, 3400 and 27000. Every time you run it, you might get different reviews.
#This retrieves the index numbers of the three randomly selected rows. An index is simply the identifier of a row in the dataset.
# the for loop (for i...) repeats an action for every item in a collection so here it means:
#Select three random rows.
#Take the first row's index and store it in i.
#Execute the code inside the loop.
#Move to the second index and repeat.
#Move to the third index and repeat.
#.loc[] allows us to retrieve information from a DataFrame using its row index and column name.
#Here: i identifies the row. 'review' identifies the column.
#[:500] tells python to display only the first 500 characters of the review , good if theyre super long
# last part : \n creates a new line.  ---label: prints a label.  imdb.loc[i, 'sentiment'] retrieves whether the review is positive or negative.

#step 2 clean the text
#Objective: Implement a robust clean_text() function (HTML removal, lowercasing, non-alpha removal, optional stemming/lemmatization, stopword removal)

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

#creating our cleaning function
def clean_text(text, do_lemmatize=True, remove_digits=True):
    # 1. HTML. this removes the HTML tags some reviews may contain
    text = BeautifulSoup(text, "html.parser").get_text()
    # 2. remove [..] annotations. so it removes anything in the square brackets and replaces the text space with a space
    text = re.sub(r'\[[^\]]*\]', ' ', text)
    # 3. keep letters (convert digits optionally). removes punctuations and optionally digits
    if remove_digits:
      text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    else:
      text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text) #Means keep: a-z: lowercase letters. A-Z: uppercase letters. \s: whitespace (spaces, tabs and newlines). Everything else is replaced with a space.
    # 4. lowercase and tokenize. it converts all text to lowercase and .split() splits the sentences into individual words called tokens
    tokens = text.lower().split()
    # 5. remove stopwords. Stopwords are common words that are often removed during text processing. like the, is, a ,an . We want to reduce unnecessary words and focus on words that may help identify sentiment.
    #One important consideration: removing stopwords can sometimes remove meaningful words such as not, which can reverse the meaning of a review. So stopword removal isn't always beneficial for sentiment analysis.
    tokens = [t for t in tokens if t not in stop_words]
    # 6. lemmatize (optional). This checks whether lemmatization is enabled. If it is, it goes through every word and applies the lemmatizer. Notice this is similar to stopword removal, but instead of removing words, we're changing their form.
    if do_lemmatize:
      tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens) # return the clean text

#This creates a function called clean_text. It takes three inputs: text: The original movie review. do_lemmatize=True: Whether we want to lemmatize words. Enabled by default. remove_digits=True: Whether we want to remove numbers. Enabled by default.


# Apply the cleaning function to our dataset
imdb_sample = imdb.sample(2000, random_state=42).copy() # selects 2000 random rows . random_state=42 makes sure we get the same random slection each time we run
imdb_sample['clean'] = imdb_sample['review'].apply(clean_text)# selects our original reviews, applies the cleaning function to every review and create a new column called clean containing the cleaned reviews.
print(imdb_sample[['review','clean']].head(3)) # displays the first three rows showing only the original review and cleaned review side by side .

#step 3 full dataset preprocessing
#Objective: Apply cleaning to whole dataset (or to a subset if runtime is a concern). Use .sample() if time-limited.

imdb['clean'] = imdb['review'].apply(clean_text) # selects the original review column , run it on every single review then create a new column containing the cleaned reviews
imdb.to_csv('imdb_clean_short_debug.csv', index=False) # optional checkpoint save
print("Missing values:", imdb['clean'].isna().sum())
#Checkpoint: imdb['clean'].head() — no NaNs. If there are NaNs, fill with empty string: imdb['clean'] = imdb['clean'].fillna('').

#step 4 feature extraction: CountVectorizer & TF-IDF. convert text into numbers so that the ml model can understand it
#Objective: Convert text → numerical features. Use max_features to keep runtime friendly.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

MAX_FEAT = 5000 # manageable for a class demo, maximum number of features
bow_vec = CountVectorizer(max_features=MAX_FEAT, ngram_range=(1,2)) # BOW counts how often words appear
tfidf_vec = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=(1,2)) # TF-IDF measures how important words are
X_bow = bow_vec.fit_transform(imdb['clean']) # Convert cleaned reviews into numbers using BOW
X_tfidf = tfidf_vec.fit_transform(imdb['clean']) # Convert cleaned reviews into numbers using TF-IDF

# Check the size of the numerical data
print("BOW shape:", X_bow.shape)
print("TFIDF shape:", X_tfidf.shape)


# Display the first 30 features (words/phrases)
print(bow_vec.get_feature_names_out()[:30])

# step 5 train and test split

from sklearn.model_selection import train_test_split

#Convert sentiment into numbers, and store these numerical variables in y
y = imdb['sentiment'].map({'positive':1, 'negative':0}).values

#split our BOW data
X_tr_bow, X_te_bow, y_tr, y_te = train_test_split(X_bow, y, test_size=0.2, random_state=42, stratify=y)
#split our TF-IDF data
X_tr_tfidf, X_te_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

print("Train shape:", X_tr_bow.shape, "Test shape:", X_te_bow.shape)

# Step 6 — Baseline models: Naive Bayes & Logistic Regression. finally training the ml models
#Objective: Fit baseline classifiers and compare results. Naive Bayes (MNB): Learns patterns using probabilities. Logistic Regression (LR): Learns patterns to classify reviews into positive or negative.

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# MultinomialNB on BOW. train naive bayes
mnb = MultinomialNB() # creates our model
mnb.fit(X_tr_bow, y_tr) # trains the model using our training data
mnb_pred = mnb.predict(X_te_bow) # uses the trained model to predict sentiment for our test reviews

#check its accuracy
print("MNB (BOW) acc:", accuracy_score(y_te, mnb_pred)) # compares the actual answers (y_te) against the models predictions (mnb_pred)
print(classification_report(y_te, mnb_pred)) # gives us more detailed performance measurements

# Logistic Regression on BOW. train LR , does exacty the same as NB
lr = LogisticRegression(max_iter=400) # allows the model up to 400 iterations to find suitable parameters during training
lr.fit(X_tr_bow, y_tr)
lr_pred = lr.predict(X_te_bow)
print("LR (BOW) acc:", accuracy_score(y_te, lr_pred))
print(classification_report(y_te, lr_pred))

#Measurement Meaning
#Accuracy, Overall percentage of correct predictions
#Precision, Of the reviews predicted positive, how many really were positive?
# Recall, Of all actual positive reviews, how many did the model identify?
# F1-score, Combines precision and recall into one score
#For your task, pay particular attention to positive recall. why? Because the worksheet asks which model is better at identifying positive reviews.

# if we want to see what happens when we use TF-IDF instead , simply repeat the process , changing the features
# NAIVE BAYES using TF-IDF
mnb_tfidf = MultinomialNB()
mnb_tfidf.fit(X_tr_tfidf, y_tr)
mnb_tfidf_pred = mnb_tfidf.predict(X_te_tfidf)

print("MNB TF-IDF:", accuracy_score(y_te, mnb_tfidf_pred))

# LOGISTIC REGRESSION using TF-IDF
lr_tfidf = LogisticRegression(max_iter=400)
lr_tfidf.fit(X_tr_tfidf, y_tr)
lr_tfidf_pred = lr_tfidf.predict(X_te_tfidf)

print("LR TF-IDF:", accuracy_score(y_te, lr_tfidf_pred))

#step 7 small model competition . we trained two models, now were trying different settings and algorithms to see whether we can improve their accuracy
#there are 3 rounds: Change features: Experiment with the number of words (5,000 vs 8,000) or word combinations (n-grams). Change text cleaning: Compare lemmatization, stemming and no cleaning reduction. Change the algorithm: Try different models such as SGDClassifier, SVM or PassiveAggressive.

# change ngram or features, experiment with TF-IDF
vec2 = TfidfVectorizer(max_features=8000, ngram_range=(1,3)) # uses up to 8000 features instead of 5000
X2 = vec2.fit_transform(imdb['clean'])
# then split and run MNB/LR as above
# try SGDClassifier
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier(max_iter=500, tol=1e-3)
sgd.fit(X_tr_tfidf, y_tr)
print("SGD (tfidf) acc:", accuracy_score(y_te, sgd.predict(X_te_tfidf)))