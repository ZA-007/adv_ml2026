"""
WEEK 2 — ENSEMBLE METHODS: EXAM COPY-AND-PASTE CODE BANK
Based on the Week 2 practical: Bagging, Boosting, Random Forest and Voting.

HOW TO USE THIS FILE
1. Read the exam case study and decide what it asks you to predict.
2. Choose the matching model section below.
3. Copy the relevant code into your exam script.
4. Replace the clearly marked dataset names/target names where needed.
5. Run it and check the results.
6. Follow your exam's rules about permitted files/code.

IMPORTANT
- These are reusable templates, not one script to run all at once.
- Each model section assumes you have already prepared X and y:
      X = the input/features columns
      y = the target/answer column you want to predict
- Do NOT include the target/answer column inside X.
- If the exam requires you to use the exact practical code or a particular evaluation
  method, follow the exam instructions first.
"""

# ============================================================
# 0. PREPARE YOUR DATA (DO THIS BEFORE THE MODEL CODE)
# ============================================================

# Example only — replace "target_column" with the actual target column in your dataset.
# If your data is stored in a DataFrame called df:
#
# X = df.drop(columns=["target_column"])
# y = df["target_column"]
#
# Replace:
#   df             -> the actual name of your DataFrame
#   target_column  -> the exact column name you are predicting
#
# Example: if the column is called "default_status":
# X = df.drop(columns=["default_status"])
# y = df["default_status"]


# ============================================================
# 1. OPTIONAL TRAIN / TEST SPLIT
# Use when you need to test a model on held-out data.
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# X and y must already have been prepared above.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=1,
    stratify=y
)

# Meaning:
# test_size=0.25 -> 25% of rows for testing, 75% for training.
# random_state=1 -> helps make the split repeatable.
# stratify=y -> keeps class proportions similar in train and test sets.
#
# If the exam's practical uses cross-validation instead, use the cross-validation
# section near the bottom. Do not automatically use both; follow the question.


# ============================================================
# 2. BAGGING — COPY THIS SECTION
# ============================================================

# WHEN TO USE:
# You want to combine multiple versions of a model trained on different bootstrap
# samples. The Week 2 practical's example uses KNN inside BaggingClassifier.

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

bagging_model = BaggingClassifier(
    estimator=KNeighborsClassifier(),
    n_estimators=10,
    random_state=1,
    n_jobs=-1
)

bagging_model.fit(X_train, y_train)
bagging_predictions = bagging_model.predict(X_test)

print("Bagging accuracy:", accuracy_score(y_test, bagging_predictions))
print("Bagging F1:", f1_score(y_test, bagging_predictions))

# WHAT YOU MAY NEED TO CHANGE:
# - X_train, y_train, X_test, y_test:
#   Keep these names if you used the train/test split code above.
#   If your own variables have different names, replace them consistently.
# - estimator=KNeighborsClassifier():
#   This is the base model used inside bagging in this template. The Week 2 practical
# - voting="hard": keep this for hard voting.
# - f1_score: for more than two classes, use average="weighted".
#
# REMEMBER:
# Hard voting = each model votes for a class; the most votes wins.


# ============================================================
# 6. SOFT VOTING — COPY THIS SECTION
# ============================================================

# WHEN TO USE:
# You want to combine the class probabilities predicted by several models. The Week 2
# practical uses SVC models with polynomial kernels of different degrees.
# probability=True is important for SVC soft voting.

from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

soft_voting_model = VotingClassifier(
    estimators=[
        ("svm1", SVC(kernel="poly", degree=1, probability=True)),
        ("svm2", SVC(kernel="poly", degree=2, probability=True)),
        ("svm3", SVC(kernel="poly", degree=3, probability=True)),
        ("svm4", SVC(kernel="poly", degree=4, probability=True)),
        ("svm5", SVC(kernel="poly", degree=5, probability=True))
    ],
    voting="soft"
)

soft_voting_model.fit(X_train, y_train)
soft_voting_predictions = soft_voting_model.predict(X_test)

print("Soft voting accuracy:", accuracy_score(y_test, soft_voting_predictions))
print("Soft voting F1:", f1_score(y_test, soft_voting_predictions))

# WHAT YOU MAY NEED TO CHANGE:
# - X_train, y_train, X_test, y_test: match your split variable names.
# - kernel="poly" and degree=1...5: the practical compares polynomial SVC models.
#   Keep these to follow the practical; change them only if the question asks.
# - probability=True: do not remove this from SVC models used for soft voting.
# - voting="soft": this is what makes the ensemble combine probabilities.
# - f1_score: for more than two classes, use average="weighted".
#
# REMEMBER:
# Soft voting = combines predicted probabilities, not just the class labels.


# ============================================================
# 7. CROSS-VALIDATION — ALTERNATIVE EVALUATION PATTERN
# Use this when the question asks you to evaluate/compare models using CV.
# This follows the Week 2 practical's repeated stratified 10-fold setup.
# ============================================================

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

cv = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=3,
    random_state=1
)

# Replace "model" below with the name of ONE model object you created, for example:
# bagging_model, boosting_model, random_forest_model, hard_voting_model,
# or soft_voting_model.
#
# Example for AdaBoost:
# scores = cross_val_score(boosting_model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
# print("Mean accuracy:", scores.mean())
# print("Standard deviation:", scores.std())

# Meaning:
# 10 folds x 3 repeats = 30 evaluation scores.
# mean = average score.
# std = how much scores vary between evaluations.
# mean 0.90 means average accuracy is 90%; it does not mean every score was 90%.
#
# CROSS-VALIDATION VS TRAIN/TEST:
# Use the approach requested by the exam. The practical uses cross-validation for
# several activities and train/test evaluation for its AdaBoost/RF comparison.
# Do not report cross-validation results as if they came from the held-out test set.