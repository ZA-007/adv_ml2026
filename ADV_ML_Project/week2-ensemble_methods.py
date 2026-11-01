

# Core libraries
import numpy as np
import pandas as pd
# Visualisation
from matplotlib import pyplot as plt
# Datasets
from sklearn.datasets import load_breast_cancer, make_classification
# Data splitting and validation, these helps us test how well a model performs
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedStratifiedKFold
)
# Pre-processing
from sklearn.preprocessing import LabelEncoder #label encoder converts category labels into numerical labels, such as 0 and 1
# Evaluation
from sklearn.metrics import accuracy_score, f1_score #accuracy is the proportion of predictions that are correct
# f1 score combines precision and recall into one measure

# Base learners, these are individual models that can be used on their won or combines into an ensemble
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Ensemble methods
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    BaggingClassifier,
    VotingClassifier)

# Load dataset
breast_cancer = load_breast_cancer()

# Create feature DataFrame (X), creates a table with the measurments and column names, think of x as a spreadsheet
X = pd.DataFrame(
    breast_cancer.data,
    columns=breast_cancer.feature_names
)  # breast_cancer.data contains the input measures used to make predictions and the .feature_names provides the names for those measurments

#Before training a supervised machine-learning model, we need to separate two things:
# X — the information the model uses to make a prediction.
# y — the correct answer the model is supposed to learn to predict.

# Create categorical target (Y) what the model is trying to predict
y = pd.Categorical.from_codes(
    breast_cancer.target,
    breast_cancer.target_names
)
#breast_cancer.target contains the target codes.
#breast_cancer.target_names contains the names of the classes, such as benign and malignant.
#pd.Categorical.from_codes(...) uses those codes and names to represent the answers as categories.
#y = saves those answers as y.
# So, if X contains 569 rows of measurements, y contains 569 corresponding answers — one for each row.
#Why do we need y? Because this is supervised learning. During training, the model needs to see both the measurements (X) and the correct answers (y) so it can learn the relationship between them.


# Convert target labels (Y) to integers/numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print("Dataset shape:", X.shape) #tells u number of rows and columns
print("Target classes:", encoder.classes_) # shows the class name the encoder found
#encoder = LabelEncoder() creates the tool that will do the conversion.
#fit_transform(y) learns which labels exist and converts them to integer codes.
#y = replaces the previous version of y with the encoded version.
#Is this necessary? Not always. It is a data-preparation choice in this practical, not a special requirement of AdaBoost. Scikit-learn's AdaBoostClassifier can generally accept class labels as text, too.


#understanding train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1, #Makes the random split repeatable. Running the code again gives the same split,
    stratify=y #Keeps class proportions approximately similar in both sets
)
print("Training observations:", X_train.shape[0])
print("Testing observations:", X_test.shape[0])

#X_train.shape[0] means the number of training rows.
#X_test.shape[0] means the number of test rows.
# [0] selects the first part of the shape, which is the row count.

# Boosting with AdaBoost, can be reused

max_depth = 1
boosting_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=1
    ),
    n_estimators=200,
    random_state=1
)
boosting_model.fit(X_train, y_train)
boost_predictions = boosting_model.predict(X_test)
boost_accuracy = accuracy_score(
    y_test,
    boost_predictions
)
boost_f1 = f1_score(
    y_test,
    boost_predictions
)
print(f"AdaBoost Accuracy: {boost_accuracy:.2f}")
print(f"AdaBoost F1 Score: {boost_f1:.2f}")

#AdaBoost builds learners sequentially. It gives more attention to training examples that earlier learners got wrong, helping later learners focus on difficult cases.
# It then combines their contributions into a final prediction.

#RandomForestClassifier(...)	Creates a Random Forest classification model.
#n_estimators=200	Builds 200 decision trees in the forest.
#max_depth=1	Restricts each tree to one split, making the trees very shallow.
#random_state=1	Helps make results reproducible.
#n_jobs=-1	Uses all available CPU cores for supported parallel work, which can speed up training.
#.fit(...)	Trains the forest.
#.predict(...)	Predicts classes for the test data.
#Accuracy and F1	Measure how well the predictions perform.




# Bagging with RandomForest

bagging_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=1,
    random_state=1,
    n_jobs=-1
)
bagging_model.fit(X_train, y_train)
bagging_predictions = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(
    y_test,
    bagging_predictions
)
bagging_f1 = f1_score(
    y_test,
    bagging_predictions
)
print(f"Random Forest Accuracy: {bagging_accuracy:.2f}")
print(f"Random Forest F1 Score: {bagging_f1:.2f}")

#Random Forest uses many decision trees and combines their predictions. Unlike AdaBoost, which trains learners sequentially and focuses more on difficult examples, Random Forest trees are generally trained independently, allowing them to be built in parallel.
#Why use it? A single decision tree can be unstable and overfit its training data. Combining many trees can make predictions more robust.



# Bagging with different base learner (KNN)
# Step 1 — Generate a classification dataset , fake one for practicing
X_knn, y_knn = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=5
)
print("Dataset shape:", X_knn.shape)

# create the KNN bagged model
knn_bagging = BaggingClassifier(
    estimator=KNeighborsClassifier(),
    n_estimators=10,
    random_state=1,
    n_jobs=-1
)

#Bagging means Bootstrap Aggregating. The idea is to train several versions of a learner on different bootstrap samples and combine their predictions. This can reduce prediction variability.
#When would you use it? When you want to test whether combining multiple versions of a base learner improves performance or stability compared with using a single learner.

# This tests the model repeatedly using different training and validation splits.
cv = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=3,
    random_state=1
)

#RepeatedStratifiedKFold	The method used to split the data and repeat the testing.
#n_splits=10	Split the data into 10 parts, called folds.
#n_repeats=3	Repeat the entire 10-fold process 3 times.
#random_state=1	Helps make the random splitting reproducible, so you can get the same splits again.
#cv=cv	Tells cross_val_score to use the cross-validation method you created above.

#Each fold gets a turn being the validation data. The model is trained and evaluated 10 times per round.
#Because your code has n_repeats=3, that means:
#10 splits × 3 repeats = 30 scores.

#Cross-validation is a way of testing how well a machine learning model performs by splitting the data into different parts multiple times.
#Imagine you have 1,000 rows of data. You don't want to test the model using only one split, because that one split might give you a score that is unusually good or bad.
#Instead, your code tests it repeatedly using different splits.


# This runs the evaluation and stores the 30 accuracy scores in knn_scores.
knn_scores = cross_val_score(
    knn_bagging,
    X_knn,
    y_knn,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    error_score="raise"
)
print(f"Mean Accuracy: {knn_scores.mean():.3f}") # the average accuracy across the 30 tests.
print(f"Standard Deviation: {knn_scores.std():.3f}") #how much the scores vary from one test to another. A smaller value means the scores are more consistent.

# Soft voting ensemble
# create the dataset
X_svm, y_svm = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=2
)


#SVM is a machine learning model that learns to separate two classes.
#Imagine your data contains two groups: cats and dogs. The SVM tries to find a boundary that separates the two groups so it can classify new examples.
#What is happening in your practical?
#Your lecturer creates five versions of SVM.
#They all use the same type of model, but degree=1, degree=2, up to degree=5 changes the polynomial complexity of the boundary they can learn.
#Soft voting combines their predicted probabilities to make one final prediction.

# create the soft voting ensemble, create five SVM models using polynomial kernels.
def create_soft_voting_model():
    models = [
        ("svm1", SVC(
            probability=True,
            kernel="poly",
            degree=1,
            random_state=1
        )),
        ("svm2", SVC(
            probability=True,
            kernel="poly",
            degree=2,
            random_state=1)),
        ("svm3", SVC(
            probability=True,
            kernel="poly",
            degree=3,
            random_state=1
        )),
        ("svm4", SVC(
            probability=True,
            kernel="poly",
            degree=4,
            random_state=1
        )),
        ("svm5", SVC(
            probability=True,
            kernel="poly",
            degree=5,
            random_state=1
        ))
    ]
    ensemble = VotingClassifier(
        estimators=models,
        voting="soft"
    )
    return ensemble


# Step 3 — Create models for comparison
def create_soft_voting_models():

    models = {
        "SVM-1": SVC(probability=True, kernel="poly", degree=1, random_state=1),
        "SVM-2": SVC(probability=True, kernel="poly", degree=2, random_state=1),
        "SVM-3": SVC(probability=True, kernel="poly", degree=3, random_state=1),
        "SVM-4": SVC(probability=True, kernel="poly", degree=4, random_state=1),
        "SVM-5": SVC(probability=True, kernel="poly", degree=5, random_state=1),
        "Soft Voting": create_soft_voting_model()
    }

    return models

#Step 4 — Evaluate using repeated cross-validation, The code tests each model across different training and validation splits.

models = create_soft_voting_models()
results = []
names = []

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_svm,
        y_svm,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        error_score="raise"
    )

    results.append(scores)
    names.append(name)

    print(
        f"{name:12s} "
        f"Mean = {scores.mean():.3f} | "
        f"Std = {scores.std():.3f}"
    )

#Step 5 — Visualise the results, The boxplot compares the accuracy of all six models.
plt.figure(figsize=(10, 5))
plt.boxplot(
results,
tick_labels=names,
showmeans=True
)
plt.ylabel("Accuracy")
plt.title("Soft Voting vs Individual SVM Models")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#HARD VOTING creat the KNN voting ensemble
def create_hard_voting_model():
 models = [
("knn1", KNeighborsClassifier(n_neighbors=1)),
("knn3", KNeighborsClassifier(n_neighbors=3)),
("knn5", KNeighborsClassifier(n_neighbors=5)),
("knn7", KNeighborsClassifier(n_neighbors=7)),
("knn9", KNeighborsClassifier(n_neighbors=9))
 ]
 ensemble = VotingClassifier(
  estimators=models,
  voting="hard"
  )
 return ensemble

#Step 2 — Create the models for comparison
def create_hard_voting_models():
 models = {
"KNN-1": KNeighborsClassifier(n_neighbors=1),
"KNN-3": KNeighborsClassifier(n_neighbors=3),
"KNN-5": KNeighborsClassifier(n_neighbors=5),
"KNN-7": KNeighborsClassifier(n_neighbors=7),
"KNN-9": KNeighborsClassifier(n_neighbors=9),
"Hard Voting": create_hard_voting_model()
 }
 return models

#Step 3 — Evaluate the models
models = create_hard_voting_models()
results = []
names = []
for name, model in models.items():
 scores = cross_val_score(
model,
X_svm,
y_svm,
scoring="accuracy",
cv=cv,
n_jobs=-1,
error_score="raise"
)
results.append(scores)
names.append(name)
print(
f"{name:12s} "
f"Mean = {scores.mean():.3f} | "
f"Std = {scores.std():.3f}"
)

#Step 4 — Visualise
plt.figure(figsize=(10, 5))
plt.boxplot(
results,
tick_labels=names,
showmeans=True
)
plt.ylabel("Accuracy")
plt.title("Hard Voting vs Individual KNN Models")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()