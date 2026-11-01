"""
YEAR 2 SURVIVAL CHEAT SHEET & ML BLUEPRINT
Copy and paste this into PyCharm for your quick-reference guide.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. LOADING & INSPECTING DATA (Pandas)
# When to use: At the very beginning of any project to verify your dataset loaded correctly.
# ==========================================

# df = pd.read_csv('your_dataset.csv')  # What it does: Loads a CSV file into a pandas DataFrame.
# df.head()          # What it does: Prints the first 5 rows. Use this to check column names.
# df.info()          # What it does: Shows data types (int, float, object) and spots non-null counts.
# df.isna().sum()    # What it does: Counts missing (NaN) values per column so you know what needs fixing.
# df.describe()      # What it does: Generates summary stats (mean, min, max) for all number columns.


# ==========================================
# 2. FILTERING & SELECTING DATA
# When to use: When you want to isolate specific rows, columns, or conditions before modeling.
# ==========================================

# ages = df['Age']                         # What it does: Selects a single column by name.
# adults = df[df['Age'] > 18]              # What it does: Filters rows where Age is greater than 18.
# targeted = df[(df['Age'] > 18) & (df['Score'] > 80)]  # What it does: Combines multiple conditions (AND = &, OR = |).
# subset = df.loc[0:5, ['Age', 'Score']]   # What it does: Slices specific row ranges and specific column names explicitly.


# ==========================================
# 3. HANDLING MISSING DATA
# When to use: When missing values cause your machine learning model or calculation to throw errors.
# ==========================================

# df_clean = df.dropna()                   # What it does: Drops any row that contains even a single missing value.
# df['Age'] = df['Age'].fillna(0)          # What it does: Replaces all NaN/missing values in 'Age' with 0.
# mean_val = df['Age'].mean()
# df['Age'] = df['Age'].fillna(mean_val)   # What it does: Fills missing values with the column average (industry standard).


# ==========================================
# 4. CORRELATION & FEATURE ANALYSIS
# When to use: To find out which variables actually relate to your target before training a model.
# ==========================================

# correlation_matrix = df.corr()           # What it does: Computes Pearson correlation coefficients between all numerical columns.

# plt.figure(figsize=(8, 6))               # What it does: Creates a blank canvas window for a chart.
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')  # What it does: Plots a visual heatmap with numbers.
# plt.title('Feature Correlation Matrix')  # What it does: Adds a title to your plot.
# plt.show()                               # What it does: Renders and pops up the visualization window.


# ==========================================
# 5. THE CORE MACHINE LEARNING BLUEPRINT
# When to use: For every standard supervised learning task (Regression/Classification) in Year 2 and Year 3.
# ==========================================
from sklearn.linear_model import LinearRegression  # Example model import
from sklearn.model_selection import train_test_split

# Step 1: Define features (X) and target (y)
# X = df[['feature1', 'feature2']]  # What it does: Independent variables (what the model uses to learn).
# y = df['target_column']           # What it does: Dependent variable (what the model is trying to predict).

# Step 2: Split data into Training and Testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# What it does: 80% of data trains the model, 20% tests it. 'random_state' ensures your split is reproducible.

# Step 3: Initialize the Model
# model = LinearRegression()        # What it does: Creates an instance of your machine learning algorithm.

# Step 4: Fit (Train) the Model on training data
# model.fit(X_train, y_train)       # What it does: Trains the model by finding patterns between X_train and y_train.

# Step 5: Make Predictions and Evaluate
# predictions = model.predict(X_test)  # What it does: Generates predictions on the unseen testing data.