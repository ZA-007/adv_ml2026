"""""
Part 1: What is the Apriori Algorithm & How Does It Work?
What it is: The Apriori algorithm is an unsupervised learning technique used for Association Rule Mining (often called Market Basket Analysis).

The Goal: It scans through transaction records (like receipts or shopping carts) to find hidden patterns and rules of the format: "If a customer buys Item X, they are also likely to buy Item Y."

The 3 Core Metrics You Need to Know:
Support: How popular/frequent an item (or group of items) is across all transactions.

Example: If 100 out of 1,000 receipts contain milk, the support for milk is 10%.

Confidence: How often the rule has been found to be true. It measures the probability of buying Y given that X was bought.

Example: Out of the 100 people who bought milk, 80 of them also bought bread. The confidence is 80%.

Lift: How much stronger the association is than expected by pure chance.

Lift > 1 means X and Y drive sales of each other (a strong, meaningful association).

Lift = 1 means X and Y are completely independent.

Lift < 1 means they substitute each other (people who buy X usually don't buy Y).

Part 2: When to Use It in an Exam? (The Problem Decider)
If you get a case study in your exam, how do you know you are supposed to use Apriori instead of Clustering or Classification?

Look for the Clues in the Case Study:

Words like: "Transaction logs," "supermarket receipts," "e-commerce basket data," "cross-selling opportunities," "what products are purchased together."

Look at the Dataset Structure:

Unlike clustering (which has numerical measurements like Age and Income), an association rule dataset looks like a transaction list. It has columns like an Invoice/Transaction ID, a Stock Code, and a Product Description. Multiple rows will share the exact same Invoice ID because a single customer bought multiple items in one trip.
Step 5: Hot Encoding the Data
Why do we need hot-encoding?
Right now, our basket tables (basket_France, basket_UK, etc.) contain raw numbers—how many quantities of an item a customer bought (e.g., a customer bought 3 mugs and 1 bag).
The Apriori algorithm doesn't care how many of an item were bought; it only cares whether the item was present in the basket or not (Yes/No, True/False, or 1/0). Hot encoding converts all quantities greater than 0 into a simple 1, and 0 otherwise.


# ==========================================
# UNIVERSAL APRIORI EXAM TEMPLATE
# ==========================================
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Load Data
data = pd.read_csv('exam_dataset.csv')

# 2. Clean Data (Drop missing values for your ID or Item columns)
data.dropna(axis=0, subset=['YOUR_GROUP_ID_COL', 'YOUR_ITEM_COL'], inplace=True)
data['YOUR_GROUP_ID_COL'] = data['YOUR_GROUP_ID_COL'].astype('str')

# 3. Create the Matrix (Group by your Grouping ID and Item column)
# NOTE: If there is NO country/region filter in the exam,
# just delete the data[data['Country'] == ...] part and start straight at .groupby()!
basket = (data
          .groupby(['YOUR_GROUP_ID_COL', 'YOUR_ITEM_COL'])['SOME_NUMERICAL_COL_OR_QUANTITY']
          .sum().unstack().reset_index().fillna(0)
          .set_index('YOUR_GROUP_ID_COL'))

# 4. Hot Encode (Quantities or counts to 1/0)
def hot_encode(x):
    if x <= 0: return 0
    if x >= 1: return 1

basket_final = basket.map(hot_encode)

# 5. Run Apriori & Rules
frq_items = apriori(basket_final, min_support=0.05, use_colnames=True)
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

# 6. View Results
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))




Part 1: Decoding the "Group ID", "Item", and "Quantity"
Think of a dataset as a giant spreadsheet of receipts or transactions. To do Market Basket Analysis, the Apriori algorithm needs to look at the data and answer two basic questions: "Who bought this?" and "What did they buy?"

1. The Group ID (InvoiceNo in our retail example)
What it is: The unique identifier for a single "event," "visit," or "basket."

In different contexts:

In a supermarket: It’s the Receipt / Invoice Number (groups all items bought on one single trip).

On a website: It’s the User ID or Session ID (groups all the web pages a person clicked during one visit).

In a hospital: It’s the Patient ID or Visit ID (groups all symptoms recorded during one doctor visit).

Does position matter? Yes! In your .groupby([...]) code, the Group ID must always go first, because you are grouping all items inside that specific basket.

2. The Item (Description in our retail example)
What it is: The actual product, object, or event that was chosen inside that group.

In different contexts:

In a supermarket: The Product Name (e.g., "Paper Cups").

On a website: The Page Name (e.g., "Checkout Page").

In a streaming app: The Movie Title (e.g., "Inception").

Does position matter? Yes! It must go second right after your Group ID in the .groupby([ 'Group_ID', 'Item_Column' ]) command.

3. The Numerical Column / Quantity (Quantity in our retail example)
What it is: A number representing how many of that item were bought, or simply a count.

Why it's there: When you group items by invoice, the code needs a numerical column to add up (e.g., .sum()).

What if your dataset doesn't have a quantity number? (For example, website click logs where every row just means "the user clicked this page once").

If your data doesn't have a quantity column, you can actually drop .sum() and use .size().unstack()... instead, but in almost all university machine learning datasets (like Online Retail), there is a quantity or a dummy count column. For the exam, just look at your CSV columns and pick whichever column holds the amounts or counts.

Part 2: Your Exam Cheat Sheet for Metric Definitions & Report Writing
If the exam asks you to write a report interpreting the results, memorize these definitions. You can copy-paste these exact explanations into your report structure to sound like an expert data scientist:

1. Antecedent (The "IF" part)
Definition: The product or item found on the left-hand side of the rule (what the customer already has).

What it shows in context: It acts as the trigger or the primary purchase. For example: "If a customer buys Paper Cups..."

2. Consequent (The "THEN" part)
Definition: The product or item found on the right-hand side of the rule (what is recommended next).

What it shows in context: It is the item frequently bought alongside or immediately after the antecedent. For example: "...then they are also likely to buy Paper Plates."

3. Support
Definition: The statistical popularity of an item or item combination across all transactions in the dataset (expressed as a decimal or percentage).

Formula: (Transactions containing both A and B) / (Total transactions).

How to write it in your report: "A support of 0.05 means that this specific combination of items appears together in 5% of all customer shopping trips. This proves it is a common behavior, not a rare fluke."


"""


