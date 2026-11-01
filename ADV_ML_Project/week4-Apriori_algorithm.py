#import the required library
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

#Load and explore the data
data = pd.read_csv('Online Retail.csv')
print(data.head())
print(data.info())

#exploring the columns of data
print(data.columns)
#explore different regions of transactions
print(data.Country.unique())

# Count unique transactions (number of transactions)
num_transactions = data['InvoiceNo'].nunique()
print("number of transactions:", num_transactions)

# Count unique products (number of products)
num_products = data['StockCode'].nunique()
print("number of products:", num_products)

#step3 cleaning the data
# Stripping extra spaces in the description eg "  mug. " to "mug" to prevent python from treating them as two completely different products
data['Description'] = data['Description'].str.strip()

# Dropping the rows without any invoice number, because without an invoice number we cant tie items to a shopping cart
data.dropna(axis = 0, subset =['InvoiceNo'], inplace = True)

# Convert invoice numbers to string, this prepares them for filtering .
data['InvoiceNo'] = data['InvoiceNo'].astype('str')

# Dropping all transactions which were done on credit
data = data[~data['InvoiceNo'].str.contains('C')]
#The tilde (~) means "NOT". This code filters out any invoice number that contains the letter 'C'. In the Online Retail dataset, an invoice starting with 'C' means a cancellation or return (e.g., a customer returned items, resulting in negative quantities).
# We drop them because we want to analyze what people buy, not what they returned.

#step4 splitting the data according to the region of transaction and creating market basket matrix
# Transactions done in France
basket_France = (data[data['Country'] =="France"]
   .groupby(['InvoiceNo', 'Description'])['Quantity']
   .sum().unstack().reset_index().fillna(0)
   .set_index('InvoiceNo'))
print(basket_France.shape)

# Transactions done in the United Kingdom
basket_UK = (data[data['Country'] =="United Kingdom"]
   .groupby(['InvoiceNo', 'Description'])['Quantity']
   .sum().unstack().reset_index().fillna(0)
   .set_index('InvoiceNo'))
print(basket_UK.shape)


# Transactions done in Portugal
basket_Por = (data[data['Country'] =="Portugal"]
.groupby(['InvoiceNo', 'Description'])['Quantity']
.sum().unstack().reset_index().fillna(0)
.set_index('InvoiceNo'))

print(basket_Por.shape)

# Transactions done in Sweden

basket_Sweden = (data[data['Country'] =="Sweden"]
.groupby(['InvoiceNo', 'Description'])['Quantity']
.sum().unstack().reset_index().fillna(0)
.set_index('InvoiceNo'))

print(basket_Sweden.shape)

#hot encoding
# Defining the hot encoding function to make the data suitable for the concerned libraries
def hot_encode(x):
    if (x <= 0):
        return 0
    if (x >= 1):
        return 1

# Encoding the datasets for each country basket
basket_encoded = basket_France.map(hot_encode)

basket_France = basket_encoded
#What it does: Applies that function across the entire France table (applymap). Every quantity gets squashed into a clean binary 1 or 0 matrix.

#for the other countries
basket_UK = basket_UK.map(hot_encode)
basket_Por = basket_Por.map(hot_encode)
basket_Sweden = basket_Sweden.map(hot_encode)

#step 6 building the apriori model and analysing the results
#This is where the actual machine learning happens.
# Building the model for France
frq_items = apriori(basket_France, min_support=0.05, use_colnames=True)
#What it does: Scans the binary basket matrix to find combinations of items that appear together at or above your minimum support threshold.
#min_support = 0.05: Means an item or itemset must appear in at least 5% of all transactions to even be considered.

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)
#What it does: Takes those frequent itemsets and builds actionable rules (e.g., If X, then Y).
#metric="lift", min_threshold=1: Filters the rules to only keep ones where the Lift is greater than 1. As we discussed earlier, a lift > 1 means the items actually drive sales of each other and aren't just matching by random coincidence

# Sort and display strongest rules
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head(10))
#Sorts the final rules so the highest confidence (probability of Y given X) and highest lift appear right at the top.
#print(rules.head(10)) prints the top 10 strongest rules so you can look at them and interpret the business value.

#uk
frq_items = apriori(basket_UK, min_support=0.01, use_colnames=True)
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)

# Sort the rules by confidence and lift
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

# Print the top strongest rules
print(rules.head(5))

#portugal
# Force pandas to show all columns and avoid truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

# Portugal Association Rules
frq_items_por = apriori(basket_Por, min_support=0.05, use_colnames=True)
rules_por = association_rules(frq_items_por, metric="lift", min_threshold=1)
rules_por = rules_por.sort_values(['confidence', 'lift'], ascending=[False, False])

print("\n--- PORTUGAL TOP RULES ---")
print(rules_por.head(5))