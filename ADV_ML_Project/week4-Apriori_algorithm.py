#import the required library
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

#Load and explore the data
data = pd.read_csv('Online Retail.csv')
print(data.head())
print(data.info())