#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import time
print("choose a store!")
store = input("select:\n1. Amazon\n2. BestBuy\n3. Nike\n4. walmart\n5. seabra\n")
if store== "6":
    quit()
    
data_list = ['Amazon','Bestbuy', 'Nike', 'Walmart', 'Seabra']
min_sup = int(input("enter the min support in percentage: "))
min_conf = int(input("enter the min confidence in percentage: "))
store = int(store)
if store <1 or store > len(data_list):
    print("enter a valid number")
    quit()
data = pd.read_csv("dataset_"+ data_list[store-1] + ".csv")
item = pd.read_csv(data_list[store-1] +".csv")
order = sorted(item['Item'])


# In[ ]:


dataset = data['Items'].str.split(',').apply(lambda items: [item.strip() for item in items]).tolist()
dataset


# In[ ]:


# Convert DataFrame to list of items
item_list = item['Item'].tolist()
item_list


# In[ ]:


dataset = data['Items'].str.split(',').apply(lambda items: [item.strip() for item in items]).tolist()


# In[ ]:


#brute force
from itertools import combinations
from collections import defaultdict

t_1 = time.time()
min_support = (min_sup/100)*20

# Count frequency of each itemset in transactions
def get_frequent_itemsets(dataset, itemset_size, min_support):
    
    itemset_counts = defaultdict(int)
    
    # Count each itemset's occurrences in the transactions
    for transaction in dataset:
        for itemset in combinations(transaction, itemset_size):
            itemset_counts[itemset] += 1

    # Filter itemsets by minimum support
    frequent_itemsets = {itemset: count for itemset, count in itemset_counts.items() if count >= min_support}

    return frequent_itemsets

# Generate all frequent itemsets
frequent_itemsets = {}
k = 1
while True:
    # Generate frequent k-itemsets
    current_frequent_itemsets = get_frequent_itemsets(dataset, k, min_support)
    if not current_frequent_itemsets:
        break
    frequent_itemsets.update(current_frequent_itemsets)
    k += 1




# Generate association rules
def generate_association_rules(frequent_itemsets, min_confidence=min_conf/100):
    rules = []
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        itemset_support = frequent_itemsets[itemset]
        
        # Generate all possible rules from the itemset
        for i in range(1, len(itemset)):
            for a in combinations(itemset, i):
                c = tuple(item for item in itemset if item not in a)
                if c:
                    a_support = frequent_itemsets.get(a, 0)
                    if a_support > 0:
                        confidence = itemset_support / a_support
                        if confidence >= min_confidence:
                            rules.append((a, c, confidence))

    return rules



print("Frequent Itemsets:")

for itemset, count in frequent_itemsets.items():
    print(f"{itemset}: {count}")
    


# In[ ]:


# Print association rules
association_rules = generate_association_rules(frequent_itemsets)
print("\nAssociation Rules:")
for a, c, confidence in association_rules:
    print(f"{a} -> {c} (Confidence: {confidence:.2f})")
b_t = time.time() - t_1


# In[ ]:


#apriori
get_ipython().system('pip install apriori_python')
get_ipython().system('pip install pyfpgrowth --upgrade')
from apriori_python.apriori import apriori

mins = min_sup/100
minc = min_conf/100
t_2 = time.time()
freqitemset, rules = apriori(dataset, minSup=mins, minConf = minc)
for i, rule in enumerate(rules):
    print(f"rule {i+1}: {rule}\n")
a_t = time.time() - t_2


# In[ ]:


#fp growth
import pyfpgrowth


# In[ ]:


transactions = dataset


# In[ ]:


t_3 = time.time()
patterns = pyfpgrowth.find_frequent_patterns(transactions, (min_sup/100)*20)
patterns
f_t = time.time() - t_3


# In[54]:


rules = pyfpgrowth.generate_association_rules(patterns, min_conf/100)
rules


# In[56]:


#time comparison of three algorithms
print("brute force time:/t")
print(b_t)
print("apriori time:/t")
print(a_t)
print("fp tree time:/t")
print(f_t)


# In[ ]:




