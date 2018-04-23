# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 13:35:05 2018

@author: mgungor
Ngram visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import nltk
from nltk.util import ngrams
%matplotlib inline

train = pd.read_csv("train.csv")

def generate_ngrams(text, n=2):
    words = text.split()
    iterations = len(words) - n + 1
    for i in range(iterations):
       yield words[i:i + n]
      
        
# DataFrame for Mary Shelley
ngrams = {}
for title in train[train.author=="MWS"]['text']:
        for ngram in generate_ngrams(title, 3):
            ngram = ' '.join(ngram)
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1

ngrams_mws_df = pd.DataFrame.from_dict(ngrams, orient='index')
ngrams_mws_df.columns = ['count']
ngrams_mws_df['author'] = 'Mary Shelley'
ngrams_mws_df.reset_index(level=0, inplace=True)

# DataFrame for Edgar Allen Poe
ngrams = {}
for title in train[train.author=="EAP"]['text']:
        for ngram in generate_ngrams(title, 3):
            ngram = ' '.join(ngram)
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1

ngrams_eap_df = pd.DataFrame.from_dict(ngrams, orient='index')
ngrams_eap_df.columns = ['count']
ngrams_eap_df['author'] = 'Edgar Allen Poe'
ngrams_eap_df.reset_index(level=0, inplace=True)

# DataFrame for HP Lovecraft
ngrams = {}
for title in train[train.author=="HPL"]['text']:
        for ngram in generate_ngrams(title, 3):
            ngram = ' '.join(ngram)
            if ngram in ngrams:
                ngrams[ngram] += 1
            else:
                ngrams[ngram] = 1

ngrams_hpl_df = pd.DataFrame.from_dict(ngrams, orient='index')
ngrams_hpl_df.columns = ['count']
ngrams_hpl_df['author'] = 'HP lovecraft'
ngrams_hpl_df.reset_index(level=0, inplace=True)

print(ngrams_eap_df.sort_values(by='count', ascending=False).head(5))
print(ngrams_hpl_df.sort_values(by='count', ascending=False).head(5))
print(ngrams_mws_df.sort_values(by='count', ascending=False).head(5))

trigram_df = pd.concat([ngrams_eap_df.sort_values(by='count', ascending=False).head(25),
                        ngrams_hpl_df.sort_values(by='count', ascending=False).head(25),
                        ngrams_mws_df.sort_values(by='count', ascending=False).head(25)])
    
g = nx.from_pandas_edgelist(trigram_df,source='author',target='index',edge_attr=True)
print(nx.info(g))

plt.figure(figsize=(20, 20))
cmap = plt.cm.coolwarm
colors = [n for n in range(len(g.nodes()))]
#k = 0.0319
k = 0.14
pos=nx.spring_layout(g, k=k)
nx.draw_networkx(g,pos, node_size=trigram_df['count'].values*150, cmap = cmap, 
                 node_color=colors, edge_color='grey', font_size=15, width=2, alpha=1)
plt.title("Network diagram of Top 20 Trigrams",
         fontsize=18)
plt.show()