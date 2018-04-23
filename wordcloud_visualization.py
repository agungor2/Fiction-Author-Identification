# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 19:46:10 2018

@author: mgungor
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
import pandas as pd

train_df = pd.read_csv("train.csv")

# Read the whole text.
text = ''
for element in train_df[train_df.author=="EAP"].text.values:
    text = text + element

horse_mask = np.array(Image.open('raven.jpg'))

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=100, mask=horse_mask,
               stopwords=stopwords)
# generate word cloud
wc.generate(text)

# store to file
wc.to_file("raven_wordcloud.png")

# change the figure size
fig2 = plt.figure(figsize = (15,15)) # create a 20 * 20  figure 
ax3 = fig2.add_subplot(111)
ax3.imshow(wc, interpolation='bilinear')
ax3.axis("off")

#########################################################################################
#Same steps for MWS

text = ''
for element in train_df[train_df.author=="MWS"].text.values:
    text = text + element

horse_mask = np.array(Image.open('frankenstein.jpg'))

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=100, mask=horse_mask,
               stopwords=stopwords)
# generate word cloud
wc.generate(text)

# store to file
wc.to_file("frankenstein_wordcloud.png")

# change the figure size
fig2 = plt.figure(figsize = (15,15)) # create a 20 * 20  figure 
ax3 = fig2.add_subplot(111)
ax3.imshow(wc, interpolation='bilinear')
ax3.axis("off")

#########################################################################################
#Same steps for MWS

text = ''
for element in train_df[train_df.author=="HPL"].text.values:
    text = text + element

horse_mask = np.array(Image.open('chuthulhu.png'))

stopwords = set(STOPWORDS)
stopwords.add("said")

wc = WordCloud(background_color="white", max_words=100, mask=horse_mask,
               stopwords=stopwords)
# generate word cloud
wc.generate(text)

# store to file
wc.to_file("chuthulhu_wordcloud.png")

# change the figure size
fig2 = plt.figure(figsize = (15,15)) # create a 20 * 20  figure 
ax3 = fig2.add_subplot(111)
ax3.imshow(wc, interpolation='bilinear')
ax3.axis("off")
