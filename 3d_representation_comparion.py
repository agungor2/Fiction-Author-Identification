# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 15:43:34 2018

@author: mgungor
3 dimensional word2vec tsne representation of training data
#To import xgboost properly use the following code
dir = r'C:\Program Files\mingw-w64\x86_64-7.2.0-posix-seh-rt_v5-rev0\mingw64\bin'
import os 
os.environ['PATH'].count(dir)
os.environ['PATH'].find(dir)
os.environ['PATH'] = dir + ';' + os.environ['PATH']
"""
import gensim
import numpy as np
import pandas as pd
import tqdm
import xgboost as xgb
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk import word_tokenize
from sklearn.metrics import f1_score, accuracy_score
import scipy.io as sc
test_author = sc.loadmat("test_author.mat")["test_author"]

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary = True)
model.init_sims()


train = pd.read_csv('train1.csv')
test = pd.read_csv('test1.csv')
ytrain = train.author.values


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

# create sentence vectors using the above function for training and validation set
xtrain_glove = [sent2vec(x) for x in tqdm(train.text.values)]
xtest_glove = [sent2vec(x) for x in tqdm(test.text.values)]

xtrain_glove = np.array(xtrain_glove)
xtest_glove = np.array(xtest_glove)


from sklearn.manifold import TSNE
n_components = 3
perplexities = 50
tsne_model = TSNE(n_components=n_components, init='random',
                         random_state=0, perplexity=perplexities)
X_vectors_tsne = tsne_model.fit_transform(xtrain_glove)

principalDf = pd.DataFrame(data = X_vectors_tsne
             , columns = ['tsne 1', 'tsne 2','tsne 3'])
author_id = pd.DataFrame(data = train_df.author.values, columns = ['author'])
finalDf = pd.concat([principalDf, author_id], axis = 1)

import pynamical
from pynamical import simulate, phase_diagram_3d
import pandas as pd, numpy as np, matplotlib.pyplot as plt, random, glob, os, IPython.display as IPdisplay
from PIL import Image
%matplotlib inline
title_font = pynamical.get_title_font()
label_font = pynamical.get_label_font()
save_folder = 'images/phase-animate'
import os
# set a filename, run the logistic model, and create the plot
gif_filename = '02-pan-rotate-logistic-phase-diagram'
working_folder = '{}/{}'.format(save_folder, gif_filename)
if not os.path.exists(working_folder):
    os.makedirs(working_folder)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize = (15,15))
ax = Axes3D(fig)
targets =np.unique(train_df.author.values)
for target in targets:
    indicesToKeep = finalDf['author'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'tsne 1']
               , finalDf.loc[indicesToKeep, 'tsne 2']
               , finalDf.loc[indicesToKeep, 'tsne 3'])
ax.set_title('Authorship Attribution', fontsize = 20)
ax.legend(targets)
ax.grid()

# look straight down at the x-y plane to start off
ax.elev = 89.9
ax.azim = 270.1
ax.dist = 11.0

# sweep the perspective down and rotate to reveal the 3-D structure of the strange attractor
for n in range(0, 100):
    if n > 19 and n < 23:
        ax.set_xlabel('')
        ax.set_ylabel('') #don't show axis labels while we move around, it looks weird
        ax.elev = ax.elev-0.5 #start by panning down slowly
    if n > 22 and n < 37:
        ax.elev = ax.elev-1.0 #pan down faster
    if n > 36 and n < 61:
        ax.elev = ax.elev-1.5
        ax.azim = ax.azim+1.1 #pan down faster and start to rotate
    if n > 60 and n < 65:
        ax.elev = ax.elev-1.0
        ax.azim = ax.azim+1.1 #pan down slower and rotate same speed
    if n > 64 and n < 74:
        ax.elev = ax.elev-0.5
        ax.azim = ax.azim+1.1 #pan down slowly and rotate same speed
    if n > 73 and n < 77:
        ax.elev = ax.elev-0.2
        ax.azim = ax.azim+0.5 #end by panning/rotating slowly to stopping position   
    if n > 76: #add axis labels at the end, when the plot isn't moving around
        ax.set_xlabel('tsne 1', fontsize = 15)
        ax.set_ylabel('tsne 2', fontsize = 15)
        ax.set_zlabel('tsne 3', fontsize = 15)

    # add a figure title to each plot then save the figure to the disk
    #fig.suptitle('Benchmarking Authorship Attribution', fontsize=16, x=0.5, y=0.85)
    plt.savefig('{}/{}/img{:03d}.png'.format(save_folder, gif_filename, n), bbox_inches='tight')

# don't display the static plot
plt.close()

# load all the static images into a list then save as an animated gif
gif_filepath = '{}/{}.gif'.format(save_folder, gif_filename)
images = [Image.open(image) for image in glob.glob('{}/*.png'.format(working_folder))]
gif = images[0]
gif.info['duration'] = 10 #milliseconds per frame
gif.info['loop'] = 0 #how many times to loop (0=infinite)
gif.save(fp=gif_filepath, format='gif', save_all=True, append_images=images[1:])
IPdisplay.Image(url=gif_filepath)