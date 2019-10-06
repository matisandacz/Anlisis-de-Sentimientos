import pandas as pd
import os
import numpy as np
import pickle
import sys

FOUT = open(sys.argv[1],"wb")

USAR_IMDB_LARGE = True
DATA_SIZE = {}
DATA_SIZE["test"] = 0
DATA_SIZE["train"] = 25000

if not USAR_IMDB_LARGE:
    df = pd.read_csv("../data/imdb_small.csv", index_col=0)
    df = df[['review','label']]
    df['label'] = (df['label'] == 'pos').astype('int')
else:
    folder = '../data/aclImdb'
    labels = {'pos': 1, 'neg': 0}
    df = pd.DataFrame()
    for f in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join(folder, f, l)
            c = 0
            for file in os.listdir (path) :
                with open(os.path.join(path, file),'r', encoding='utf-8') as infile:
                    txt = infile.read()
                df = df.append([[txt, labels[l]]],ignore_index=True)
                c+=1
                if c == DATA_SIZE[f]//2:
                    break
    df.columns = ['review', 'label']

X = np.array(df['review'].tolist())
y = np.array(df['label'].tolist())
orden = np.arange(X.shape[0])
np.random.shuffle(orden)
X = X[orden]
y = y[orden]


X_train = X[:DATA_SIZE["train"]]
y_train = y[:DATA_SIZE["train"]]

print("Class balance : {} pos {} neg".format(
    y_train.sum() / y_train.shape[0],
    (y_train.shape[0] - y_train.sum()) / y_train.shape[0]
))

pickle.dump((X_train,y_train),FOUT)
FOUT.close()
