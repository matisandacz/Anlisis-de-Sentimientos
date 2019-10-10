"""Clasifica el conjunto de entrada usando el mejor clasificador encontrado

Nota: Asumimos que el .so generado (sentiment.cpython....so) está en la carpeta
`notebooks/`

"""
# Estas dos líneas permiten que python encuentre la librería sentiment en notebooks/
import sys
sys.path.append("../notebooks/")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentiment import PCA, KNNClassifier
from nltk.corpus import stopwords
import numpy as np

def agregar_negaciones(text_train):
    negaciones = ["no","nor","not","none","without","never","nobody","never","barely","hardly","seldom","less","little","rarely","scarcely"]
    #ademas considero las palabras que terminan en n't
    finalizadores = [".",",",";","<br/>","<br>"]
    for i in range(len(text_train)):
        text_train[i] = "<br/>".join(text_train[i].split("<br />"))
        por_palabras = text_train[i].split(" ")
        for num_palabra in range(len(por_palabras)):
            if por_palabras[num_palabra] in negaciones or "n't" in por_palabras[num_palabra]:
                j = 1
                while num_palabra+j < len(por_palabras) and por_palabras[num_palabra+j] not in finalizadores and por_palabras[num_palabra+j][-1] not in finalizadores:
                    por_palabras[num_palabra+j] = "_NOT"+por_palabras[num_palabra+j]
                    j += 1
        text_train[i] = " ".join(por_palabras)
    return text_train

def vectorizar(text_train, BINARIO, IDF, NEGACIONES, STOP_WORDS):
    if STOP_WORDS:
        stopwords_nuestras = stopwords.words('english')
        stopwords_nuestras.append("br")
        vectorizer = TfidfVectorizer(ngram_range = (1,3), binary = BINARIO, strip_accents = 'unicode', stop_words = stopwords_nuestras, use_idf = IDF, smooth_idf = IDF, max_df = 0.9, min_df = 0.001, max_features = 5000)
    else:
        vectorizer = TfidfVectorizer(ngram_range = (1,3), binary = BINARIO, strip_accents = 'unicode', use_idf = IDF, smooth_idf = IDF, max_df = 0.9, min_df = 0.001, max_features = 5000)
    if NEGACIONES:
        text_train = agregar_negaciones(text_train)
    vectorizer.fit(text_train)
    X_train = vectorizer.transform(text_train)
    return X_train

def get_instances(df, SIZE_TRAIN):
    text_train = np.array(df[df.type == 'train']["review"].tolist())
    orden = np.arange(len(text_train))
    np.random.shuffle(orden)
    text_train = text_train[orden[:SIZE_TRAIN]]

    return text_train

if __name__ == '__main__':
    #si caso == 1 calculo el alpha necesario para que el porcentaje de informacion sea P
    #si caso == 0 calculo el P dado alpha
    caso = sys.argv[1]
    if caso == "1":
        P = float(sys.argv[2])
    elif caso == "0":
        ALPHA = float(sys.argv[2])
    else:
        print("Parametro invalido.")
        exit()

    TRAIN_SIZE = 6225
    NEGACIONES = True
    BINARIO = True
    NORMA_PESADA = True
    STOP_WORDS = True
    IDF = True

    df_train = pd.read_csv("../data/imdb_small.csv")

    text_train = get_instances(df_train,TRAIN_SIZE)
    X_train = vectorizar(text_train, BINARIO, IDF, NEGACIONES, STOP_WORDS).todense()

    if caso == "0":
        var_total = np.std(X_train,axis = 1).sum()
        pca = PCA(alpha)
        pca.fit(X_train)
        X_train = pca.transform(X_train, alpha)
        var_con_pca = np.std(X_train,axis = 1).sum()
    else:
        var_total = np.std(X_train,axis = 1).sum()
        pca = PCA(X_train.shape[1])
        pca.fit(X_train)
        X_train = pca.transform(X_train, X_train.shape[1])
        var_parcial = np.std(X_train[:,0])
        for i in range(1,X_train.shape[1]):
            if var_parcial/var_total > P:
                print(i)
                break
            var_parcial = var_parcial+np.std(X_train[:,i])
