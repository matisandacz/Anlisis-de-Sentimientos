
#Nota: Asumimos que el .so generado (sentiment.cpython....so) está en la carpeta
#`notebooks/`

# Estas dos líneas permiten que python encuentre la librería sentiment en notebooks/
import sys
sys.path.append("../notebooks/")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentiment import PCA, KNNClassifier
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics import accuracy_score
import pickle

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

def vectorizar(text_train, label_train, text_test, label_test, BINARIO, IDF, NEGACIONES):
    vectorizer = TfidfVectorizer(ngram_range = (1,3), binary = BINARIO, strip_accents = 'unicode', use_idf = IDF, smooth_idf = IDF, max_df = 0.9, min_df = 0.001, max_features = 5000)
    if NEGACIONES:
        text_train = agregar_negaciones(text_train)
    vectorizer.fit(text_train)
    X_train, y_train = vectorizer.transform(text_train), np.array(label_train)
    X_test, y_test = vectorizer.transform(text_test), np.array(label_test)
    return X_train, y_train, X_test, y_test

def get_instances(df, SIZE_TRAIN, SIZE_TEST):
    text_train = df[df.type == 'train']["review"].tolist()[:SIZE_TRAIN]
    label_train = df[df.type == 'train']["label"].tolist()[:SIZE_TRAIN]

    text_test = df[df.type == 'test']["review"].tolist() #tomamos un conjunto de test al azar
    label_test = df[df.type == 'test']["label"].tolist()
    indices_test = np.arange(0,len(text_test))
    np.random.shuffle(indices_test)
    indices_test = np.array(indices_test)
    text_test = np.array(text_test)[indices_test[:SIZE_TEST]]
    label_test = np.array(label_test)[indices_test[:SIZE_TEST]]

    print("Cantidad de instancias de entrenamiento = {}".format(len(text_train)))
    print("Cantidad de instancias de test = {}".format(len(text_test)))
    return text_train, label_train, text_test, label_test

def run_test(df,TRAIN_SIZE = 6225,TEST_SIZE = 500,ALPHA = None,K = None,BINARIO = False, NEGACIONES = False, NORMA_PESADA = False, IDF = False):
    print("--------------------------------------------------------")
    print("Test empezado con:")
    print("Train size:",TRAIN_SIZE)
    print("Test size:",TEST_SIZE)
    print("Alpha:",ALPHA)
    print("K:",K)
    print("Negaciones:",NEGACIONES)
    print("Binario:",BINARIO)
    print("Norma pesada:",NORMA_PESADA)
    print("IDF:",IDF)
    text_train, label_train, text_test, label_test = get_instances(df,TRAIN_SIZE,TEST_SIZE)

    print("Vectorizando...")
    X_train, y_train, X_test, y_test = vectorizar(text_train, label_train, text_test, label_test, BINARIO, IDF, NEGACIONES)
    if ALPHA != None:
        print("Obteniendo componentes principales...")
        pca = PCA(ALPHA)
        pca.fit(X_train.todense())
        X_train = pca.transform(X_train, ALPHA)
        X_test = pca.transform(X_test, ALPHA)

    clf = KNNClassifier(K)

    clf.fit(X_train, y_train)
    print("Prediciendo...")
    if not NORMA_PESADA:
        y_pred = clf.predict(X_test)
    else:
        y_train_norm = y_train - y_train.mean()
        ystd = np.std(y_train)

        covarianzas = np.zeros(X_train.shape[1])
        correlaciones = np.zeros(X_train.shape[1])
        for i in range(X_train.shape[1]):
            if ALPHA:
                covarianzas[i] = ((X_train[:,i]-X_train[:,i].mean())*y_train_norm).sum()
                correlaciones[i] = covarianzas[i]/(np.std(X_train[:,i])*ystd)
            else:
                covarianzas[i] = (((X_train.todense())[:,i]-X_train[:,i].mean())*y_train_norm).sum()
                correlaciones[i] = covarianzas[i]/(np.std((X_train.todense())[:,i])*ystd)
        y_pred = clf.predict_weighted(X_test,np.abs(correlaciones))
    print("Test finalizado")
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: {}".format(acc))
    return acc

if __name__ == '__main__':

    #leo el dataset
    df = pd.read_csv("../data/imdb_small.csv")
    df['label'] = (df['label'] == 'pos').astype('int')

    acc_binario = []
    acc_negaciones = []
    acc_norma_pesada = []
    acc_idf = []

    acc_binario.append(run_test(df,K = 150,BINARIO = True))
    acc_negaciones.append(run_test(df,K = 150,NEGACIONES = True))
    acc_norma_pesada.append(run_test(df,K = 150,NORMA_PESADA = True))
    acc_idf.append(run_test(df,K = 150,IDF = True))
    for ALPHA in [100,400]:
        for K in [50, 300]:
            acc_binario.append(run_test(df,ALPHA = ALPHA,K = K,BINARIO = True))
            acc_negaciones.append(run_test(df,ALPHA = ALPHA,K = K,NEGACIONES = True))
            acc_norma_pesada.append(run_test(df,ALPHA = ALPHA,K = K,NORMA_PESADA = True))
            acc_idf.append(run_test(df,ALPHA = ALPHA,K = K,IDF = True))

    fout = open("resultados_exp1.pkl","wb")
    pickle.dump((acc_binario,acc_negaciones,acc_norma_pesada,acc_idf),fout)
    fout.close()
