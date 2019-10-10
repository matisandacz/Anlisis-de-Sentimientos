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

def vectorizar(text_train, label_train, text_test, BINARIO, IDF, NEGACIONES, STOP_WORDS):
    if STOP_WORDS:
        stopwords_nuestras = stopwords.words('english')
        stopwords_nuestras.append("br")
        vectorizer = TfidfVectorizer(ngram_range = (1,3), binary = BINARIO, strip_accents = 'unicode', stop_words = stopwords_nuestras, use_idf = IDF, smooth_idf = IDF, max_df = 0.9, min_df = 0.001, max_features = 5000)
    else:
        vectorizer = TfidfVectorizer(ngram_range = (1,3), binary = BINARIO, strip_accents = 'unicode', use_idf = IDF, smooth_idf = IDF, max_df = 0.9, min_df = 0.001, max_features = 5000)
    if NEGACIONES:
        text_train = agregar_negaciones(text_train)
        text_test = agregar_negaciones(text_test)
    vectorizer.fit(text_train)
    X_train, y_train = vectorizer.transform(text_train), np.array(label_train)
    X_test = vectorizer.transform(text_test)
    print(X_train.shape)
    return X_train, y_train, X_test

def get_instances(df, df_test, SIZE_TRAIN):
    text_train = np.array(df[df.type == 'train']["review"].tolist())
    label_train = np.array(df[df.type == 'train']["label"].tolist())
    orden = np.arange(len(text_train))
    np.random.shuffle(orden)
    text_train = text_train[orden[:SIZE_TRAIN]]
    label_train = label_train[orden[:SIZE_TRAIN]]

    text_test = df_test["review"].tolist() #tomamos un conjunto de test al azar
    ids_test = df_test["id"]

    print("Cantidad de instancias de entrenamiento = {}".format(len(text_train)))
    print("Cantidad de instancias de test = {}".format(len(text_test)))
    return text_train, label_train, text_test, ids_test

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: python classify archivo_de_test archivo_salida")
        exit()

    ALPHA = None
    K = 100
    TRAIN_SIZE = 50000
    NEGACIONES = True
    BINARIO = True
    NORMA_PESADA = True
    STOP_WORDS = True
    IDF = True

    test_path = sys.argv[1]
    out_path = sys.argv[2]

    df_train = pd.read_csv("../data/imdb_large.csv")
    df_test = pd.read_csv(test_path)

    print("Vectorizando datos...")
    text_train, label_train, text_test, ids_test = get_instances(df_train,df_test,TRAIN_SIZE)
    X_train, y_train, X_test = vectorizar(text_train, label_train, text_test, BINARIO, IDF, NEGACIONES, STOP_WORDS)

    if ALPHA != None:
        print("Obteniendo componentes principales...")
        pca = PCA(ALPHA)
        pca.fit(X_train.todense())
        X_train = pca.transform(X_train, ALPHA)
        X_test = pca.transform(X_test, ALPHA)

    clf = KNNClassifier(K)

    clf.fit(X_train, y_train)
    print("Prediciendo...")
    print(X_test.shape)
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

    # Convierto a 'pos' o 'neg'
    labels = ['pos' if val == 1 else 'neg' for val in y_pred]

    df_out = pd.DataFrame({"id": ids_test, "label": labels})

    df_out.to_csv(out_path, index=False)

    print("Salida guardada en {}".format(out_path))
