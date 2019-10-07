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

def get_instances(df, df_test):
    """
    Lee instancias de entrenamiento y de test
    """
    text_train = df[df.type == 'train']["review"]
    label_train = df[df.type == 'train']["label"]

    text_test = df_test["review"]
    ids_test = df_test["id"]

    print("Cantidad de instancias de entrenamiento = {}".format(len(text_train)))
    print("Cantidad de instancias de test = {}".format(len(text_test)))

    stopwords_nuestras = stopwords.words('english')
    stopwords_nuestras.append("br")

    vectorizer = TfidfVectorizer(ngram_range = (1,3), binary = True, strip_accents = 'unicode', stop_words = stopwords_nuestras, use_idf = True, smooth_idf = True, max_df = 0.9, min_df = 0.001, max_features = 5000)

    vectorizer.fit(text_train)

    X_train, y_train = vectorizer.transform(text_train), (label_train == 'pos').values

    X_test = vectorizer.transform(text_test)

    return X_train, y_train, X_test, ids_test

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: python classify archivo_de_test archivo_salida")
        exit()

    test_path = sys.argv[1]
    out_path = sys.argv[2]

    df = pd.read_csv("../data/imdb_small.csv")
    df_test = pd.read_csv(test_path)

    print("Vectorizando datos...")
    X_train, y_train, X_test, ids_test = get_instances(df, df_test)

    ALPHA = 100
    pca = PCA(ALPHA)

    print("Entrenando PCA")
    pca.fit(X_train.toarray())
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    """
    Entrenamos KNN
    """
    clf = KNNClassifier(100)

    clf.fit(X_train, y_train)

    """
    Testeamos
    """
    print("Prediciendo etiquetas...")

    NORMA_PESADA = True

    if not NORMA_PESADA:
        y_pred = clf.predict(X_test)
    else:
        y_train_norm = y_train - y_train.mean()
        ystd = np.std(y_train)
        print(X_train.shape)

        covarianzas = np.zeros(ALPHA)
        correlaciones = np.zeros(ALPHA)
        for i in range(ALPHA):
            covarianzas[i] = ((X_train[:,i]-X_train[:,i].mean())*y_train_norm).sum()
            correlaciones[i] = covarianzas[i]/(np.std(X_train[:,i])*ystd)
        y_pred = clf.predictC(X_test,covarianzas)

    y_pred = clf.predict(X_test).reshape(-1)
    # Convierto a 'pos' o 'neg'
    labels = ['pos' if val == 1 else 'neg' for val in y_pred]

    df_out = pd.DataFrame({"id": ids_test, "label": labels})

    df_out.to_csv(out_path, index=False)

    print("Salida guardada en {}".format(out_path))
