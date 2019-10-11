#Modulos para implementar el algoritmo de clasificacion / algebra lineal.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

import nltk
from nltk.tokenize import punkt
from nltk.corpus import stopwords, words, wordnet



df = pd.read_csv("data/imdb_small.csv", index_col=0)

print("Cantidad de documentos: {}".format(df.shape[0]))

text_train = df[df.type == 'train']["review"]
label_train = df[df.type == 'train']["label"]

text_test = df[df.type == 'test']["review"]
label_test = df[df.type == 'test']["label"]

print("Cantidad de instancias de entrenamiento = {}".format(len(text_train)))
print("Cantidad de instancias de test = {}".format(len(text_test)))


vectorizer = CountVectorizer(max_df=0.90, min_df=0.01, max_features=5000)

vectorizer.fit(text_train)

X_train, y_train = vectorizer.transform(text_train), (label_train == 'pos').values
X_test, y_test = vectorizer.transform(text_test), (label_test == 'pos').values
y_train = y_train.astype(int)
y_test = y_test.astype(int)

X_train = X_train.todense()
X_test = X_test.todense()



print X_train.shape

print("Cantidad de instancias de entrenamiento = {}".format(len(text_train)))
print("Cantidad de instancias de test = {}".format(len(text_test)))


componentes = 20
print "Reduciendo  componentes con PCA"
pca = PCA(n_components=componentes)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print X_train.shape


def distancia_a_voto(distancia):
    return 2.0**(-distancia)



fibo = np.zeros(300)
for x in xrange(300):
    if x == 0 or x == 1:
        fibo[x] = 1
    else:
        fibo[x] = fibo[x-1] + fibo[x-2]

class KNNClassifier:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.n_clases = 2

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.weight = np.concatenate([np.ones(self.n_neighbors, dtype = float), np.zeros(X.shape[0] - self.n_neighbors, dtype = float)])
        self.weight = self.weight/self.n_neighbors
        print(self.weight)
        
        
    def _predict_row(self, row):


        """
        Calculamos distancias
        """
        dist = np.linalg.norm(self.X - row, axis = 1 , ord = 1)
        """
        Obtener vecinos mas cercanos
        """
        vecinos_mas_cercanos = np.argsort(dist)
        labels_mas_cercanos = self.y[vecinos_mas_cercanos]        
        
        """
        Decidimos por votacion
        """
        
        """
        Esto ciclo supone que las clases tiene label 0 , 1, 2 , 3...
        """
        vector_votacion = np.zeros(self.n_clases)
        vecinoLabel = 0
        i = 0


        while vecinoLabel < self.n_neighbors:
            if labels_mas_cercanos[i] == 0:
                distancia = dist[vecinos_mas_cercanos[i]]
                voto = fibo[self.n_neighbors - vecinoLabel - 1]*distancia
                voto = distancia
                vector_votacion[0] += voto
                vecinoLabel+=1
            i+=1



        vecinoLabel = 0
        i = 0


        while vecinoLabel < self.n_neighbors:
            if labels_mas_cercanos[i] == 1:
                distancia = dist[vecinos_mas_cercanos[i]]
                voto = fibo[self.n_neighbors - vecinoLabel - 1]*distancia
                voto = distancia
                vector_votacion[1] += voto
                vecinoLabel+=1
            i+=1
        
        """
        Devolvemos el conponente mas granvector_votacionde
        
        """

        if vector_votacion[0]<vector_votacion[1]:
            return 0
        else:
            return 1
    
        
    def optimize_weights(self, n_epochs):
        batch = 100
        delta = 0.5
        
        for epoch in range(n_epochs):
            X_core, X_train, y_core, y_train = train_test_split(self.X, self.y,test_size = batch)
            
            gradiente_promedio = np.zeros(self.weight.shape[0])
            
            for sample in range(X_train.shape[0]):
                row = X_train[sample]
                dist = np.linalg.norm(X_core - row, axis = 1 , ord = 2)

                vecinos_mas_cercanos = np.argsort(dist)
                labels_mas_cercanos = y_core[vecinos_mas_cercanos]
                        
                """
                Corregimos por gradiente usando distancia cuadratica como error
                """
                
                label_correcta = y_train[sample]
                
                
                for peso in range(labels_mas_cercanos.shape[0]):
                    label_del_vecino = labels_mas_cercanos[peso]
                    """
                    Si era correcta se corrige de cierta forma, y si no de otra
                    """
                    if label_del_vecino == label_correcta:
                        gradiente_promedio[peso] +=  self.weight[peso]**2
    
                    else:
                        gradiente_promedio[peso] -=  self.weight[peso]**2
            
            
        gradiente_promedio = gradiente_promedio/ (1.0/batch)
        gradiente_promedio /= np.linalg.norm( gradiente_promedio , ord = 2)
        gradiente_promedio *= delta
        
        self.weight += gradiente_promedio
        
        """
        Normalizado
        """
        self.weight /= np.linalg.norm(self.weight , ord = 2)

            
        print(self.weight[:10])
    
    def optimize_weights2(self):
        batch = self.X.shape[0]
        porcentaje = np.zeros(self.X.shape[0])
        
        for sample in range(batch):
            X_core, X_train, y_core, y_train = train_test_split(self.X, self.y,test_size = 1)

            print "Sample " + str(sample)
            row = X_train[0]
            dist = np.linalg.norm(X_core - row, axis = 1 , ord = 2)

            vecinos_mas_cercanos = np.argsort(dist)
            labels_mas_cercanos = y_core[vecinos_mas_cercanos]
            
            label_correcta = y_train[0]
            
            for prediction in range(len(labels_mas_cercanos)):
                label_del_vecino = labels_mas_cercanos[prediction]

                if label_del_vecino == label_correcta:
                    porcentaje[prediction] += 1.0

            
        porcentaje/= batch
        
        self.weight = porcentaje


        
            
    
    def predict(self, X, y=None):
        ret = np.zeros(X.shape[0])
        for k, row in enumerate(X):
            print k
            ret[k] = self._predict_row(row)
        return ret





