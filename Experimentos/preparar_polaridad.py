from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import sys

FIN = open(sys.argv[1],"rb")
MIN_DF = float(sys.argv[2])

FOUT = open("polaridad_MDF_"+str(MIN_DF)+".pkl","wb")
X_train,y_train = pickle.load(FIN)
FIN.close()

def get_polarity(ngramvocab,vocab,polaridad):
    polaridades = []
    print(len(ngramvocab))
    print(len(vocab))
    total = len(ngramvocab)
    i = 0
    for ngram in ngramvocab:
        ngrampolaridad = 0
        i+=1
        if i % 10000 == 0:
            print("%"+str((i / total)*100))
        for word in ngram.split(" "):
            try:
                indice = vocab.index(word)
            except ValueError: #si una palabra no esta en el vocabulario la ignoro
                continue
            ngrampolaridad += np.abs(polaridad[indice])
        ngrampolaridad /= len(ngram.split(" "))
        polaridades.append(ngrampolaridad)
    return polaridades

fvocab = open("../data/aclImdb/imdb.vocab","r")
fpolaridad = open("../data/aclImdb/imdbEr.txt","r")
vocab = list(map(lambda x : x[:-1],list(fvocab))) #elimino \n del final de las palabras
polaridad = list(map(float,list(fpolaridad)))
fvocab.close()
fpolaridad.close()

#Obtengo el vocabulario completo con ngramas
ngrams = TfidfVectorizer(ngram_range=(1,3), norm=None, smooth_idf=False, use_idf=False, min_df = MIN_DF)
ngrams.fit(X_train)
ngramvocab = ngrams.get_feature_names()
ngrampolaridad = get_polarity(ngramvocab,vocab,polaridad)

pickle.dump((ngramvocab,ngrampolaridad),FOUT)
