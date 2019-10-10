import pickle
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

fin = open("resultados_exp1.pkl","rb")
resultados = list(pickle.load(fin))
fin.close()
fin2 = open("resultados_exp1_var.pkl","rb")
variancia_ninguna = pickle.load(fin2)
fin.close()

print("Varias ejecuciones sin niguna mejora para K = 50 y ALPHA = 100")
print(variancia_ninguna)

x = PrettyTable()

colLabels = ["Ninguna","Binario","Negaciones","Norma pesada","IDF","Stop words"]
x.field_names = ["","K = 150 sin PCA","K = 50 ALPHA = 100","K = 300 ALPHA = 100","K = 50 ALPHA = 400","K = 300 ALPHA = 400"]

for i in range(len(resultados)):
    x.add_row([""]*6)
    x.add_row([colLabels[i]]+resultados[i])

print(x)
