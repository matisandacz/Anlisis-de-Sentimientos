import numpy as np
import matplotlib.pyplot as plt
import pickle

fin = open("resultados_exp5.pkl","rb")
M = pickle.load(fin)
fin.close()

print(M)
#print(M.shape)
#for i in range(M.shape[0]):
#  plt.plot(np.arange(M.shape[1])[:1000],M[i,:1000])
#  plt.show()
