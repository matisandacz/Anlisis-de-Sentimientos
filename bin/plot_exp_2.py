import numpy as np
import matplotlib.pyplot as plt

fin = open("accuracy2.npy","rb")
M = np.load(fin)
fin.close()

print(M.shape)
for i in range(M.shape[0]):
  plt.plot(np.arange(M.shape[1])[:1000],M[i,:1000])
  plt.show()
