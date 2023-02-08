import numpy as np
import matplotlib.pyplot as plt

base_path='C:/Users/lliu10/OneDrive - Inside MD Anderson/siemenproject/data/'
data=np.load(base_path+'SiemensCiosSpin_Siemens.npz')

plt.plot(data['ProjectionMatrices'][0])
plt.show()

#%%