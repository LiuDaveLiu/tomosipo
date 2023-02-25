import tomosipo as ts
import numpy as np
from medpy.io import load
import matplotlib.pyplot as plt
import torch
from cmaes import CMA
import scipy

# from registration import ProjectionMatrix
from registration import helper_functions
from registration import normxcorr2
from registration import DownweightingMap

base_path='C:/Users/lliu10/OneDrive - Inside MD Anderson/siemenproject/data/'
mage_data, image_header = load(base_path+'17_1.mha')

data=np.load(base_path+'SiemensCiosSpin_Siemens.npz')
fluro, fh = load(base_path+'43_1')

# bone threshold
for i in range(np.shape(mage_data)[0]):
    mage_data[i,:,:]=helper_functions.thr_image(mage_data[i,:,:],0,1700)

# gpu
mage_data=torch.from_numpy(mage_data)
vg = ts.volume(shape=np.shape(mage_data), size=(1, 1, 1))

fluro2=np.float32(np.mean(fluro,axis=2))
fluro2=scipy.ndimage.zoom(fluro2, 0.25, order=1) # resample 
dx1, dy1 = DownweightingMap.getGradientXY(fluro2)
plt.imshow(fluro2,cmap=plt.cm.gray)

def drr(r1,r2,r3,t1,t2,t3):
    R1 = ts.rotate(pos=0, axis=(1, 0, 0), angles=r1)
    R2 = ts.rotate(pos=0, axis=(0, 1, 0), angles=r2)
    R3 = ts.rotate(pos=0, axis=(0, 0, 1), angles=r3)
    T = ts.translate((t1, t2, t3))
      
    vg_rot=R1*R2*R3*T*vg.to_vec()
    pg = ts.cone(angles=1, shape=(488, 488), size=(2, 2), src_orig_dist=5, src_det_dist=10)
    
    A = ts.operator(vg_rot, pg)
    
    y=np.squeeze(A(mage_data))
    
    dx2, dy2 = DownweightingMap.getGradientXY(np.float32(y))

    gc = -(np.sum(normxcorr2.normxcorr2(dx1, dx2,'same'))+np.sum(normxcorr2.normxcorr2(dy1, dy2,'same')))/2
    return gc

#%% optimize
if __name__ == "__main__":
    bounds = np.array([[0, 2*np.pi], [0, 2*np.pi], [0, 2*np.pi], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]])

    mean = np.array([np.pi, np.pi, np.pi, 0, 0, 0])
    sigma = 2

    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = drr(x[0], x[1], x[2], x[3], x[4], x[5])
            solutions.append((x, value))
        print(f"#{generation} {value}")
        optimizer.tell(solutions)
#%%
def drr_generate(r1,r2,r3,t1,t2,t3):
    R1 = ts.rotate(pos=0, axis=(1, 0, 0), angles=r1)
    R2 = ts.rotate(pos=0, axis=(0, 1, 0), angles=r2)
    R3 = ts.rotate(pos=0, axis=(0, 0, 1), angles=r3)
    T = ts.translate((t1, t2, t3))
      
    vg_rot=R1*R2*R3*T*vg.to_vec()
    pg = ts.cone(angles=1, shape=(488, 488), size=(2, 2), src_orig_dist=5, src_det_dist=10)
    
    A = ts.operator(vg_rot, pg)
    
    y=np.squeeze(A(mage_data))
    
    return y

y = drr_generate(solutions[0][0][0], solutions[0][0][1], solutions[0][0][2], solutions[0][0][3], solutions[0][0][4], solutions[0][0][5])
plt.imshow(y,cmap=plt.cm.gray)
drr(solutions[0][0][0], solutions[0][0][1], solutions[0][0][2], solutions[0][0][3], solutions[0][0][4], solutions[0][0][5])