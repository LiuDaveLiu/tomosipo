import tomosipo as ts
import numpy as np
from medpy.io import load
import matplotlib.pyplot as plt
import torch
import scipy

from registration import ProjectionMatrix
from registration import helper_functions
from registration import normxcorr2
from registration import DownweightingMap

base_path='C:/Users/lliu10/OneDrive - Inside MD Anderson/siemenproject/data/'
mage_data, image_header = load(base_path+'17_1.mha')

data=np.load(base_path+'SiemensCiosSpin_Siemens.npz')
fluro, fh = load(base_path+'43_1')

fluro2=np.float32(np.mean(fluro,axis=2))
fluro2=scipy.ndimage.zoom(fluro2, 0.25, order=1) # resample 
plt.imshow(fluro2,cmap=plt.cm.gray)
plt.show()
#%% projection matrix
K,R,t=ProjectionMatrix.decompose(data['ProjectionMatrices'][0])
K1,R1,t1=ProjectionMatrix.decompose(data['ProjectionMatrices'][100])
#%% mage data threshold
for i in range(np.shape(mage_data)[0]):
    mage_data[i,:,:]=helper_functions.thr_image(mage_data[i,:,:],0,1700)
#%% mage data pad
pad=128
mage_data_pad=np.zeros((np.shape(mage_data)[0]+2*pad,np.shape(mage_data)[1]+2*pad,np.shape(mage_data)[2]+2*pad))
for i in range(np.shape(mage_data)[0]):
    mage_data_pad[i+pad,:,:]=helper_functions.add_pad(mage_data[i,:,:],pad,pad)
for i in range(np.shape(mage_data_pad)[1]):
    mage_data_pad[:,i,:]=helper_functions.add_pad(mage_data_pad[pad:-pad,i,:],0,pad)
#%%
mage_data=torch.from_numpy(mage_data)
vg = ts.volume(shape=np.shape(mage_data), size=(1, 1, 1))

# def drr_generate(r1,r2,r3,t1,t2,t3):
def drr_generate(r1,r2,r3):
    R1 = ts.rotate(pos=0, axis=(1, 0, 0), angles=r1)
    R2 = ts.rotate(pos=0, axis=(0, 1, 0), angles=r2)
    R3 = ts.rotate(pos=0, axis=(0, 0, 1), angles=r3)
    # T = ts.translate((t1, t2, t3))
    T = ts.translate((0, 0, 0))
      
    vg_rot=R1*R2*R3*T*vg.to_vec()
    pg = ts.cone(angles=1, shape=np.shape(fluro2), size=(2, 2), src_orig_dist=5, src_det_dist=10)
    
    A = ts.operator(vg_rot, pg)
    
    y=np.squeeze(A(mage_data))
    
    return y
#%%
# y = drr_generate(solutions[0][0][0], solutions[0][0][1], solutions[0][0][2], solutions[0][0][3], solutions[0][0][4], solutions[0][0][5])
y = drr_generate(np.pi, np.pi, 0)
plt.imshow(y,cmap='gray') # first projection
plt.axis('off')
#%% similarity matrix
dx1, dy1 = DownweightingMap.getGradientXY(fluro2)    
dx2, dy2 = DownweightingMap.getGradientXY(np.float32(y))

gc = -(np.sum(normxcorr2.normxcorr2(dx1, dx2,'same'))+np.sum(normxcorr2.normxcorr2(dy1, dy2,'same')))/2
#%%
angles=np.arange(0.2*np.pi,9)
for i in range(9):
    plt.subplot(3, 3, i+1)
    y = drr_generate(np.pi, angles[i], 0)
    dx1, dy1 = DownweightingMap.getGradientXY(fluro2)    
    dx2, dy2 = DownweightingMap.getGradientXY(np.float32(y))

    gc = -(np.sum(normxcorr2.normxcorr2(dx1, dx2,'same'))+np.sum(normxcorr2.normxcorr2(dy1, dy2,'same')))/2
    plt.imshow(y,cmap='gray') # first projection
    plt.title(np.round(gc))
    plt.axis('off')
# gradient image

    img=DownweightingMap.downweightMap(y[:, i, :])
    plt.subplot(3, 3, i+1)
    plt.imshow(img,cmap=plt.cm.gray)
    plt.axis('off')
#%%
# cor = signal.correlate2d(sx, sy)
cor = normxcorr2.normxcorr2(y[:, i, :], fluro,'same')
#%% optimize
def quadratic(x1, x2):
    return -(1-((x1 - 3) ** 2 + (10 * (x2 + 2)) ** 2))

if __name__ == "__main__":
    optimizer = CMA(mean=np.zeros(2), sigma=1.3)

    for generation in range(50):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = quadratic(x[0], x[1])
            solutions.append((x, value))
            print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
        optimizer.tell(solutions)