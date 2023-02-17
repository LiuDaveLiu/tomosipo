import tomosipo as ts
import numpy as np
from scipy import ndimage
from scipy import misc
from medpy.io import load
import matplotlib.pyplot as plt
import torch
import time
from registration import ProjectionMatrix
from registration import helper_functions
from registration import normxcorr2
from registration import DownweightingMap

base_path='C:/Users/lliu10/OneDrive - Inside MD Anderson/siemenproject/data/'
mage_data, image_header = load(base_path+'17_1.mha')

data=np.load(base_path+'SiemensCiosSpin_Siemens.npz')
fluro, fh = load(base_path+'43_1')
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
R = ts.rotate(pos=0, axis=(1, 0, 0), angles=np.pi)
# R1 = ts.rotate(pos=0, axis=(0, 1, 0), angles=angles)
# R2 = ts.rotate(pos=0, axis=(0, 0, 1), angles=angles)
# T = ts.translate((0, 0, 1))

vg = ts.volume(shape=np.shape(mage_data), size=(1, 1, 1))
pg = ts.cone(angles=1, shape=(512, 512), size=(2, 2), src_orig_dist=5, src_det_dist=10)
# vg = ts.volume(shape=np.shape(mage_data_pad), size=(1, 1, 1))
# pg = ts.cone(angles=9, shape=(512, 512), size=(1, 1), src_orig_dist=5, src_det_dist=10)
svg = ts.svg(vg, pg)
svg.save("intro_forward_projection_geometries_cone.svg")
A = ts.operator(vg, pg)
#%%
start_time = time.time()
y=A(mage_data)
# y=A(mage_data_pad)
print("--- %s seconds ---" % (time.time() - start_time))
#%% gpu
mage_data_gpu=torch.from_numpy(mage_data)
# mage_data_gpu=torch.from_numpy(mage_data_pad)
start_time = time.time()
y=A(mage_data_gpu)
print("--- %s seconds ---" % (time.time() - start_time))
#%% similarity matrix

#%%
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(y[:, i, :],cmap='gray') # first projection
    plt.axis('off')
#%% gradient image
for i in range(9):
    img=DownweightingMap.downweightMap(y[:, i, :])
    plt.subplot(3, 3, i+1)
    plt.imshow(img,cmap=plt.cm.gray)
    plt.axis('off')
#%%
# cor = signal.correlate2d(sx, sy)
cor = normxcorr2.normxcorr2(y[:, i, :], fluro,'same')
#%%
fluro2=misc.imresize(np.float32(np.mean(fluro,axis=2)),np.shape(y[:,i,:]))
plt.imshow(fluro2,cmap=plt.cm.gray)
plt.show()
#%%
img=DownweightingMap.downweightMap(fluro2[::2,::2])
plt.imshow(img,cmap=plt.cm.gray)
plt.axis('off')
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