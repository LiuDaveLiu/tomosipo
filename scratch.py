import tomosipo as ts
import numpy as np
from scipy import ndimage
from scipy import signal
from medpy.io import load
import matplotlib.pyplot as plt
import torch
import time
from registration import ProjectionMatrix
from registration import helper_functions

base_path='C:/Users/lliu10/OneDrive - Inside MD Anderson/siemenproject/data/'
mage_data, image_header = load(base_path+'17_1.mha')

data=np.load(base_path+'SiemensCiosSpin_Siemens.npz')
#%% projection matrix
K,R,t=ProjectionMatrix.decompose(data['ProjectionMatrices'][0])
K1,R1,t1=ProjectionMatrix.decompose(data['ProjectionMatrices'][100])
#%% mage data threshold
for i in range(np.shape(mage_data)[0]):
    mage_data[i,:,:]=helper_functions.thr_image(mage_data[i,:,:],-100,1700)
#%% mage data pad
pad=128
mage_data_pad=np.zeros((np.shape(mage_data)[0]+2*pad,np.shape(mage_data)[1]+2*pad,np.shape(mage_data)[2]+2*pad))
for i in range(np.shape(mage_data)[0]):
    mage_data_pad[i+pad,:,:]=helper_functions.add_pad(mage_data[i,:,:],pad,pad)
for i in range(np.shape(mage_data_pad)[1]):
    mage_data_pad[:,i,:]=helper_functions.add_pad(mage_data_pad[pad:-pad,i,:],0,pad)
#%%
# vg = ts.volume(shape=np.shape(mage_data), size=(1, 1, 1))
# pg = ts.cone(angles=9, shape=(512, 512), size=(1.5, 1.5), src_orig_dist=5, src_det_dist=10)
vg = ts.volume(shape=np.shape(mage_data_pad), size=(1, 1, 1))
pg = ts.cone(angles=9, shape=(512, 512), size=(1, 1), src_orig_dist=5, src_det_dist=10)
svg = ts.svg(vg, pg)
svg.save("intro_forward_projection_geometries_cone.svg")
A = ts.operator(vg, pg)
#%%
start_time = time.time()
# y=A(mage_data)
y=A(mage_data_pad)
print("--- %s seconds ---" % (time.time() - start_time))
#%% gpu
# mage_data_gpu=torch.from_numpy(mage_data)
mage_data_gpu=torch.from_numpy(mage_data_pad)
start_time = time.time()
y=A(mage_data_gpu)
print("--- %s seconds ---" % (time.time() - start_time))
#%%
np.shape(y)
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(y[:, i, :],cmap='gray') # first projection
    plt.axis('off')
#%% gradient image
# Get x-gradient in "sx"
sx = ndimage.sobel(y[:, i, :],axis=0,mode='constant')
# Get y-gradient in "sy"
sy = ndimage.sobel(y[:, i, :],axis=1,mode='constant')
# Get square root of sum of squares
sobel=np.hypot(sx,sy)

plt.imshow(sx,cmap=plt.cm.gray)
plt.show()
plt.imshow(sy,cmap=plt.cm.gray)
plt.show()
# Hopefully see some edges
plt.imshow(sobel,cmap=plt.cm.gray)
plt.show()
